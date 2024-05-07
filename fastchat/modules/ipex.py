from dataclasses import dataclass
import sys
import os
import re
from pathlib import Path
import json
import deepspeed
import torch
from huggingface_hub import snapshot_download
from transformers.utils import is_offline_mode
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    T5ForConditionalGeneration,
    AutoProcessor
) 
@dataclass
class IpexConfig:
    max_length: int = 4096
    num_beams: int = 1
    eos_token_id: int = -1
    pad_token_id: int = -1
    num_return_sequences: int = 1
    is_encoder_decoder: bool = False
    padding: bool = False
    early_stopping: bool = False
    data_type: str = "bfloat16"
    weight_only_quantization: bool = False
    weight_dtype: str = "INT8"
    lowp_mode: str = "AUTO"
    local_rank: int = 0
    

class IpexModel:
    def __init__(self, ipex_model, ipex_config):
        self.model = ipex_model
        self.config = ipex_config

def get_int_from_env(env_keys, default):
    """Returns the first positive env value found in the `env_keys` list or the default."""
    for e in env_keys:
        val = int(os.environ.get(e, -1))
        if val >= 0:
            return val
    return default

local_rank = get_int_from_env(["LOCAL_RANK", "MPI_LOCALRANKID"], 0)
world_size = get_int_from_env(["WORLD_SIZE", "PMI_SIZE"], 1)

def print_rank0(*msg):
    if local_rank != 0:
        return
    print(*msg)

def load_ipex_model(model_path, ipex_config: IpexConfig):
    try:
        import intel_extension_for_pytorch as ipex
        torch._C._jit_set_texpr_fuser_enabled(False)
        if world_size > 1:
            
            from deepspeed.accelerator import get_accelerator
            import deepspeed.comm as dist
            deepspeed.init_distributed(get_accelerator().communication_backend_name())
            
    except ImportError as e:
        print(f"Error: Failed to load IPEX. {e}")
        sys.exit(-1)
    try:
        ipex._C.disable_jit_linear_repack()
    except Exception:
        pass
    
    MODEL_CLASSES = {
        "gpt-j": (AutoModelForCausalLM, AutoTokenizer),
        "gptj": (AutoModelForCausalLM, AutoTokenizer),
        "gpt-neox": (AutoModelForCausalLM, AutoTokenizer),
        "gptneox": (AutoModelForCausalLM, AutoTokenizer),
        "llama": (AutoModelForCausalLM, AutoTokenizer),
        "opt": (AutoModelForCausalLM, AutoTokenizer),
        "falcon": (AutoModelForCausalLM, AutoTokenizer),
        "chatglm": (AutoModelForCausalLM, AutoTokenizer),
        "bloom": (AutoModelForCausalLM, AutoTokenizer),
        "codegen": (AutoModelForCausalLM, AutoTokenizer),
        "baichuan2": (AutoModelForCausalLM, AutoTokenizer),
        "baichuan": (AutoModelForCausalLM, AutoTokenizer),
        "gptbigcode": (AutoModelForCausalLM, AutoTokenizer),
        "t5": (T5ForConditionalGeneration, AutoTokenizer),
        "mistral": (AutoModelForCausalLM, AutoTokenizer),
        "mixtral": (AutoModelForCausalLM, AutoTokenizer),
        "mpt": (AutoModelForCausalLM, AutoTokenizer),
        "stablelm": (AutoModelForCausalLM, AutoTokenizer),
        "qwen": (AutoModelForCausalLM, AutoTokenizer),
        "git": (AutoModelForCausalLM, AutoProcessor),
        "yuan": (AutoModelForCausalLM, AutoTokenizer),
        "auto": (AutoModelForCausalLM, AutoTokenizer),
    }
    # the Deepspeed team made these so it's super fast to load (~1 minute), rather than wait 10-20min loading time.
    tp_presharded_models = [
        "microsoft/bloom-deepspeed-inference-int8",
        "microsoft/bloom-deepspeed-inference-fp16",
    ]

    tp_presharded_mode = True if model_path in tp_presharded_models else False

    load_dtype = getattr(torch, ipex_config.data_type)
    print_rank0(f"*** Loading the model {model_path}")
    model_type = next(
        (x for x in MODEL_CLASSES.keys() if x in model_path.lower()), "auto"
    )
    model_class = MODEL_CLASSES[model_type]
    tokenizer = model_class[1].from_pretrained(model_path, trust_remote_code=True)
    
    config = AutoConfig.from_pretrained(
            model_path, torchscript=True, trust_remote_code=True
        )
    if re.search("falcon", config.architectures[0], re.IGNORECASE) or re.search(
        "rw", config.architectures[0], re.IGNORECASE
    ):
        model_type = "falcon"
    if re.search("gptbigcode", config.architectures[0], re.IGNORECASE):
        model_type = "gptbigcode"
    if re.search("gptneox", config.architectures[0], re.IGNORECASE):
        model_type = "gpt-neox"
    if re.search("yuan", config.architectures[0], re.IGNORECASE):
        model.config.batch_size = ipex_config.num_beams

    if model_type == "falcon":
        model_input_names = ["input_ids", "attention_mask"]
        tokenizer.model_input_names = model_input_names
    
    if not hasattr(config, "lm_head_generation"):
        config.lm_head_generation = True
    
    if world_size == 1 or model_type in ["falcon", "baichuan", "baichuan2", "gptbigcode", "git", "qwen", "yuan"]:
        model = model_class[0].from_pretrained(
            model_path,
            config=config,
            low_cpu_mem_usage=True,
            torch_dtype=load_dtype,
            trust_remote_code=True,
        )
    else: # Construct model with fake meta tensors, later will be replaced during ds-inference ckpt load
        with deepspeed.OnDevice(dtype=load_dtype, device="meta"):
            if  model_type in ["t5"]:
                model =  model_class[0](config=config)
            else:
                model = (
                    model_class[0].from_config(config, trust_remote_code=True).to(load_dtype)
                )
    
    model = model.eval()
    model = model.to(memory_format=torch.channels_last)
    ipex_woq_enabled = ipex_config.weight_only_quantization and (world_size > 1)
    
    if world_size > 1:
        # Model loading and instantiating on GPUs
        def get_repo_root(model_name_or_path):
            if os.path.exists(model_name_or_path):
                # local path
                return model_name_or_path
            # checks if online or not
            if is_offline_mode():
                print_rank0("Offline mode: forcing local_files_only=True")
            # download only on first process
            allow_patterns = ["*.bin", "*.model", "*.json", "*.txt", "*.py", "*LICENSE"]
            if local_rank == 0:
                snapshot_download(
                    model_name_or_path,
                    local_files_only=is_offline_mode(),
                    cache_dir=os.getenv("TRANSFORMERS_CACHE", None),
                    allow_patterns=allow_patterns,
                    # ignore_patterns=["*.safetensors"],
                )

            dist.barrier()

            return snapshot_download(
                model_name_or_path,
                local_files_only=is_offline_mode(),
                cache_dir=os.getenv("TRANSFORMERS_CACHE", None),
                allow_patterns=allow_patterns,
                # ignore_patterns=["*.safetensors"],
            )
            
        def get_checkpoint_files(model_name_or_path):
            cached_repo_dir = get_repo_root(model_name_or_path)

            # extensions: .bin | .pt
            # creates a list of paths from all downloaded files in cache dir
            file_list = [
                str(entry)
                for entry in Path(cached_repo_dir).rglob("*.[bp][it][n]")
                if entry.is_file()
            ]
            return file_list

        checkpoints_json = "checkpoints.json"

        def write_checkpoints_json():
            checkpoint_files = get_checkpoint_files(model_path)
            if local_rank == 0:
                # model.config.model_type.upper()
                data = {"type": "BLOOM", "checkpoints": checkpoint_files, "version": 1.0}
                json.dump(data, open(checkpoints_json, "w"))

        repo_root = get_repo_root(model_path)
        if tp_presharded_mode:
            # tp presharded repos come with their own checkpoints config file
            checkpoints_json = os.path.join(repo_root, "ds_inference_config.json")
        else:
            # for normal bloom repo we need to write the checkpoints config file
            write_checkpoints_json()
            dist.barrier()
        model = deepspeed.init_inference(
            model,
            mp_size=world_size,
            base_dir=repo_root,
            dtype=load_dtype,
            checkpoint=checkpoints_json,
        )
        model = model.module

        
        if ipex_woq_enabled:
            from intel_extension_for_pytorch.quantization import WoqWeightDtype
            if ipex_config.weight_dtype == "INT8":
                weight_dtype = WoqWeightDtype.INT8
            elif ipex_config.weight_dtype == "INT4":
                weight_dtype = WoqWeightDtype.INT4
            else:
                assert ipex_config.weight_dtype == "NF4"
                weight_dtype = WoqWeightDtype.NF4
            if ipex_config.lowp_mode == "INT8":
                lowp_mode = ipex.quantization.WoqLowpMode.INT8
            elif ipex_config.lowp_mode == "FP32":
                lowp_mode = ipex.quantization.WoqLowpMode.NONE
            elif ipex_config.lowp_mode == "FP16":
                lowp_mode = ipex.quantization.WoqLowpMode.FP16
            elif ipex_config.lowp_mode == "BF16":
                lowp_mode = ipex.quantization.WoqLowpMode.BF16
            else:  # AUTO
                if weight_dtype == WoqWeightDtype.INT4:
                    lowp_mode = ipex.quantization.WoqLowpMode.INT8
                else:
                    lowp_mode = ipex.quantization.WoqLowpMode.BF16

            act_quant_mode_dict = {
                "PER_TENSOR": ipex.quantization.WoqActQuantMode.PER_TENSOR,
                "PER_IC_BLOCK": ipex.quantization.WoqActQuantMode.PER_IC_BLOCK,
                "PER_BATCH": ipex.quantization.WoqActQuantMode.PER_BATCH,
                "PER_BATCH_IC_BLOCK": ipex.quantization.WoqActQuantMode.PER_BATCH_IC_BLOCK,
            }
            qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
                weight_dtype=weight_dtype,
                lowp_mode=lowp_mode,
                act_quant_mode=act_quant_mode_dict["PER_IC_BLOCK"],
                group_size=-1,
            )
    
    
    model = ipex.llm.optimize(
        model.eval(),
        dtype=load_dtype,
        quantization_config=qconfig if ipex_woq_enabled else None,
        inplace=True,
        deployment_mode=True,
    )
    
    print_rank0(f"Load model {model_type} done.")

    model = IpexModel(ipex_model=model, ipex_config=ipex_config)
    
    return model, tokenizer
