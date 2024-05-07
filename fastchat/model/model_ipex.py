import gc
from threading import Thread
import os
import re
import torch
from transformers import TextIteratorStreamer

def get_int_from_env(env_keys, default):
    """Returns the first positive env value found in the `env_keys` list or the default."""
    for e in env_keys:
        val = int(os.environ.get(e, -1))
        if val >= 0:
            return val
    return default

local_rank = get_int_from_env(["LOCAL_RANK", "MPI_LOCALRANKID"], 0)
world_size = get_int_from_env(["WORLD_SIZE", "PMI_SIZE"], 1)
    
@torch.inference_mode()
def generate_stream_ipex(
    model,
    tokenizer,
    params,
    device,
    context_len=8192,
    stream_interval=2,
    judge_sent_end=False,
):
    prompt = params["prompt"]
    repetition_penalty = float(params.get("repetition_penalty", 1.0))

    temperature = float(params.get("temperature", 1.0))
    max_new_tokens = int(params.get("max_new_tokens", 4096))
    
    echo = params.get("echo", True)
    
    inputs = tokenizer(
        prompt, return_tensors="pt", padding=model.config.padding
    ).input_ids
    input_echo_len = len(inputs[0])
    max_len = max_new_tokens + input_echo_len

    decode_config = dict(skip_special_tokens=True, clean_up_tokenization_spaces=True)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, **decode_config)
        
    generate_kwargs = dict(
        do_sample=True, 
        temperature=temperature, 
        num_beams=model.config.num_beams,
        num_return_sequences=model.config.num_return_sequences, 
        early_stopping=model.config.early_stopping,
        max_length=max_len, 
        length_penalty=repetition_penalty, 
        # eos_token_id=model.config.eos_token_id,
        # pad_token_id=model.config.pad_token_id,
        streamer=streamer
    )

    if re.search("t5", model.model.config.architectures[0], re.IGNORECASE):
        generate_kwargs["max_length"] = generate_kwargs["max_new_tokens"]
        generate_kwargs.pop("max_new_tokens")
    
    with torch.inference_mode(), torch.no_grad(), torch.cpu.amp.autocast(
        enabled=True
    ):  
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        generate_kwargs["input_ids"] = input_ids
        thread = Thread(target=model.model.generate, kwargs=generate_kwargs)
        thread.start()
        if echo:
            # means keep the prompt
            output = prompt
        else:
            output = ""
        i = 0
        for i, new_text in enumerate(streamer):
            output += new_text
            yield {
                "text": output,
                "usage": {
                    "prompt_tokens": input_echo_len,
                    "completion_tokens": i,
                    "total_tokens": input_echo_len + i,
                },
                "finish_reason": None,
            }
        output = output.strip()
        if i == max_new_tokens - 1:
            finish_reason = "length"
        else:
            finish_reason = "stop"
        yield {
            "text": output,
            "usage": {
                "prompt_tokens": input_echo_len,
                "completion_tokens": i,
                "total_tokens": input_echo_len + i,
            },
            "finish_reason": finish_reason,
        }
        gc.collect()

