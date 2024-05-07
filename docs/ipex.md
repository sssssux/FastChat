# Integration of Intel Extension for Pytorch

## Install xFasterTransformer

Setup environment (please refer to [this link](https://github.com/intel/intel-extension-for-pytorch/tree/v2.2.0%2Bcpu/examples/cpu/inference/python/llm) for more details):

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install intel-extension-for-pytorch pyyaml deepspeed
python -m pip install oneccl_bind_pt --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/cpu/us/
```


Chat with the CLI:
```bash
#run inference on all CPUs and using bfloat16
python3 -m fastchat.serve.cli \
    --model-path /path/to/models \
    --ipex \
    --ipex-dtype bfloat16
```
or with numactl on multi-socket server for better performance
```bash
#run inference on numanode 0 and with data type bf16_fp16 (first token uses bfloat16, and rest tokens use float16)
numactl -N 0  -m 0 \
python3 -m fastchat.serve.cli \
    --model-path /path/to/models/llama2-7b/ \
    --ipex \
    --ipex-dtype bfloat16
```
or using MPI to run inference on 2 sockets for better performance
```bash
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so # Intel OpenMP
# Tcmalloc is a recommended malloc implementation that emphasizes fragmentation avoidance and scalable concurrency support.
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so

#run inference with deepspeed auto tp on multi-socket cpu system and with data type bf16 
OMP_NUM_THREADS=$CORE_NUM_PER_SOCKET deepspeed --bind_cores_to_rank  --num_gpus $NUM_SOCKET /path/to/FastChat/fastchat/serve/cli.py \
    --model-path /path/to/models/llama2-7b/ \
    --ipex \
    --ipex-dtype bfloat16
```
