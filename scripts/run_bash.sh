#!/bin/bash

SCRIPT_DIR="$(dirname "$0")"
WORK_DIR="$(realpath "$SCRIPT_DIR/..")"

export PYTHONPATH="${WORK_DIR}:$PYTHONPATH"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ENABLE_HF_TRANSFER=1

# config for thinker engine
thinker_model_name_or_path="Qwen/QwQ-32B"
thinker_name="thinker"
thinker_gpu_ids="0,1,2,3"
thinker_port="8009"
thinker_uri="http://localhost:${thinker_port}/v1"

# config for summarizer engine
summarizer_model_name_or_path="Qwen/Qwen2.5-7B-Instruct"
summarizer_name="summarizer"
summarizer_gpu_ids="4,5"
summarizer_port="8010"
summarizer_uri="http://localhost:${summarizer_port}/v1"

# LOG_DIR
LOG_DIR="${WORK_DIR}/logs"
mkdir -p "${LOG_DIR}"

# thinker engine
CUDA_VISIBLE_DEVICES=${thinker_gpu_ids} \
nohup python -m vllm.entrypoints.openai.api_server \
  --port ${thinker_port} \
  --served-model-name ${thinker_name} \
  --model ${thinker_model_name_or_path} \
  --tensor-parallel-size $(echo $thinker_gpu_ids | awk -F, '{print NF}') \
  --gpu-memory-utilization 0.95 \
  --max-model-len 32768 \
  --max-num-batched-tokens 131072 \
  --max-num-seqs 64 \
  --disable-log-requests \
> "${LOG_DIR}/thinker_engine.log" 2>&1 &

# summarizer engine
CUDA_VISIBLE_DEVICES=${summarizer_gpu_ids} \
nohup python -m vllm.entrypoints.openai.api_server \
  --port ${summarizer_port} \
  --served-model-name ${summarizer_name} \
  --model ${summarizer_model_name_or_path} \
  --tensor-parallel-size $(echo $summarizer_gpu_ids | awk -F, '{print NF}') \
  --gpu-memory-utilization 0.95 \
  --max-model-len 32768 \
  --max-num-batched-tokens 131072 \
  --max-num-seqs 64 \
  --disable-log-requests \
> "${LOG_DIR}/summarizer_engine.log" 2>&1 &

# Wait for vLLM servers to be fully up
echo "Waiting for vLLM servers to start..."
for PORT in ${thinker_port} ${summarizer_port}; do
  while ! curl -s http://localhost:${PORT}/health > /dev/null; do
    sleep 1  # Check every second
  done
done
echo "vLLM servers have started, proceeding with the next steps..."

python ${WORK_DIR}/src/bash_cli.py \
  --thinker_name ${thinker_name} \
  --thinker_uri ${thinker_uri} \
  --thinker_tokenizer_path ${thinker_model_name_or_path} \
  --summarizer_name ${summarizer_name} \
  --summarizer_uri ${summarizer_uri} \
  --summarizer_tokenizer_path ${summarizer_model_name_or_path}

# Stop Servers
pkill -f "vllm.entrypoints.openai.api_server"
echo "vllm server has been stopped."