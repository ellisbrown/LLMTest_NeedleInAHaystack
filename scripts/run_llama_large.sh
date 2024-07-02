#!/bin/sh

export NIAH_EVALUATOR_API_KEY=$OPENAI_API_KEY
MODEL=meta-llama/Llama-2-7b-chat-hf

source venv/bin/activate

python -m needlehaystack.run \
    --provider local \
    --model_name $MODEL \
    --results_dir ./results/linear_35x35_llama_2-7b-chat-hf \
    --num_concurrent_requests 35 \
    --context_lengths_min 128 \
    --context_lengths_max 4096
