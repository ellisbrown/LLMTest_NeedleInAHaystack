#!/bin/sh

export NIAH_EVALUATOR_API_KEY=$OPENAI_API_KEY
MODEL=meta-llama/Llama-2-7b-chat-hf

source venv/bin/activate

python -m needlehaystack.run \
    --provider local \
    --model_name $MODEL \
    --results_dir ./results/linear_10x10_llama_2-7b-chat-hf \
    --num_concurrent_requests 10 \
    --document_depth_percents "[10,20,30,40,50,60,70,80,90,100]" \
    --context_lengths "[1963,2389,2816,3243,3669,4096]"
    # --context_lengths "[683,1109,1536,1963,2389,2816,3243,3669,4096]"
