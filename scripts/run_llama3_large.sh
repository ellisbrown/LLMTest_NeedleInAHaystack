#!/bin/sh

export NIAH_EVALUATOR_API_KEY=$OPENAI_API_KEY
MODEL=meta-llama/Meta-Llama-3-8B-Instruct

source venv/bin/activate

python -m needlehaystack.run \
    --provider llama3 \
    --model_name $MODEL \
    --results_dir ./results/linear_35x35_llama_3-8b-instruct \
    --context_lengths_min 128 \
    --context_lengths_max 8192

