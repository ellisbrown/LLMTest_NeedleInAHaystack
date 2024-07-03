#!/bin/sh

export NIAH_EVALUATOR_API_KEY=$OPENAI_API_KEY
MODEL=meta-llama/Meta-Llama-3-8B-Instruct

source venv/bin/activate

python -m needlehaystack.run \
    --provider llama3 \
    --model_name $MODEL \
    --document_depth_percents "[50]" \
    --context_lengths "[2048]" \
    --results_dir ./results/test_llama3-8b-ins

