#!/bin/sh

export NIAH_EVALUATOR_API_KEY=$OPENAI_API_KEY
MODEL=meta-llama/Llama-2-7b-chat-hf

source venv/bin/activate

python -m needlehaystack.run \
    --provider llama2 \
    --model_name $MODEL \
    --document_depth_percents "[50]" \
    --context_lengths "[2000]"


