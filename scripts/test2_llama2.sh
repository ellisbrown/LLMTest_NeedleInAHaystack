#!/bin/sh

export NIAH_EVALUATOR_API_KEY=$OPENAI_API_KEY
MODEL=meta-llama/Llama-2-7b-chat-hf

source venv/bin/activate

python -m needlehaystack.run \
    --provider llama2 \
    --model_name $MODEL \
    --retrieval_question "What is the best thing to do in San Francisco?" \
    --document_depth_percents "[50]" \
    --context_lengths "[2048]" \
    --results_dir ./results/test2_llama2-7b-chat-hf
