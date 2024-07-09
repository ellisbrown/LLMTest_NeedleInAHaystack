#!/bin/sh

export NIAH_EVALUATOR_API_KEY=$OPENAI_API_KEY
MODEL=meta-llama/Meta-Llama-3-8B-Instruct

source venv/bin/activate

python -m needlehaystack.run \
    --provider llama3 \
    --model_name $MODEL \
    --document_depth_percents "[50]" \
    --context_lengths "[2048]" \
    --haystack_dir "PassKey" \
    --needle "The pass key is 9054. Remember it. 9054 is the pass key." \
    --retrieval_question "What is the pass key? The pass key is" \
    --results_dir ./results/passkey_llama3-8b-instruct_test
