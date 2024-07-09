#!/bin/sh

export NIAH_EVALUATOR_API_KEY=$OPENAI_API_KEY
MODEL=meta-llama/Meta-Llama-3-8B-Instruct
OUTPUT_DIR=./results/passkey_35x35_llama3-8b-instruct
MAX_CTX_LEN=8192

source venv/bin/activate

python -m needlehaystack.run \
    --provider llama3 \
    --model_name $MODEL \
    --results_dir $OUTPUT_DIR \
    --context_lengths_min 128 \
    --context_lengths_max $MAX_CTX_LEN \
    --haystack_dir "PassKey" \
    --needle "The pass key is 9054. Remember it. 9054 is the pass key." \
    --retrieval_question "What is the pass key? The pass key is" \
    --results_dir ./results/passkey_llama3-8b-instruct_test


# generate plot
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
python $DIR/gen_plot.py \
    --folder_path $OUTPUT_DIR \
    --model_name Llama-3-8b-Instruct \
    --max_ctx_len $MAX_CTX_LEN
