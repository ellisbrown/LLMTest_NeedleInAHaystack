#!/bin/sh

export NIAH_EVALUATOR_API_KEY=$OPENAI_API_KEY
MODEL=meta-llama/Meta-Llama-3-8B-Instruct

# MIN_CTX_LEN=4096
# MAX_CTX_LEN=16384
# N_CTX=48
MIN_CTX_LEN=6144
MAX_CTX_LEN=10240
N_CTX=17  # 16 + end
N_DEPTH=9  # 8 + end
OUTPUT_DIR=./results/long_passkey_ctx_${MIN_CTX_LEN}-${MAX_CTX_LEN}_${N_CTX}x${N_DEPTH}_llama_3-8b-instruct

source venv/bin/activate
python -m needlehaystack.run \
    --provider llama3 \
    --model_name $MODEL \
    --results_dir $OUTPUT_DIR \
    --context_lengths_min $MIN_CTX_LEN \
    --context_lengths_num_intervals $N_CTX \
    --document_depth_percent_intervals $N_DEPTH \
    --context_lengths_max $MAX_CTX_LEN \
    --haystack_dir "PassKey" \
    --needle "The pass key is 9054. Remember it. 9054 is the pass key." \
    --retrieval_question "What is the pass key? The pass key is"


# generate plot
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
python $DIR/gen_plot.py \
    --folder_path $OUTPUT_DIR \
    --model_name Llama-3-8b-Instruct \
    --max_ctx_len $MAX_CTX_LEN
