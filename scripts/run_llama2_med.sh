#!/bin/sh

export NIAH_EVALUATOR_API_KEY=$OPENAI_API_KEY
MODEL=meta-llama/Llama-2-7b-chat-hf

MIN_CTX_LEN=128
MAX_CTX_LEN=4096
N_CTX=17
N_DEPTH=17
OUTPUT_DIR=./results/linear_${MIN_CTX_LEN}-${MAX_CTX_LEN}_${N_CTX}x${N_DEPTH}_llama_2-7b-chat-hf

source venv/bin/activate
python -m needlehaystack.run \
    --provider llama2 \
    --model_name $MODEL \
    --results_dir $OUTPUT_DIR \
    --context_lengths_min $MIN_CTX_LEN \
    --context_lengths_num_intervals $N_CTX \
    --document_depth_percent_intervals $N_DEPTH \
    --context_lengths_max $MAX_CTX_LEN

# generate plot
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
python $DIR/gen_plot.py \
    --folder_path $OUTPUT_DIR \
    --model_name Llama-2-7b-Chat-hf \
    --max_ctx_len $MAX_CTX_LEN
