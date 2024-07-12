#!/bin/sh

export NIAH_EVALUATOR_API_KEY=$OPENAI_API_KEY
MODEL=meta-llama/Meta-Llama-3-8B-Instruct

MIN_CTX_LEN=512
MAX_CTX_LEN=8704  # 8192 + 512
N_CTX=33
N_DEPTH=33

OUTPUT_DIR=./results/linear_${N_CTX}x${N_DEPTH}_llama_3-8b-instruct

source venv/bin/activate
python -m needlehaystack.run \
    --provider llama3 \
    --model_name $MODEL \
    --results_dir $OUTPUT_DIR \
    --context_lengths_min $MIN_CTX_LEN \
    --context_lengths_num_intervals $N_CTX \
    --document_depth_percent_intervals $N_DEPTH \
    --context_lengths_max $MAX_CTX_LEN \

# generate plot
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
python $DIR/gen_plot.py \
    --folder_path $OUTPUT_DIR \
    --model_name Llama-3-8b-Instruct \
    --max_ctx_len $MAX_CTX_LEN
