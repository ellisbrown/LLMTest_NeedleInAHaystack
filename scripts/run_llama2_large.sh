#!/bin/sh

export NIAH_EVALUATOR_API_KEY=$OPENAI_API_KEY
MODEL=meta-llama/Llama-2-7b-chat-hf
OUTPUT_DIR=./results/linear_35x35_llama_2-7b-chat-hf
MAX_CTX_LEN=4096

source venv/bin/activate

python -m needlehaystack.run \
    --provider llama2 \
    --model_name $MODEL \
    --results_dir $OUTPUT_DIR \
    --context_lengths_min 128 \
    --context_lengths_max $MAX_CTX_LEN \

# generate plot
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
python $DIR/gen_plot.py \
    --folder_path $OUTPUT_DIR \
    --model_name Llama-2-7b-chat-hf \
    --max_ctx_len $MAX_CTX_LEN
