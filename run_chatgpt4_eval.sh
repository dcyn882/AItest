#! /bin/bash
export PROJ_HOME=$PWD
export KMP_DUPLICATE_LIB_OK=TRUE




exp_name=chatgpt-4.0
exp_date=$(date +"%Y%m%d%H%M%S")
output_path=$PROJ_HOME/output_dir/${exp_name}/$exp_date

echo "exp_date": $exp_date
echo "output_path": $output_path

python eval_chatgpt.py \
    --openai_key ${OPENAI_API_KEY} \
    --cot False \
    --few_shot False \
    --n_times 1 \
    --ntrain 5 \
    --do_test False \
    --do_save_csv True \
    --output_dir ${output_path} \
    --model_name gpt-4
