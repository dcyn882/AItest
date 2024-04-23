#! /bin/bash
#/*******
##add by weego at 2023-07-24
#1、增加CPU加载模型的选择，参数化配置
#2、增加模型的类型选择，参数化配置
#3、新增支持动态lora权重加载，参数化配置
#4、支持GPU加载的数量设定
#*******/
export PROJ_HOME=$PWD
export KMP_DUPLICATE_LIB_OK=TRUE

# 设置模型类型和路径
model_type=auto
model_path=/models/Qwen1.5-7B-Chat

# 生成实验名称和结果输出路径
exp_name=Qwen1.5-7B-Chat
exp_date=$(date +"%Y%m%d%H%M%S")
output_path=$PROJ_HOME/output_dir/${exp_name}/$exp_date

echo "exp_date: $exp_date"
echo "output_path: $output_path"

# 定义choices选项，示例选项，根据实际情况调整
choices="A B C D AB AC AD BC BD CD ABC ABD BCD ABCD"

# 运行Python脚本
python eval.py \
    --model_type ${model_type} \
    --model_path ${model_path} \
    ${lora_model:+--lora_model "$lora_model"} \
    --cot False \
    --few_shot True \
    --with_prompt False \
    --ntrain 5 \
    --constrained_decoding True \
    --temperature 0.2 \
    --n_times 1 \
    --do_save_csv True \
    --do_test False \
    --gpus 0 \
    --only_cpu False \
    --output_dir ${output_path} \
    --choices $choices
