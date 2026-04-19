#!/bin/bash
model=DeepSeek-R1-Distill-Qwen-1.5B
data=aime_25

MODEL_BASE_PATH=./grid_search/${model}_${data}
GRID_SEARCH_PATH=$MODEL_BASE_PATH/grid_search_result.jsonl
TEST_SAVE_PATH=$MODEL_BASE_PATH/test_result

TEST_DATA=./model_embeds/${model}_${data}
echo "Running test_predictor_with_class_weights.py with model $MODEL_BASE_PATH and data $TEST_DATA"
python -u ./src/test_predictor_with_class_weights.py \
    --threshold 0.5 \
    --test_data_dir $TEST_DATA \
    --grid_search_result_path $GRID_SEARCH_PATH \
    --topk 10 \
    --save_path $TEST_SAVE_PATH \
    --model_name $model