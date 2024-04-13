import os

os.system("python run.py \
    --api_key= \
    --output_dir=../saved_models/ \
    --model_type=gpt-4-0613 \
    --do_test \
    --test_type=zero-shot-prompt\
    --codebase_data_file=../../dataset/MutantBench_code_db_java.csv \
    --test_data_file=../../dataset/Mutant_B_hierarchical.csv \
    --seed 0")