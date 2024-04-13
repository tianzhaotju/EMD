import os

os.system("python run.py \
    --api_key=\
    --output_dir=../saved_models/ \
    --model_type=gpt-3.5-turbo-0125 \
    --do_test \
    --test_type=few-shot-prompt\
    --codebase_data_file=../../dataset/MutantBench_code_db_java.csv \
    --train_data_file=../../dataset/Mutant_A_hierarchical.csv\
    --test_data_file=../../dataset/Mutant_B_hierarchical.csv \
    --seed 0")