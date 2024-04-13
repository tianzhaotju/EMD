import os

os.system("python run.py \
    --output_dir=../saved_models/ \
    --model_type=codellama \
    --model_name_or_path=codellama/CodeLlama-7b-Instruct-hf \
    --tokenizer_name=codellama/CodeLlama-7b-Instruct-hf \
    --do_train \
    --codebase_data_file=../../dataset/MutantBench_code_db_java.csv \
    --train_data_file=../../dataset/Mutant_A_hierarchical.csv \
    --test_data_file=../../dataset/Mutant_B_hierarchical.csv \
    --epoch 10 \
    --gradient_accumulation_steps 4\
    --train_batch_size 2 \
    --learning_rate 5e-5 \
    --seed 0")