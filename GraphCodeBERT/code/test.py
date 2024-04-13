import os

os.system("CUDA_VISIBLE_DEVICES=0 python run.py \
        --output_dir=saved_models_nofreeze/MutantEq \
        --config_name=../../graphcodebert-base \
        --model_name_or_path=../../graphcodebert-base \
        --tokenizer_name=../../graphcodebert-base \
        --requires_grad 0 \
        --do_test \
        --code_db_file=../../dataset/MutantBench_code_db_java.csv \
        --train_data_file=../../dataset/Mutant_A_hierarchical.csv \
        --eval_data_file=../../dataset/Mutant_B_hierarchical.csv \
        --test_data_file=../../dataset/Mutant_B_hierarchical.csv 2>&1")