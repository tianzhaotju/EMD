import os

os.system("python run.py \
        --output_dir=saved_models/MutantEq \
        --config_name=../../codet5p-6b \
        --model_name_or_path=../../codet5p-6b \
        --tokenizer_name=../../codet5p-6b \
        --requires_grad  0 \
        --main_gpu  1 \
        --vice_gpu  0 \
        --do_test \
        --code_db_file=../../dataset/MutantBench_code_db_java.csv \
        --train_data_file=../../dataset/Mutant_A_hierarchical.csv \
        --eval_data_file=../../dataset/Mutant_B_hierarchical.csv \
        --test_data_file=../../dataset/Mutant_B_hierarchical.csv \
        --train_batch_size 1 \
        --eval_batch_size 1 \
        --evaluate_during_training 2>&1")