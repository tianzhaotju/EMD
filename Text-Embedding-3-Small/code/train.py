import os

os.system("python run.py \
    --api_key=\
    --output_dir=../saved_models/ \
    --checkpoint_file=../saved_models/checkpoints/text-embedding-3-small.bin\
    --model_type=text-embedding-3-small \
    --do_train \
    --epoch 10\
    --learning_rate 1e-4\
    --train_batch_size 64\
    --eval_batch_size 1\
    --codebase_data_file=../../dataset/MutantBench_code_db_java.csv \
    --test_data_file=../../dataset/Mutant_B_hierarchical.csv \
    --seed 0")