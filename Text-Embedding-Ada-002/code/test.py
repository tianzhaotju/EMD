import os

os.system("python run.py \
    --output_dir=../saved_models/ \
    --checkpoint_file=../saved_models/checkpoints/text-embedding-ada-002.bin\
    --model_type=text-embedding-ada-002 \
    --do_test \
    --eval_batch_size 1\
    --codebase_data_file=../../dataset/MutantBench_code_db_java.csv \
    --pretrained_embedding_file=../saved_models/embeddings/text-embedding-ada-002.npy\
    --test_data_file=../../dataset/Mutant_B_hierarchical.csv \
    --seed 0")