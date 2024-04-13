# Replication of Text-Embedding-3-Small Model on EMD

Before replicating the experiment results, please make sure that you have put 1 codebase (i.e., **MutantBench_code_db_java.csv**), 2 mutant pair datasets for training/testing (i.e., **Mutant_A_hierarchical.csv** & **Mutant_B_hierarchical.csv**) on the ```../dataset``` folder. 

### (1) Training
Before starting training, please add your OpenAI API key to the ```--api_key=``` in ```./code/train.py```.  
You can generate the code embedding and use it to train a classifier through the following commands:
```
cd code
python train.py
```

### (2) Testing
Please ensure you have the **text-embedding-3-small.bin** on ```./saved_models/checkpoints/``` folder, 
and **text-embedding-3-small.npy** on ```./saved_models/embeddings/``` folder.  
After that, you can apply the pre-trained code embedding and trained classifier on test dataset, through the following commands:
```
cd code
python test.py
```

