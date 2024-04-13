# Replication of Text-Embedding-3-Large Model on EMD

Before replicating the experiment results, ensure that you have placed 1 codebase file (i.e.,*MutantBench_code_db_java.csv*), and 2 mutant-pair files for training/testing (i.e.,*Mutant_A_hierarchical.csv* and *Mutant_B_hierarchical.csv*) in the `../dataset` folder.

### (1) Training
Before starting training, please add your OpenAI API key to the ```--api_key=``` in ```./code/train.py```.  
You can generate the code embedding and use it to train a classifier through the following commands:
```
cd code
python train.py
```

### (2) Testing
Please ensure you download [text-embedding-3-large.bin](https://zenodo.org/records/10967393) on ```./saved_models/checkpoints/``` folder, 
and [text-embedding-3-large.npy](https://zenodo.org/records/10967393) on ```./saved_models/embeddings/``` folder.  
After that, you can apply the pre-trained code embedding and trained the classifier on test dataset, through the following commands:
```
cd code
python test.py
```

