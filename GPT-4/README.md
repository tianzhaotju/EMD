# Replication of GPT-4 on EMD

Before replicating the experiment results, please make sure that you have put 1 codebase (i.e., **MutantBench_code_db_java.csv**), 2 mutant pair datasets for training/testing (i.e., **Mutant_A_hierarchical.csv** & **Mutant_B_hierarchical.csv**) on the ```../dataset``` folder.

### (1) Testing
**Inference of Zero-shot Prompting**  
Before starting inference, please add your OpenAI API key to the ```--api_key=``` in ```./code/test_zsp.py```.  
If you want to apply GPT-4 with Zero-shot Prompting on the test dataset, run the following commands:
```
cd code
python test_zsp.py
```
**Inference of Few-shot Prompting**  
Before starting inference, please add your OpenAI API key to the ```--api_key=``` in ```./code/test_zsp.py```.  
If you want to apply GPT-4 with Few-shot Prompting on the test dataset, run the following commands:
```
cd code
python test_fsp.py
```

