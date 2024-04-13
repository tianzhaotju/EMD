# Replication of GPT-4 on EMD

Before replicating the experiment results, ensure that you have placed 1 codebase file (i.e.,*MutantBench_code_db_java.csv*), and 2 mutant-pair files for training/testing (i.e.,*Mutant_A_hierarchical.csv* and *Mutant_B_hierarchical.csv*) in the `../dataset` folder.

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

