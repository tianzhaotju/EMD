# Replication of GPT-3.5-Turbo on EMD

Before replicating the experiment results, ensure that you have placed 1 codebase file (i.e.,*MutantBench_code_db_java.csv*), and 2 mutant-pair files for training/testing (i.e.,*Mutant_A_hierarchical.csv* and *Mutant_B_hierarchical.csv*) in the `../dataset` folder.

### (1) Training
Before starting SFT, please add your OpenAI API key to the ```--api_key=``` in ```./code/train.py```.  
You can fine-tune the GPT-3.5-Turbo and apply the fine-tuned model through the following commands:
```
cd code
python train.py
```

### (2) Testing
**Inference of Zero-shot Prompting**  
Before starting inference, please add your OpenAI API key to the ```--api_key=``` in ```./code/test_zsp.py```.  
If you want to apply GPT-3.5-Turbo with Zero-shot Prompting on the test dataset, run the following commands:
```
cd code
python test_zsp.py
```
**Inference of Few-shot Prompting**  
Before starting inference, please add your OpenAI API key to the ```--api_key=``` in ```./code/test_fsp.py```.  
If you want to apply GPT-3.5-Turbo with Few-shot Prompting on the test dataset, run the following commands:
```
cd code
python test_fsp.py
```

