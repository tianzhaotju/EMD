# Replication of Code Llama on EMD

Before replicating the experiment results, ensure that you have placed 1 codebase file (i.e.,*MutantBench_code_db_java.csv*), and 2 mutant-pair files for training/testing (i.e.,*Mutant_A_hierarchical.csv* and *Mutant_B_hierarchical.csv*) in the `../dataset` folder.

### (1) Training
You can train the Code Llama through the following commands:
```
cd code
python train.py
```
Then, the program will automatically train the Code Llama via SFT manner and do inference after training.



### (2) Testing:
**Inference of Zero-shot Prompting**  
To run Code Llama with Zero-shot Prompting to make inferences on the test dataset, run the following  commands:
```
cd code
python test_zsp.py
```

**Inference of Few-shot Prompting**  
To run Code Llama with Few-shot Prompting to make inferences on the test dataset, run the following  commends:
```
cd code
python test_fsp.py
```

**Inference of Fine-tuning with Instruction**  
Before you start the inference, please make sure that you have downloaded the [checkpoint](https://zenodo.org/records/10967393) and saved it under the ```save_models/``` folder.
Note that the checkpoints only saved the fine-tuned parameters of Lora adapters. When you run the following comments, the code will automatically download the Codellama checkpoint from huggingface and then load the fine-tuned adapters.
To run Code Llama with Fine-tune with Instruction to make inferences on the test dataset, run the following commands:
```
cd code
python test_ft.py
```
