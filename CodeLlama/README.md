# Replication of Code Llama on EMD

Before replicating the experiment results please download the dataset as described below, please make sure that you have put 1 codebase (i.e., **MutantBench_code_db_java.csv**), 2 mutant pair datasets for training/testing (i.e., **Mutant_A_hierarchical.csv** & **Mutant_B_hierarchical.csv**) on the ```../dataset``` folder. 

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
Before you start the inference, please make sure that you have downloaded the [checkpoint](https://zenodo.org/records/10963111?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjMwZmMzNjkyLTUyNmYtNDY0Ny1iNzEwLTM4MjcyNmFmZjFkZCIsImRhdGEiOnt9LCJyYW5kb20iOiI5OTU3YTlhN2EzY2YzZjM3M2NiOGExZGNkYTQ2YTZkMiJ9.y0M8Ru3xYwTD0dQ1yQR_oj3Pnh87s4VSMm7JMe-qeoBPaXHCAYUhKVM9Mk8bB_WCSaiBBq-CfuE8d0e4nKXwsw) and saved it under the ```save_models/``` folder.
Note that the checkpoints only saved the fine-tuned parameters of Lora adapters. When you run the following comments, the code will automatically download the Codellama checkpoint from huggingface and then load the fine-tuned adapters.
To run Code Llama with Fine-tune with Instruction to make inferences on the test dataset, run the following commands:
```
cd code
python test_ft.py
```
