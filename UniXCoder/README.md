# Replication of UniXCoder on EMD

Before replicating the experiment results please download the dataset as described below, please make sure that you have put 1 codebase (i.e., **MutantBench_code_db_java.csv**), 2 mutant pair datasets for training/testing (i.e., **Mutant_A_hierarchical.csv** & **Mutant_B_hierarchical.csv**) on the ```../dataset``` folder. 

### (1) Training
You can train the original model through the following commands:
```
cd code
python train.py
```

### (2) Testing
To run our fine-tuned model to make inferences on the test dataset, run the following commands:

```
cd code
python test.py
```

*Note 1:* Before you start the inference, please make sure that you have downloaded the [fine-tuned model](https://zenodo.org/records/10963111?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjMwZmMzNjkyLTUyNmYtNDY0Ny1iNzEwLTM4MjcyNmFmZjFkZCIsImRhdGEiOnt9LCJyYW5kb20iOiI5OTU3YTlhN2EzY2YzZjM3M2NiOGExZGNkYTQ2YTZkMiJ9.y0M8Ru3xYwTD0dQ1yQR_oj3Pnh87s4VSMm7JMe-qeoBPaXHCAYUhKVM9Mk8bB_WCSaiBBq-CfuE8d0e4nKXwsw) and saved it under the ```save_models/``` folder.

*Note 2:* In train.py and test.py, `--requires_grad=0` indicates pre-trained code embedding strategy, and `--requires_grad=1` indicates fine-tuned code embedding strategy.