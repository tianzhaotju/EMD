# Replication of StarCoder on EMD

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

*Note:* Before you start the inference, please make sure that you have downloaded the [fine-tuned model](https://zenodo.org/records/10957683?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjI4MDE1ZDBmLWFkNzktNGViMy04MjZiLTU4NzdkYThkOGU2MCIsImRhdGEiOnt9LCJyYW5kb20iOiI5MmRiYjVjNGRlZDlhYjdmN2IwMGQ1NmY1MmQyNGE0MSJ9.Da-YoAZnsc3riqTb3E8d2Fxf5VVL4b4Td-08vpKDzFfQjfq751JwPX8W0aCN9HVqk96lNM_4_bLdgNW_lhwaRQ) and saved it under the ```save_models/``` folder.