# Replication of StarCoder on EMD

Before replicating the experiment results, ensure that you have placed 1 codebase file (i.e.,*MutantBench_code_db_java.csv*), and 2 mutant-pair files for training/testing (i.e.,*Mutant_A_hierarchical.csv* and *Mutant_B_hierarchical.csv*) in the `../dataset` folder.

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

*Note 1:* Before you start the inference, please make sure that you have downloaded the [fine-tuned model](https://zenodo.org/records/10957683) and saved it under the ```save_models/``` folder.

*Note 2:* In train.py and test.py, `--requires_grad=0` indicates pre-trained code embedding strategy, and `--requires_grad=1` indicates fine-tuned code embedding strategy.