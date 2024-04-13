# Large Language Models for Equivalent Mutant Detection: How Far are We?

<img src="./figs/overview.png" alt="drawing" width="1000">

In this study, we empirically investigate various LLMs with different learning strategies on equivalent mutant detection. This is a replication package for our empirical study. 

--- --- ---


## 1. Environment
* Python 3.7.7

* PyTorch 1.13.1+cu117

* Sciki-learn 1.2.2

* Transformers 4.37.0.dev0

* TRL 0.7.11

* Numpy 1.18.1

* Pandas 1.3.0

* Matplotlib 3.4.2

* Openai 1.2.3

--- --- ---

## 2. Dataset
###  (1) Statistics of Java programs from MutantBench

<img src="./figs/dataset.png" alt="drawing" width="700">



We construct a (Java) Equivalent Mutant Detection dataset based on the [MutantBench](https://github.com/MutantBench/MutantBench), which consist of [*MutantBench<sub>train</sub>*](dataset/Mutant_A_hierarchical.csv) for fine-tuning and [*MutantBench<sub>test</sub>*](dataset/Mutant_B_hierarchical.csv) for testing. 
Specifically, the dataset can be divided in to two parts:

* **Codebase** contains 3 columns that we used to conduct our experiments: 
  (1) id (int): The code id is used for retreiving the Java methods. 
  (2) code (str): The original method/mutant written in Java. 
  (3) operator (str): The type of mutation operators. 


* **Mutant-Pair Datasets** (i.e., [*MutantBench<sub>train</sub>*](dataset/Mutant_A_hierarchical.csv) and [*MutantBench<sub>test</sub>*](dataset/Mutant_B_hierarchical.csv)) contains 4 columns that we used to conduct our experiments: 
  (1) id (int): The id of mutant pair. 
  (2) code_id_1 (int): The code id is used to retreive the Java methods in Codebase. 
  (3) code_id_2 (int): The code id is used to retreive the Java methods in Codebase. 
  (4) label (int): The label that determines whether a mutant pair is equivalent or not (i.e., 1 indicates equivalent, 0 indicates non-equivalent). 


### (2) How to access the dataset
All the pre-processed data used in our experiments can be downloaded from [`./dataset`](dataset).

--- --- ---

## 3. Models

### How to access the models
All the models' checkpoints in our experiments can be downloaded from our anonymous Zenodo([link1](https://zenodo.org/records/10963111?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjMwZmMzNjkyLTUyNmYtNDY0Ny1iNzEwLTM4MjcyNmFmZjFkZCIsImRhdGEiOnt9LCJyYW5kb20iOiI5OTU3YTlhN2EzY2YzZjM3M2NiOGExZGNkYTQ2YTZkMiJ9.y0M8Ru3xYwTD0dQ1yQR_oj3Pnh87s4VSMm7JMe-qeoBPaXHCAYUhKVM9Mk8bB_WCSaiBBq-CfuE8d0e4nKXwsw),[link2](https://zenodo.org/records/10957683?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjI4MDE1ZDBmLWFkNzktNGViMy04MjZiLTU4NzdkYThkOGU2MCIsImRhdGEiOnt9LCJyYW5kb20iOiI5MmRiYjVjNGRlZDlhYjdmN2IwMGQ1NmY1MmQyNGE0MSJ9.Da-YoAZnsc3riqTb3E8d2Fxf5VVL4b4Td-08vpKDzFfQjfq751JwPX8W0aCN9HVqk96lNM_4_bLdgNW_lhwaRQ)).

-- --- ---

## 4. Experiment Replication 

For running the **open-source LLMs**, we recommend to use GPU with 48 GB up memory for training and testing, since StarCoder (7B), CodeT5+ (7B), and Code Llama (7B) are computing intensive. 

For running the **closed-source LLMs** (i.e., ChatGPT and Text-Embedding Models), you should prepare your own *OpenAI account* and *API KEY*. 


### Demo
Let's take the *pre-trained UniXCoder* as an example. 
The `dataset` folder contains the training and inference data. 

#### (1) Training phase
You can train the model through the following commands:
```
cd ./UniXCoder/code;
python train.py;
```

#### (2) Inference phase
To run the fine-tuned model to make inferences on the test dataset, run the following commands:

```
cd ./UniXCoder/code;
python test.py;
```

**How to run the remaining models and strategies** 
All the code can be accessed from respective directories.
Please find their README.md files to run respective models.

--- --- ---

## 5. Experimental Results
--- ---
#### 1)  The performance of baselines and state-of-the-art LLMs on equivalent mutant detection.
<img src="./figs/rq1.png" alt="drawing" width="700">

--- ---

#### 2)  The performance of different LLM strategies on equivalent mutant detection.
<img src="./figs/rq2.png" alt="drawing" width="700">

--- ---

#### 3) Unique correct detections (↑) and unique incorrect detections (↓) across studied EMD techniques.
<img src="./figs/rq3_veen.png" alt="drawing" width="1400">

--- ---


#### 4) Detection performance on Top-10 mutation operators across various EMD techniques (x-axis shows mutation operators and y-axis shows the correct detection percentage).

##### 4-1) Performance of 4 EMD categories on Top-10 mutation operators. Detailed results for all 28 mutation operators are available in [`./results/EMD_categories_all_operators.csv`](dataset/EMD_categories_all_operators.csv).
<img src="./figs/rq3_bar1.png" alt="drawing" width="1000">

--- ---

##### 4-2) Performance of 5 LLM strategies on Top-10 mutation operators. Detailed results for all 28 mutation operators are available in [`./results/LLM_strategies_all_operators.csv`](./results/LLM_strategies_all_operators.csv).
<img src="./figs/rq3_bar2.png" alt="drawing" width="1000">

--- ---

#### 5) Time efficiency of studied EMD techniques.
<img src="./figs/rq4.png" alt="drawing" width="700">

--- ---

#### 6) t-SNE plots showing the embedding of mutant pairs. EQ/NEQ represents equivalent/non-equivalent, respectively.
<img src="./figs/embedding.png" alt="drawing" width="700">

--- ---