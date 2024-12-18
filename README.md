# Large Language Models for Equivalent Mutant Detection: How Far Are We?

<img src="./figs/overview.png" alt="drawing" width="800">

In this study, we empirically investigate various LLMs with different learning strategies for equivalent mutant detection. This is a replication package for our empirical study. 

🏆 **ACM SIGSOFT Distinguished Paper Award** (ISSTA 2024)

--- --- ---


***Contact***: Feel free to contact Zhao Tian (tianzhao@tju.edu.cn), Honglin Shu (shu.honglin.167@s.kyushu-u.ac.jp), and Xuejie Cao (caoxuejie@tju.edu.cn) if you have any further questions about the code.


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

<img src="./figs/dataset.png" alt="drawing" width="600">



We construct a (Java) Equivalent Mutant Detection dataset based on the [MutantBench](https://github.com/MutantBench/MutantBench), which consists of [*MutantBench<sub>train</sub>*](dataset/Mutant_A_hierarchical.csv) for fine-tuning and [*MutantBench<sub>test</sub>*](dataset/Mutant_B_hierarchical.csv) for testing. 
Specifically, the dataset can be divided into two parts:

* **Codebase**  (i.e., [`./dataset/MutantBench_code_db_java.csv`](dataset/MutantBench_code_db_java.csv)) contains 3 columns that we used to conduct our experiments: 
  (1) id (int): The code id is used for retrieving the Java methods. 
  (2) code (str): The original method/mutant written in Java. 
  (3) operator (str): The type of mutation operators. 


* **Mutant-Pair Datasets** (i.e., [*MutantBench<sub>train</sub>*](dataset/Mutant_A_hierarchical.csv) and [*MutantBench<sub>test</sub>*](dataset/Mutant_B_hierarchical.csv)) contains 4 columns that we used to conduct our experiments: 
  (1) id (int): The id of mutant pair. 
  (2) code_id_1 (int): The code id is used to retrieve the Java methods in Codebase. 
  (3) code_id_2 (int): The code id is used to retrievethe Java methods in Codebase. 
  (4) label (int): The label that determines whether a mutant pair is equivalent or not (i.e., 1 indicates equivalent, 0 indicates non-equivalent). 


### (2) How to access the dataset
All the pre-processed data used in our experiments can be downloaded from [`./dataset`](dataset).

--- --- ---

## 3. Models

### How to access the models
All the models' checkpoints in our experiments can be downloaded from our anonymous Zenodo([link1](https://zenodo.org/records/10967393),[link2](https://zenodo.org/records/10957683)).

-- --- ---

## 4. Experiment Replication 

For running the **open-source LLMs**, we recommend using GPU with 48 GB up memory for training and testing, since StarCoder (7B), CodeT5+ (7B), and Code Llama (7B) are computing intensive. 

For running the **closed-source LLMs** (i.e., ChatGPT and Text-Embedding Models), you should prepare your own *OpenAI account* and *API KEY*. 


### Demo
Let's take the *pre-trained UniXCoder* as an example. 
The `./dataset` folder contains the training and test data. 

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
<img src="./figs/rq1.png" alt="drawing" width="600">

--- ---

#### 2)  The performance of different LLM strategies on equivalent mutant detection.
<img src="./figs/rq2.png" alt="drawing" width="600">

--- ---

#### 3) Unique correct detections (↑) and unique incorrect detections (↓) across studied EMD techniques.
<img src="./figs/rq3_veen.png" alt="drawing" width="1200">

--- ---


#### 4) Detection performance on Top-10 mutation operators across various EMD techniques (x-axis shows mutation operators and y-axis shows the correct detection percentage).

##### 4-1) Performance of 4 EMD categories on Top-10 mutation operators. Detailed results for all 28 mutation operators are available in [`./results/EMD_categories_all_operators.csv`](dataset/EMD_categories_all_operators.csv).
<img src="./figs/rq3_bar1.png" alt="drawing" width="1000">

--- ---

##### 4-2) Performance of 5 LLM strategies on Top-10 mutation operators. Detailed results for all 28 mutation operators are available in [`./results/LLM_strategies_all_operators.csv`](./results/LLM_strategies_all_operators.csv).
<img src="./figs/rq3_bar2.png" alt="drawing" width="1000">

--- ---

#### 5) t-SNE plots showing the embedding of mutant pairs. EQ/NEQ represents equivalent/non-equivalent, respectively.
<img src="./figs/embedding.png" alt="drawing" width="600">

--- ---
