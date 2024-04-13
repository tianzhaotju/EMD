import os
from datetime import datetime
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import random
import json
import argparse
import torch
from torch.utils.data import Dataset
from sklearn.metrics import precision_recall_fscore_support
from transformers import (AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, TrainingArguments)
from datasets import load_dataset
from trl import SFTTrainer
from peft import (
    PeftModel,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
)


class EqDataset(Dataset):
    def __init__(self, codebase_file, dataset_file):
        self.codebase = pd.read_csv(codebase_file)
        self.dataset = pd.read_csv(dataset_file)

        self.codebase_dict = {}
        for idx, code in zip(self.codebase['id'].tolist(), self.codebase['code'].tolist()):
            self.codebase_dict[idx] = code

        self.instance_list = []
        for code_id_1, code_id_2, label in zip(self.dataset['code_id_1'].tolist(), self.dataset['code_id_2'].tolist(),
                                               self.dataset['label'].tolist()):
            self.instance_list.append((code_id_1, code_id_2, label))

    def get_labels(self):
        return self.dataset['label'].tolist()

    def __len__(self):
        return len(self.dataset['label'].tolist())

    def __getitem__(self, idx):
        code_1 = self.codebase_dict[self.instance_list[idx][0]]
        code_2 = self.codebase_dict[self.instance_list[idx][1]]
        label = self.instance_list[idx][2]
        return code_1, code_2, label


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def ChatCompletion(instruction, content, model, tokenizer):
    messages = [
        {
            "role": "system",
            "content": instruction,
        },
        {
            "role": "user",
            "content": content
        }
    ]
    tokenized_chat = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                   return_tensors="pt").to("cuda")
    model.eval()
    with torch.no_grad():
        outputs = model.generate(tokenized_chat, max_new_tokens=3)
    pred = tokenizer.decode(outputs[0][tokenized_chat.shape[1]:])
    return pred


def SFT_dataset(codebase_data_file, train_data_file, test_data_file, output_dir):
    train_dataset = EqDataset(codebase_data_file, train_data_file)
    test_dataset = EqDataset(codebase_data_file, test_data_file)
    train_messages_list = []
    test_messages_list = []
    for idx in tqdm(range(len(train_dataset))):
        code_1 = train_dataset[idx][0]
        code_2 = train_dataset[idx][1]
        label = train_dataset[idx][2]
        instruction = "Please analyze the two following provided code files in C or Java. Identify if they are semantically equal. 'Semantically equal' means two codes have the same meaning, that they have the same output given the same input."
        query = "Please identify if the two following codes are semantically equal. Please only answer 'yes' or 'no', 'yes' means they are semantically equal. 'no' means they are not. \n" + \
                "Input: \n" + \
                "'''Code 1 \n" + \
                code_1 + \
                "\n'''\n" + \
                "'''Code 2 \n" + \
                code_2 + \
                "\n'''"
        if int(label) == int(1):
            text_label = "yes"
        else:
            text_label = "no"
        messages = {"messages": [{"role": "system", "content": instruction},
                                 {"role": "user", "content": query},
                                 {"role": "assistant", "content": text_label}]}
        train_messages_list.append(messages)
    for idx in tqdm(range(len(test_dataset))):
        code_1 = test_dataset[idx][0]
        code_2 = test_dataset[idx][1]
        label = test_dataset[idx][2]
        instruction = "Please analyze the two following provided code files in C or Java. Identify if they are semantically equal. 'Semantically equal' means two codes have the same meaning, that they have the same output given the same input."
        query = "Please identify if the two following codes are semantically equal. Please only answer 'yes' or 'no', 'yes' means they are semantically equal. 'no' means they are not. \n" + \
                "Input: \n" + \
                "'''Code 1 \n" + \
                code_1 + \
                "\n'''\n" + \
                "'''Code 2 \n" + \
                code_2 + \
                "\n'''"
        if int(label) == int(1):
            text_label = "yes"
        else:
            text_label = "no"
        messages = {"messages": [{"role": "system", "content": instruction},
                                 {"role": "user", "content": query},
                                 {"role": "assistant", "content": text_label}]}
        test_messages_list.append(messages)
    with open('{}/train_data_sft.jsonl'.format(output_dir), 'w') as f:
        for m in train_messages_list:
            f.write(json.dumps(m) + '\n')
    with open('{}/test_data_sft.jsonl'.format(output_dir), 'w') as f:
        for m in test_messages_list:
            f.write(json.dumps(m) + '\n')


def train(args, model, tokenizer, output_file, train_data, eval_data):
    tokenizer.add_eos_token = True
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"

    model.train()  # put model back into training mode
    model = prepare_model_for_int8_training(model)
    config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    # batch_size = args.train_batch_size
    # per_device_train_batch_size = 2
    # args.gradient_accumulation_steps
    # gradient_accumulation_steps = batch_size // per_device_train_batch_size

    training_args = TrainingArguments(
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_strategy="epoch",
        optim="adamw_torch",
        save_strategy="epoch",  # if val_set_size > 0 else "no",
        output_dir=output_file,
        load_best_model_at_end=False,
        gradient_checkpointing=True,
        report_to="none",
        run_name=f"codellama-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",    # if use_wandb else "none", wandb
        save_safetensors=False,    # if use_wandb else None,
        # evaluation_strategy="epoch"
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        # eval_dataset=eval_data,
        max_seq_length=4096,
        args=training_args
    )
    model.config.use_cache = False
    trainer.train()
    output_dir = '{}/checkpoints/'.format(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    trainer.model.save_pretrained(output_dir, safe_serialization=False)   # save last checkpoints for inference
    return model


def test(args, model, tokenizer, output_file, test_type, codebase_data, test_data):
    if test_type == "zero-shot-prompt" or test_type == "eval_after_ft":
        dataset = EqDataset(codebase_data, test_data)
        messages_list = []
        for idx in tqdm(range(len(dataset))):
            code_1 = dataset[idx][0]
            code_2 = dataset[idx][1]
            label = dataset[idx][2]
            instruction = "Please analyze the two following provided code files in C or Java. Identify if they are semantically equal. 'Semantically equal' means two codes have the same meaning, that they have the same output given the same input."
            content = "Please identify if the two following codes are semantically equal. Please only answer 'yes' or 'no', 'yes' means they are semantically equal. 'no' means they are not. \n" + \
                      "Input: \n" + \
                      "'''Code 1 \n" + \
                      code_1 + \
                      "\n'''\n" + \
                      "'''Code 2 \n" + \
                      code_2 + \
                      "\n'''"
            message = ChatCompletion(instruction, content, model, tokenizer)
            messages_list.append(message)

        prediction_list = []
        for item in messages_list:
            if 'yes' in item.lower():
                prediction_list.append(1)
            elif 'no' in item.lower():
                prediction_list.append(0)

        ground_truth = dataset.get_labels()
        precision, recall, fscore, support = precision_recall_fscore_support(ground_truth, prediction_list, average='macro')
        print('Precision: ', precision)
        print('Recall:', recall)
        print('F1-score:', fscore)
        result_dict = {'Message': messages_list, 'Numerical_Pred': prediction_list}
        result_df = pd.DataFrame.from_dict(result_dict)
        saved_path = '{}/{}/'.format(output_file, args.model_type)
        if not os.path.exists(saved_path):
            os.mkdir(saved_path)
        if test_type == "zero-shot-prompt":
            result_df.to_csv(os.path.join(saved_path, 'codellama_zsp_result.csv'), index=False)
            # result_df.to_csv('{}/{}/codellama_zsp_result.csv'.format(output_file, args.model_type), index=False)
        elif test_type == "eval_after_ft":
            result_df.to_csv(os.path.join(saved_path, 'codellama_ft_result.csv'), index=False)
            # result_df.to_csv('{}/{}/codellama_ft_result.csv'.format(output_file, args.model_type), index=False)
    elif test_type == "few-shot-prompt":
        example_candidates = EqDataset(codebase_data, args.train_data_file)
        dataset = EqDataset(codebase_data, test_data)
        pos_examples = []
        neg_examples = []
        for i in range(len(example_candidates)):
            if example_candidates[i][2] == 1:
                pos_examples.append(example_candidates[i])
            else:
                neg_examples.append(example_candidates[i])
        # 3 few-shot examples (2 Eq mutant pairs, 1 Neq mutant pair) from random sampling
        exp_code_0_1 = pos_examples[146][0]
        exp_code_0_2 = pos_examples[146][1]
        explaination_0 = "Yes. The two codes are semantically equal because 'quicksort( data, upper + 1, last-- )' first passes the original 'last' into the function 'quicksort()' and then dose 'last--'. Therefore, 'quicksort( data, upper + 1, last-- )' is the same with 'quicksort( data, upper + 1, last )'."
        exp_code_1_1 = pos_examples[8][0]
        exp_code_1_2 = pos_examples[8][1]
        explaination_1 = "Yes. The two codes are semantically equal because 'a[i]=a[j]++' first does 'a[i]=a[j]' and then 'a[j]++'. Therefore, 'a[i]=a[j]++' is the same with 'a[i]=a[j]'."
        exp_code_2_1 = neg_examples[878][0]
        exp_code_2_2 = neg_examples[878][1]
        explaination_2 = "No. The two codes aren't semantically equal because 'offset = spaceToWrapAt / 1;' does 'spaceToWrapAt / 1;' operation and 'offset = spaceToWrapAt + 1;' does 'spaceToWrapAt + 1' operation. Those two operatioins are different. Therefore, 'offset = spaceToWrapAt / 1;' is the different with 'offset = spaceToWrapAt + 1;'."

        messages_list = []
        for idx in tqdm(range(len(dataset))):
            code_1 = dataset[idx][0]
            code_2 = dataset[idx][1]
            label = dataset[idx][2]
            instruction = "Please analyze the two following provided code files in C or Java. Identify if they are semantically equal. 'Semantically equal' means two codes have the same meaning, that they have the same output given the same input."
            example_0 = " Here are three semantically equal examples: \n" + \
                        "The first example pair is \n" + \
                        "'''Code 1: \n" + \
                        exp_code_0_1 + \
                        "\n'''\n" + \
                        "'''Mutant Code 1: \n" + \
                        exp_code_0_2 + \
                        "\n'''\n" + \
                        explaination_0 + "\n"
            example_1 = "The second example pair is \n" + \
                        "'''Code 2: \n" + \
                        exp_code_1_1 + \
                        "\n'''\n" + \
                        "'''Mutant Code 2: \n" + \
                        exp_code_1_2 + \
                        "\n'''\n" + \
                        explaination_1 + "\n"
            example_2 = "The third example pair is \n" + \
                        "'''Code 3: \n" + \
                        exp_code_2_1 + \
                        "\n'''\n" + \
                        "'''Mutant Code 3: \n" + \
                        exp_code_2_2 + \
                        "\n'''\n" + \
                        explaination_2
            content = "Please identify if the two following codes are semantically equal. Please only answer 'yes' or 'no', 'yes' means they are semantically equal. 'no' means they are not. \n" + \
                      "Input: \n" + \
                      "'''Code 1: \n" + \
                      code_1 + \
                      "\n'''\n" + \
                      "'''Code 2: \n" + \
                      code_2 + \
                      "\n'''"
            message = ChatCompletion(instruction + example_0 + example_1 + example_2, content, model, tokenizer)
            messages_list.append(message)

        prediction_list = []
        for item in messages_list:
            if 'yes' in item.lower():
                prediction_list.append(1)
            elif 'no' in item.lower():
                prediction_list.append(0)

        ground_truth = dataset.get_labels()
        precision, recall, fscore, support = precision_recall_fscore_support(ground_truth, prediction_list, average='macro')
        print('Precision: ', precision)
        print('Recall:', recall)
        print('F1-score:', fscore)
        result_dict = {'Message': messages_list, 'Numerical_Pred': prediction_list}
        result_df = pd.DataFrame.from_dict(result_dict)
        saved_path = '{}/{}/'.format(output_file, args.model_type)
        if not os.path.exists(saved_path):
            os.mkdir(saved_path)
        result_df.to_csv(os.path.join(saved_path, 'codellama_fsp_result.csv'), index=False)
        # result_df.to_csv('{}/{}/codellama_fsp_result.csv'.format(output_file, args.model_type), index=False)
    else:
        dataset = EqDataset(codebase_data, test_data)
        messages_list = []
        for idx in tqdm(range(len(dataset))):
            code_1 = dataset[idx][0]
            code_2 = dataset[idx][1]
            label = dataset[idx][2]
            instruction = "Please analyze the two following provided code files in C or Java. Identify if they are semantically equal. 'Semantically equal' means two codes have the same meaning, that they have the same output given the same input."
            content = "Please identify if the two following codes are semantically equal. Please only answer 'yes' or 'no', 'yes' means they are semantically equal. 'no' means they are not. \n" + \
                      "Input: \n" + \
                      "'''Code 1 \n" + \
                      code_1 + \
                      "\n'''\n" + \
                      "'''Code 2 \n" + \
                      code_2 + \
                      "\n'''"
            message = ChatCompletion(instruction, content, model, tokenizer)
            messages_list.append(message)

        prediction_list = []
        for item in messages_list:
            if 'yes' in item.lower():
                prediction_list.append(1)
            elif 'no' in item.lower():
                prediction_list.append(0)

        ground_truth = dataset.get_labels()
        precision, recall, fscore, support = precision_recall_fscore_support(ground_truth, prediction_list,
                                                                             average='macro')
        print('Precision: ', precision)
        print('Recall:', recall)
        print('F1-score:', fscore)
        result_dict = {'Message': messages_list, 'Numerical_Pred': prediction_list}
        result_df = pd.DataFrame.from_dict(result_dict)
        saved_path = '{}/{}/'.format(output_file, args.model_type)
        if not os.path.exists(saved_path):
            os.mkdir(saved_path)
        result_df.to_csv(os.path.join(saved_path, 'codellama_ft_result.csv'), index=False)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--codebase_data_file", default=None, type=str, required=True,
                        help="The code db data file (a csv file).")
    parser.add_argument("--train_data_file", default=None, type=str,
                        help="The input training data file (a csv file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a csv file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a csv file).")
    parser.add_argument("--checkpoint_dir", default=None, type=str,
                        help="An optional fine-tuned checkpoints for inference")
    parser.add_argument("--test_type", default=None, type=str,
                        help="Three type of testing: zero-shot-prompt, few-shot-prompt, and inference_from_ckpt")
    parser.add_argument("--model_type", default="codellama", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoints for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoints every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--epoch', type=int, default=10,
                        help="num of epoch for training model")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # Set seed
    set_seed(args.seed)

    # Load LLM checkpoints and tokenizer
    # model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    # Training
    if args.do_train:
        # create SFT dataset
        T1_train = time.perf_counter()
        SFT_dataset(args.codebase_data_file, args.train_data_file, args.test_data_file, args.output_dir)

        # load SFT dataset
        train_dataset = load_dataset('json', data_files='{}/train_data_sft.jsonl'.format(args.output_dir),
                                     split="train")
        eval_dataset = load_dataset('json', data_files='{}/test_data_sft.jsonl'.format(args.output_dir),
                                    split="train")

        # load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                load_in_8bit=True,
                torch_dtype=torch.float16,
                device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

        # train codellama
        model = train(args, model, tokenizer, args.output_dir, train_dataset, eval_dataset)
        T2_train = time.perf_counter()

        # evaluate on last ckpt
        test_type = "eval_after_ft"
        T1_test = time.perf_counter()
        test(args, model, tokenizer, args.output_dir, test_type, args.test_data_file, args.codebase_data_file)
        T2_test = time.perf_counter()
        print('Training Time (Total): %s s' % (T2_train - T1_train))
        print('Inference Time (per mutant pair): %s s' % ((T2_test - T1_test)/1650))

    # Testing
    if args.do_test:
        if args.test_type == "zero-shot-prompt" or args.test_type == "few-shot-prompt":
            model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
            model.to(torch.device('cuda:0'))
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
            T1 = time.perf_counter()
            test(args, model, tokenizer, args.output_dir, args.test_type, args.codebase_data_file, args.test_data_file)
            T2 = time.perf_counter()
            print('Inference Time (per mutant pair): %s s' % ((T2 - T1)/1650))
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                load_in_8bit=True,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            model = PeftModel.from_pretrained(model, args.checkpoint_dir)

            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
            T1 = time.perf_counter()
            test(args, model, tokenizer, args.output_dir, args.test_type, args.codebase_data_file, args.test_data_file)
            T2 = time.perf_counter()
            print('Inference Time (per mutant pairs): %s s' % ((T2 - T1) / 1650))



if __name__ == "__main__":
    main()