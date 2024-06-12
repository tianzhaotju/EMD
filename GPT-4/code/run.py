import os
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
from retry import retry
import logging
from openai import OpenAI
import argparse
import random
import torch
from torch.utils.data import Dataset
from sklearn.metrics import precision_recall_fscore_support
logger = logging.getLogger(__name__)


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


@retry(Exception, tries=5, delay=1, backoff=2, max_delay=120)
def ChatCompletion(content, prompt, client, model='gpt-4', temperature = 0, top_p=1, max_tokens=256, frequency_penalty=0, presence_penalty=0):
    completion = client.chat.completions.create(
    model=model,
    messages=[
        {
         "role": "system",
         "content": prompt
        },
        {
         "role": "user",
         "content": content
        }
    ],
    temperature=temperature,
    max_tokens=max_tokens,
    top_p=top_p,
    frequency_penalty=frequency_penalty,
    presence_penalty=presence_penalty)
    return completion.choices[0].message


def train(args):
    pass


def test(args):
    pass


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--codebase_data_file", default=None, type=str, required=True,
                        help="The code db data file (a csv file).")
    parser.add_argument("--pretrained_embedding_file", default=None, type=str,
                        help="The pretrained embedding used to inference (a npy file).")
    parser.add_argument("--train_data_file", default=None, type=str,
                        help="The input training data file (a csv file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--api_key", default=None, type=str,
                        help="The Openai API KEY to access openai api")

    ## Other parameters
    parser.add_argument("--checkpoint_file", default=None, type=str,
                        help="The checkpoint file (a bin file).")
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a csv file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a csv file).")
    parser.add_argument("--checkpoint_dir", default=None, type=str,
                        help="An optional fine-tuned checkpoints for inference")
    parser.add_argument("--test_type", default="zero-shot prompt", type=str,
                        help="Three type of testing: zero-shot prompt, few-shot prompt, and inference_from_ckpt")
    parser.add_argument("--model_type", default=None, type=str,
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
    parser.add_argument("--eval_batch_size", default=1, type=int,
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

    if args.do_train:
        api_key = args.api_key
        pass

    if args.do_test:
        api_key = args.api_key
        client = OpenAI(api_key=api_key)
        dataset = EqDataset(args.codebase_data_file, args.test_data_file)
        model = args.model_type
        if args.test_type == "zero-shot-prompt":
            T1 = time.perf_counter()
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
                message = ChatCompletion(content, instruction, client, model)
                messages_list.append(message)
            T2 = time.perf_counter()
            Inference_time = (T2 - T1)

            prediction_list = []
            for item in messages_list:
                if 'yes' in item.lower():
                    prediction_list.append(1)
                elif 'no' in item.lower():
                    prediction_list.append(0)
            ground_truth = dataset.get_labels()
            precision, recall, fscore, support = precision_recall_fscore_support(ground_truth, prediction_list,
                                                                                 average='macro')
            print('Precision:', precision)
            print('Recall:', recall)
            print('F1-score:', fscore)
            print('Inference Time (per mutant pair): %s s' % (Inference_time / 1650))

            results_dict = {'Messages': messages_list, 'Numerical_Pred': prediction_list}
            results_df = pd.DataFrame.from_dict(results_dict)
            results_df.to_csv('.{}/zero-shot-prompt_gpt-4.csv'.format(args.output_dir), index=False)
        elif args.test_type == "few-shot-prompt":
            api_key = args.api_key
            client = OpenAI(api_key=api_key)
            model = args.model_type

            dataset = EqDataset(args.codebase_data_file, args.test_data_file)
            example_candidates = EqDataset(args.codebase_data_file, args.train_data_file)
            pos_examples = []
            neg_examples = []
            for i in range(len(example_candidates)):
                if example_candidates[i][2] == 1:
                    pos_examples.append(example_candidates[i])
                else:
                    neg_examples.append(example_candidates[i])

            # pos_id (seed: 10): 146, 8
            exp_code_0_1 = pos_examples[146][0]
            exp_code_0_2 = pos_examples[146][1]
            explaination_0 = "Yes. The two codes are semantically equal because 'quicksort( data, upper + 1, last-- )' first passes the original 'last' into the function 'quicksort()' and then dose 'last--'. Therefore, 'quicksort( data, upper + 1, last-- )' is the same with 'quicksort( data, upper + 1, last )'."

            exp_code_1_1 = pos_examples[8][0]
            exp_code_1_2 = pos_examples[8][1]
            explaination_1 = "Yes. The two codes are semantically equal because 'a[i]=a[j]++' first does 'a[i]=a[j]' and then 'a[j]++'. Therefore, 'a[i]=a[j]++' is the same with 'a[i]=a[j]'."

            # neg_id (seed: 10): 878
            exp_code_2_1 = neg_examples[878][0]
            exp_code_2_2 = neg_examples[878][1]
            explaination_2 = "No. The two codes aren't semantically equal because 'offset = spaceToWrapAt / 1;' does 'spaceToWrapAt / 1;' operation and 'offset = spaceToWrapAt + 1;' does 'spaceToWrapAt + 1' operation. Those two operatioins are different. Therefore, 'offset = spaceToWrapAt / 1;' is the different with 'offset = spaceToWrapAt + 1;'."

            T1 = time.perf_counter()
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
                message = ChatCompletion(content, instruction + example_0 + example_1 + example_2, client, model)
                messages_list.append(message)
            T2 = time.perf_counter()
            Inference_time = (T2 - T1)

            prediction_list = []
            for item in messages_list:
                if 'yes' in item.lower():
                    prediction_list.append(1)
                elif 'no' in item.lower():
                    prediction_list.append(0)
            ground_truth = dataset.get_labels()
            precision, recall, fscore, support = precision_recall_fscore_support(ground_truth, prediction_list,
                                                                                 average='macro')
            print('Precision:', precision)
            print('Recall:', recall)
            print('F1-score:', fscore)
            print('Inference Time (per mutant pair): %s s' % (Inference_time / 1650))

            results_dict = {'Messages': messages_list, 'Numerical_Pred': prediction_list}
            results_df = pd.DataFrame.from_dict(results_dict)
            results_df.to_csv('.{}/few-shot-prompt_gpt-4.csv'.format(args.output_dir), index=False)
        else:
            print("If args.do_test = True, args.test_type should be zero-shot prompt or few-shot prompt")


if __name__ == "__main__":
    main()
