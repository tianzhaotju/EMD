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
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, AdamW
logger = logging.getLogger(__name__)


class RobertaClassificationHead(nn.Module):
    """Head for EMD classification tasks."""

    def __init__(self, hidden_size, hidden_dropout_prob, num_class=2):
        super().__init__()
        self.dense = nn.Linear(hidden_size * 2, hidden_size, dtype=torch.float64)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.out_proj = nn.Linear(hidden_size, num_class, dtype=torch.float64)
        self.softmax = torch.nn.Softmax()

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        x = self.softmax(x)
        return x


class EqDataset(Dataset):
    def __init__(self, embedding_file, dataset_file, code_db_file):
        self.dataset_df = pd.read_csv(dataset_file)
        self.code_db = pd.read_csv(code_db_file)
        self.code_embedding = np.load(embedding_file)

        self.embedding_dict = {}
        for idx, item in enumerate(self.code_db['id'].tolist()):
            self.embedding_dict[item] = self.code_embedding[idx]

        self.code_pairs = []
        self.code_id_1 = self.dataset_df['code_id_1'].tolist()
        self.code_id_2 = self.dataset_df['code_id_2'].tolist()
        self.labels = self.dataset_df['label'].tolist()
        if len(self.code_id_1) == len(self.labels) and len(self.code_id_2) == len(self.labels):
            for i in range(len(self.labels)):
                self.code_pairs.append((self.code_id_1[i], self.code_id_2[i], self.labels[i]))
        else:
            print('code_id_1 and code_id_2 should has same length to labels.')

        if len(self.code_pairs) != len(self.labels):
            print('code_pairs should has same length to labels.')

    def get_labels(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        code_1 = self.embedding_dict[self.code_pairs[idx][0]]
        code_2 = self.embedding_dict[self.code_pairs[idx][1]]
        label = self.code_pairs[idx][2]
        return code_1, code_2, label


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


@retry(Exception, tries=5, delay=1, backoff=2, max_delay=120)
def getEmbedding(code, client, encoder='text-embedding-ada-002'):
    embedding = client.embeddings.create(input=code, model=encoder).data[0].embedding
    return embedding


def train(args, train_dataloader, eval_dataloader, model, loss_fn, epochs, learning_rate, adam_epsilon=1e-8,
          gradient_accumulation_steps=1):
    max_steps = epochs * len(train_dataloader)
    save_steps = len(train_dataloader)
    warmup_steps = max_steps // 5
    print(warmup_steps)
    print(max_steps)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_steps)

    global_step = 0
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_f1 = 0

    model.zero_grad()
    for idx in range(epochs):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            (x1, x2, y) = [x for x in batch]
            model.train()
            logits = model(x1, x2)
            loss = loss_fn(logits, y)

            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss

            avg_loss = round(train_loss / tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))

            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                output_flag = True
                avg_loss = round(
                    np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)

                if global_step % save_steps == 0:
                    results = evaluate(eval_dataloader, model, loss_fn, eval_when_training=True)
                    print(results)

                    # Save model checkpoints
                    if results['eval_f1'] > best_f1:
                        best_f1 = results['eval_f1']
                        logger.info("  " + "*" * 20)
                        logger.info("  Best f1:%s", round(best_f1, 4))
                        logger.info("  " + "*" * 20)

                        checkpoint_prefix = 'checkpoints-best-f1'
                        output_dir = os.path.join(
                            '.{}/checkpoints/{}'.format(args.output_dir, checkpoint_prefix))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(
                            model, 'module') else model
                        output_dir = os.path.join(
                            output_dir, '{}'.format('model.bin'))
                        print('SAVING')
                        torch.save(model_to_save.state_dict(), output_dir)
                        logger.info(
                            "Saving model checkpoints to %s", output_dir)


def evaluate(eval_dataloader, model, loss_fn, eval_when_training=False):
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []
    for batch in tqdm(eval_dataloader):
        (x1, x2, y) = [x for x in batch]
        with torch.no_grad():
            logit = model(x1, x2)
            lm_loss = loss_fn(logit, y)
            eval_loss += lm_loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(y.cpu().numpy())
        nb_eval_steps += 1

    # calculate scores
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    best_threshold = 0.5

    y_preds = logits[:, 1] > best_threshold
    from sklearn.metrics import recall_score
    recall = recall_score(y_trues, y_preds, average='macro')
    from sklearn.metrics import precision_score
    precision = precision_score(y_trues, y_preds, average='macro')
    from sklearn.metrics import f1_score
    f1 = f1_score(y_trues, y_preds, average='macro')
    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_threshold": best_threshold,
    }

    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    return result


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
    parser.add_argument("--learning_rate", default=2e-5, type=float,
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

    if args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.do_train:
        mutant_df = pd.read_csv(args.codebase_data_file)
        mutant_code = {}
        for i in range(mutant_df.shape[0]):
            index = mutant_df.iloc[i]['id']
            code = mutant_df.iloc[i]['code']
            mutant_code[index] = code
        mutant_set = list(mutant_code.values())

        api_key = args.api_key
        client = OpenAI(api_key=api_key)

        code_embedding = []
        T1 = time.perf_counter()
        for item in tqdm(mutant_set):
            embedding = getEmbedding(code=item, encoder=args.model_type, client=client)
            code_embedding.append(embedding)
        T2 = time.perf_counter()
        embedding_generation_time = T2 - T1

        code_embedding_arr = np.array(code_embedding)

        checkpoint_dir = os.path.join(args.output_dir, 'checkpoints/')
        saved_file = checkpoint_dir + args.model_type + '.npy'
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
        np.save(saved_file, code_embedding_arr)

        train_dataset = EqDataset(saved_file, args.train_data_file, args.codebase_data_file)
        test_dataset = EqDataset(saved_file, args.test_data_file, args.codebase_data_file)
        train_dataloader = DataLoader(train_dataset, args.train_batch_size)
        test_dataloader = DataLoader(test_dataset, args.eval_batch_size)
        input_dim = train_dataset[0][0].shape[0]
        num_classes = 2
        dropout = 0.25
        loss_fn = nn.CrossEntropyLoss()
        model = RobertaClassificationHead(input_dim, dropout, num_classes)
        model.to(device)

        T1 = time.perf_counter()
        train(args, train_dataloader, test_dataloader, model, loss_fn, args.epoch, args.learning_rate)
        T2 = time.perf_counter()
        clf_training_time = T2 - T1

        T1 = time.perf_counter()
        results = evaluate(test_dataloader, model, loss_fn)
        T2 = time.perf_counter()
        print(results)
        clf_inference_time = T2 - T1

        tot_training_time = (embedding_generation_time/len(code_embedding)) * 1652 * 2 * args.epoch + clf_training_time
        avg_inference_time = ((embedding_generation_time/len(code_embedding)) * 1650 * 2 + clf_inference_time) / 1650
        print('Training Time: %s s' % tot_training_time)
        print('Inference Time: %s s' % avg_inference_time)

    if args.do_test:
        dataset = EqDataset(args.pretrained_embedding_file, args.test_data_file, args.codebase_data_file)
        dataloader = DataLoader(dataset, args.eval_batch_size)
        input_dim = dataset[0][0].shape[0]
        num_classes = 2
        dropout = 0.25
        loss_fn = nn.CrossEntropyLoss()
        model = RobertaClassificationHead(input_dim, dropout, num_classes)
        model.load_state_dict(torch.load(args.checkpoint_file))
        model.to(device)

        T1 = time.perf_counter()
        results = evaluate(dataloader, model, loss_fn)
        print(results)
        T2 = time.perf_counter()

        clf_inf_time = T2 - T1
        embedding_gen_time = 924.4112
        print('Inference Time: %s s' % (((embedding_gen_time/3112) * 1650 * 2 + clf_inf_time)/1650))


if __name__ == "__main__":
    main()
