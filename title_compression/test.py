# -*- coding: utf-8 -*-
import io, os
import sys
sys.path.append(".")
from title_compression.utils import *
import argparse
import torch
import random
from tqdm import tqdm

from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from pytorch_pretrained_bert.modeling import BertForTokenClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def get_word_pred_labels(segment_ids, logits, word_cnt):
    word_idx = 0
    word_logits = np.zeros((word_cnt, 2))
    for i in range(1, logits.shape[0]):
        word_logits[word_idx] += logits[i]
        if segment_ids[i] == 3 or segment_ids[i] == 2:
            word_idx += 1
        if word_idx >= word_cnt:
            break
    return sigmoid(word_logits[:, 1]) >= 0.3


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--type_vocab_size", default=2, type=int)
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    processors = {
        "tc": TitleCompressionProcessor
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
    #     raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    type_words_dict = {}
    with io.open('title_compression/type_words.txt', encoding='utf-8') as fin:
        lines = fin.readlines()
        for line in lines:
            arr = line.strip().split('\t')
            type = arr[0]
            words = arr[1].split(' ')
            type_words_dict[type] = words

    processor = processors[task_name](type_words_dict)
    label_list = processor.get_labels()
    num_labels = len(label_list)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # Prepare model
    print(device)
    model = load_model(args.output_dir, device, num_labels=num_labels)
    if args.fp16:
        model.half()
    model.to(device)

    if args.do_eval:
        eval_examples = processor.get_dev_examples(args.data_dir)
    else:
        eval_examples = processor.get_test_examples(args.data_dir)
    eval_features = convert_examples_to_features(
        eval_examples, label_list, processor.get_BIES_labels(), args.max_seq_length, tokenizer)
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    # Run prediction for full data
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    word_acc = 0.
    word_num = 0

    pred_texts = []
    example_idx = 0
    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating"):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
            logits = model(input_ids, segment_ids, input_mask)

        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        input_mask = input_mask.to('cpu').numpy()
        tmp_eval_accuracy = accuracy(logits, label_ids, input_mask)

        # pred_ids = logits.argmax(-1)
        # input_ids = input_ids.cpu().numpy()
        # for i in range(pred_ids.shape[0]):
        #     retained_ids = input_ids[i][pred_ids[i] == 1]
        #     retained_tokens = BertTokenizer.convert_ids_to_tokens(tokenizer, retained_ids)
        #     pred_texts.append(''.join(retained_tokens))

        segment_ids = segment_ids.cpu().numpy()
        for i in range(logits.shape[0]):
            example = eval_examples[example_idx]
            word_pred_labels = get_word_pred_labels(segment_ids[i], logits[i], len(example.words))
            word_acc += np.sum(np.equal(word_pred_labels, example.labels))
            word_num += len(example.words)
            pred_words = [example.words[t] for t, lb in enumerate(word_pred_labels) if lb == 1]
            pred_types = [example.types[t] for t, lb in enumerate(word_pred_labels) if lb == 1]
            for i in range(len(pred_words)):
                for j in range(len(pred_words)):
                    if i != j and pred_words[i] in pred_words[j]:
                        pred_words[i] = ''
                        pred_types[i] = ''
            pred_words = [word for word in pred_words if word != '']
            product_idx = -1
            for i in reversed(range(len(pred_words))):
                if pred_types[i] == '品类':
                    product_idx = i
                    break
            if product_idx >= 0 and product_idx != len(pred_words) - 1:
                tmp = pred_words[product_idx]
                pred_words[product_idx] = pred_words[-1]
                pred_words[-1] = tmp
            pred_texts.append(''.join(pred_words))
            example_idx += 1
        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += np.sum(input_mask)
        nb_eval_steps += 1



    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples

    print(eval_loss)
    print(eval_accuracy)
    print(word_acc / word_num)

    with io.open('pred_res.txt', 'w+', encoding='utf-8') as fout:
        for example, pred_text in zip(eval_examples, pred_texts):
            fout.write('%s\t%s\t%s\n' % (example.text_a, ''.join([str(lb) for lb in example.labels]), pred_text))


if __name__ == '__main__':
    main()
