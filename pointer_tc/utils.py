# -*- coding: utf-8 -*-

import logging, os, io, csv
import numpy as np
import torch
import jieba
from collections import defaultdict
import re

from pytorch_pretrained_bert.modeling import BertForTokenClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_char_labels_to_token_labels(tokens, labels):
    idx = 0
    token_labels = []
    for token in tokens:
        # 这个可能有潜在问题
        if token == '[UNK]':
            token_labels.append(labels[idx])
            idx += 1
            continue
        if token[:2] == '##':
            token = token[2:]
        token_labels.append(labels[idx + len(token) - 1])
        idx += len(token)
    return token_labels


def get_bies_labels(tokens, words):
    labels = []
    for word in words:
        if len(word) == 1:
            labels += ['S']
        else:
            word_labels = ['B'] + ['I'] * (len(word) - 2) + ['E']
            labels += word_labels
    return convert_char_labels_to_token_labels(tokens, labels)


def get_ner_labels(tokens, words, types, ner_type_id):
    ner_labels = []
    for word, type in zip(words, types):
        if len(type) == 1:
            ner_labels += ['S_' + type]
        elif len(type) > 1:
            ner_labels += ['B_' + type] + ['I_' + type] * (len(word) - 2) + ['E_' + type]
        else:
            ner_labels += ['O'] * len(word)
    ner_labels = np.asarray([ner_type_id.get(type) for type in ner_labels])
    return convert_char_labels_to_token_labels(tokens, ner_labels)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, eid, src, tgt, labels=None):
        """Constructs a InputExample."""

        self.eid = eid
        self.src = src
        self.tgt = tgt


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, src_ids, tgt_ids, src_mask, tgt_mask, segment_ids, labels):
        self.src_ids = src_ids
        self.src_mask = src_mask
        self.tgt_mask = tgt_mask
        self.segment_ids = segment_ids
        self.tgt_ids = tgt_ids
        self.labels = labels


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with io.open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class TitleCompressionProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.txt")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "val.txt")), "dev")

    def get_test_examples(self, data_dir):
        examples = []
        with io.open(os.path.join(data_dir, "test.txt"), encoding='utf-8') as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.strip().lower()
                arr = line.split('\t')
                src = arr[0]
                examples.append(InputExample(eid="", src=src, tgt=""))
            return examples

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            src = line[0]
            tgt = line[2]
            examples.append(
                InputExample(eid="", src=src, tgt=tgt))
        return examples


def convert_examples_to_features(examples, max_src_length, max_tgt_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
    features = []
    for (ex_index, example) in enumerate(examples):
        src_tokens = tokenizer.tokenize(example.src)

        if len(src_tokens) > max_src_length - 2:
            src_tokens = src_tokens[:(max_src_length - 2)]

        src_tokens = ["[CLS]"] + src_tokens + ["[SEP]"]

        segment_ids = [0] * len(src_tokens)

        src_ids = tokenizer.convert_tokens_to_ids(src_tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        src_mask = [1] * len(src_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_src_length - len(src_ids))
        src_ids += padding
        src_mask += padding
        segment_ids += padding

        assert len(src_ids) == max_src_length
        assert len(src_mask) == max_src_length
        assert len(segment_ids) == max_src_length

        tgt_tokens = tokenizer.tokenize(example.tgt)
        if len(tgt_tokens) > max_tgt_length - 2:
            tgt_tokens = tgt_tokens[:(max_tgt_length - 2)]
        tgt_tokens = ["[CLS]"] + tgt_tokens + ["[SEP]"]
        tgt_ids = tokenizer.convert_tokens_to_ids(tgt_tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        tgt_mask = [1] * len(tgt_ids)

        # Zero-pad up to the sequence length.
        tgt_padding = [0] * (max_tgt_length - len(tgt_ids))
        tgt_ids += tgt_padding
        tgt_mask += tgt_padding

        labels = []
        idx = 0
        for tgt_token in tgt_tokens:
            idx = src_tokens.index(tgt_token, start=idx)
            labels += [idx]
        labels += tgt_padding
        assert len(tgt_ids) == max_tgt_length
        assert len(tgt_mask) == max_tgt_length
        assert len(labels) == max_tgt_length

        if ex_index < 5:
            logger.info("*** Example ***")
            # logger.info("guid: %s" % example.guid)
            logger.info("src tokens: %s" % " ".join(
                [str(x) for x in src_tokens]))
            logger.info("tgt tokens: %s" % " ".join(
                [str(x) for x in tgt_tokens]))
            logger.info("src_ids: %s" % " ".join([str(x) for x in src_ids]))
            logger.info("tgt_ids: %s" % " ".join([str(x) for x in tgt_ids]))
            logger.info("src_mask: %s" % " ".join([str(x) for x in src_mask]))
            logger.info("tgt_mask: %s" % " ".join([str(x) for x in tgt_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info(
                "labels: %s" % " ".join([str(x) for x in labels]))

        features.append(
            InputFeatures(src_ids=src_ids,
                          src_mask=src_mask,
                          tgt_ids=tgt_ids,
                          tgt_mask=tgt_mask,
                          segment_ids=segment_ids,
                          labels=labels))
    return features


def accuracy(out, labels, mask):
    out = out.argmax(-1)
    acc = np.sum(np.equal(out, labels) * mask)
    return acc


def load_model(dir, device, num_labels):
    output_config_file = os.path.join(dir, CONFIG_NAME)
    output_model_file = os.path.join(dir, WEIGHTS_NAME)
    config = BertConfig(output_config_file)
    model = BertForTokenClassification(config, num_labels=num_labels)
    model.load_state_dict(torch.load(output_model_file, map_location=device))

    return model