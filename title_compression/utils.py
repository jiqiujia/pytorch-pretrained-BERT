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

    def __init__(self, eid, words, types, text_a, labels=None):
        """Constructs a InputExample."""

        self.eid = eid
        self.words = words
        self.types = types
        self.text_a = text_a
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids, ner_label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.ner_label_ids = ner_label_ids


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

    pattern = re.compile(r"[a-z0-9]+")
    def __init__(self, type_words_dict: dict = None, ner_type_id: dict = None):
        if type_words_dict is not None:
            self.type_words_dict = type_words_dict
            self.word_type_dict = {}
            for type, words in self.type_words_dict.items():
                for word in words:
                    jieba.add_word(word)
                    if word in self.word_type_dict:
                        logger.info(word)
                        if self.word_type_dict[word] == '品类':
                            continue
                    self.word_type_dict[word] = type
        self.ner_type_id = ner_type_id

    def get_title_words(self, title: str):
        matches = self.pattern.finditer(title)
        words = []
        start = 0
        for m in matches:
            ms, me = m.span()
            sub_title = title[start:ms]
            words += list(jieba.cut(sub_title))
            words += [title[ms:me]]
            start = me
        words += list(jieba.cut(title[start:]))
        return words

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        examples = []
        with io.open(os.path.join(data_dir, "test.txt"), encoding='utf-8') as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.strip().lower()
                words = self.get_title_words(line)
                words = [word for word in words if word in self.word_type_dict]
                types = [self.word_type_dict.get(word, "") for word in words]
                text = ''.join(words)
                if len(text) == 0:
                    continue
                examples.append(InputExample(eid="", words=words, types=types,
                                             text_a=text, labels=np.zeros(len(words), dtype=np.int8)))
        return examples

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def get_BIES_labels(self):
        return ['B', 'I', 'E', 'S']

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            eid = line[0]
            words = line[1].split(' ')
            types = line[2].split(' ')
            labels = [int(float(lb)) for lb in line[3].split(' ')]
            labels = np.asarray(labels)

            text_a = ''.join(words)
            examples.append(
                InputExample(eid=eid, words=words, types=types, text_a=text_a, labels=labels))
        return examples


def convert_examples_to_features(examples: InputExample, label_list, bies_list, max_seq_length, tokenizer,
                                 ner_type_id: dict = None):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}
    bies_map = {label: i for i, label in enumerate(bies_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]

        # segment_ids = [0] * len(tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = ['S'] + get_bies_labels(tokens_a, example.words) + ['S']
        segment_ids = [bies_map[label] for label in segment_ids]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        labels = ''.join([str(lb) * len(word) for word, lb in zip(example.words, example.labels)])
        token_labels = convert_char_labels_to_token_labels(tokens_a, labels)
        label_ids = [0] + [label_map[label] for label in token_labels] + [0]
        label_ids += padding

        ner_label_ids = None
        if ner_type_id is not None:
            ner_label_ids = get_ner_labels(tokens_a, example.words, example.types, ner_type_id)
            ner_label_ids = [ner_type_id['O']] + ner_label_ids + [ner_type_id['O']]
            ner_label_ids += padding
            assert len(ner_label_ids) == max_seq_length

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length, '%d' % len(label_ids)

        if ex_index < 5:
            logger.info("*** Example ***")
            # logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %s)" % (''.join([str(lb) for lb in example.labels]),
                                                 "".join([str(x) for x in label_ids])))
            if ner_label_ids is not None:
                logger.info(
                    "ner labels: %s" % " ".join([str(x) for x in ner_label_ids]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_ids=label_ids,
                          ner_label_ids=ner_label_ids))
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
