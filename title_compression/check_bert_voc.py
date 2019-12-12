# -*- coding: utf-8 -*-
import io
from collections import defaultdict
from hanziconv import HanziConv

with io.open('../../data/bert/chinese_L-12_H-768_A-12/vocab.txt', encoding='utf-8') as fin1, \
    io.open('../../data/title_compression/short_title_data.txt', encoding='utf-8') as fin2:
    bert_vocab = set()
    for line in fin1:
        bert_vocab.add(line.strip().replace("##", ""))
    out_of_vocab_dict = defaultdict(lambda: 0)
    for line in fin2:
        arr = line.strip().split('\t')
        arr[1] = HanziConv.toSimplified(arr[1])
        for ch in arr[1]:
            if ch not in bert_vocab:
                out_of_vocab_dict[ch] += 1
    print(len(out_of_vocab_dict))
    for item in out_of_vocab_dict.items():
        print(item)