# -*- coding: utf-8 -*-

import io
from collections import defaultdict

BIOES = ['B', 'I', 'E', 'S']
with io.open('../../data/title_compression/short_title_data.txt', encoding='utf-8') as fin:
    type_cnt = defaultdict(lambda: 0)
    type_words = defaultdict(lambda: set())
    word_cnt = defaultdict(lambda: 0)
    for line in fin.readlines():
        arr = line.split('\t')
        types = arr[2].split(' ')
        words = arr[1].split(' ')
        for word, type in zip(words, types):
            type_words[type].add(word)
            type_cnt[type] += 1
            word_cnt[word] += 1

    type_cnt = sorted(list(type_cnt.items()), key=lambda x: x[1])
    print(type_cnt[0])
    with io.open('type_cnt.txt', 'w+', encoding='utf-8') as fout:
        for idx, type in enumerate(type_cnt):
            fout.write('%d\t%s\t%d\n' % (idx, type[0], type[1]))

    with io.open('type_words.txt', 'w+', encoding='utf-8') as fout:
        for key, val in type_words.items():
            fout.write(key + '\t' + ' '.join(val) + '\n')

    types = type_words.keys()
    with io.open('tgt_vocab.txt', 'w+', encoding='utf-8') as fout:
        for type in types:
            for lb in BIOES:
                fout.write(lb + '_' + type + '\n')
        fout.write('O\n')

    words = [word for word, cnt in word_cnt.items() if cnt >= 5]
    with io.open('src_vocab.txt', 'w+', encoding='utf-8') as fout:
        for word in words:
            fout.write(word + '\n')
