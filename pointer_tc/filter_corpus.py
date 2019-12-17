# -*- coding: utf-8 -*-

import io

with io.open('E:\\dlprojects\\data\\title_compression\\ProductTitleSummarizationCorpus\\big_corpus\\big_corpus.txt', encoding='utf-8') as fin:
    cnt = 0
    max_tgt_len = 0
    for line in fin:
        arr = line.strip().split('\t')
        all_in = True
        title = arr[0]
        short_title = arr[2]
        if len(short_title) > max_tgt_len:
            max_tgt_len = len(short_title)
        for ch in short_title:
            if ch not in title:
                all_in = False
                break
        if not all_in:
            cnt += 1
    print(cnt)
    print(max_tgt_len)