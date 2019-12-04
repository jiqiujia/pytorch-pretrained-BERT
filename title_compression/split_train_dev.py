# -*- coding: utf-8 -*-

import io
import sys
import os
import random

if __name__ == '__main__':
    infile = sys.argv[1]
    outpath = sys.argv[2]
    train_num = int(sys.argv[3])
    dev_num = int(sys.argv[4])

    with io.open(infile, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()
        random.shuffle(lines)

        train_lines = lines[:train_num]
        dev_lines = lines[-dev_num:]

        with io.open(os.path.join(outpath, 'train.tsv'), 'w+', encoding='utf-8') as fout:
            for line in train_lines:
                fout.write(line)
        with io.open(os.path.join(outpath, 'dev.tsv'), 'w+', encoding='utf-8') as fout:
            for line in dev_lines:
                fout.write(line)