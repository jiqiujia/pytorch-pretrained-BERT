# -*- coding: utf-8 -*-
import io
import random


def main():
    """对短标题中有不一样字的进行过滤；单个长标题随机取一个对应短标题"""
    with io.open('../../data/title_compression/short_title_data.txt', encoding='utf-8') as fin, \
            io.open("../../data/title_compression/short_title_data_filter.txt", "w+", encoding="utf-8") as fout, \
            io.open("../../data/title_compression/short_title_data_filter_sample.txt", "w+", encoding="utf-8") as fout2:
        lines = fin.readlines()
        cnt = 0
        id_set = set()
        for line in lines:
            line = line.strip()
            arr = line.split('\t')
            lid = int(arr[0])
            words = arr[1].split(' ')
            labels = [int(float(lb)) for lb in arr[3].split(' ')]
            chars = ''.join([word for word, lb in zip(words, labels) if lb == 1])
            short_title_chars = ''.join(arr[5].split(' '))
            ok = True
            for ch in short_title_chars:
                if ch not in chars:
                    ok = False
                    break
            if ok:
                cnt += 1
                print(cnt)
                fout.write(line + '\n')

                if lid not in id_set:
                    id_set.add(lid)
                    fout2.write(line + '\n')


if __name__ == '__main__':
    main()
