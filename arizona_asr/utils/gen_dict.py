# -*- coding: utf-8 -*-

import os
from collections import Counter

def gen_dict(transcript_path: str=None, save_dir: str='./.denver/asr/'):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    dictionary = os.path.join(save_dir, 'dict.ltr.txt')
    
    with open(transcript_path, encoding='utf-8') as f:
        data = f.read().splitlines()
    
    words = [d.split('\t')[1].upper() for d in data]

    letters = [d.replace(' ','|') for d in words]
    letters = [' '.join(list(d)) + ' |' for d in letters]

    chars = [l.split() for l in letters]
    chars = [j for i in chars for j in i]
    char_stats = list(Counter(chars).items())
    char_stats = sorted(char_stats, key=lambda x : x[1], reverse = True)
    char_stats = [c[0] + ' ' + str(c[1]) for c in char_stats]
    
    with open(dictionary, 'w') as f:
        f.write('\n'.join(char_stats))
