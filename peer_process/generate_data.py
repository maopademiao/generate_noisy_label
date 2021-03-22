# -*- coding: UTF-8 -*-
# @Time    : 2021/3/7
# @Author  : xiangyuejia@qq.com
# Apache License
# Copyright©2020-2021 xiangyuejia@qq.com All Rights Reserved
import json
from tqdm import tqdm
from extract_feature import BertVector
bert = BertVector()

file_in = 'train_text'
file_out = 'train_enc'
with open(file_in, 'r', encoding='utf8') as fin:
    data = [l.strip() for l in fin.readlines()]

outdata = []
for d in tqdm(data):
    print(d)
    v = bert.encode([d])
    outdata.append([d, v])

with open(file_out, 'w', encoding='utf8') as fout:
    json.dump(fout, outdata)
