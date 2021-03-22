# -*- coding: UTF-8 -*-
# @Time    : 2021/3/7
# @Author  : xiangyuejia@qq.com
# Apache License
# Copyright©2020-2021 xiangyuejia@qq.com All Rights Reserved
import json
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--input_label', type=str, help='input file', default='')
parser.add_argument('--input_emb', type=str, help='input file', default='')
parser.add_argument('--output_file', type=str, help='output file', default='')

args = parser.parse_args()

with open(args.input_label, 'r', encoding='utf8') as fin:
    data_label = [l.strip() for l in fin.readlines()]


data_emb = []
with open(args.input_emb, 'r') as fin:
    ot = fin.readline()
    while ot:
        t = ot.strip()
        if len(t) == 0:
            continue
        o = json.loads(t)
        data_emb.append(o['features'][0]['layers'][0]['values'])
        ot = fin.readline()

print('len of label: ', len(data_label))
print('len of emb: ', len(data_emb))
with open(args.output_file, 'w', encoding='utf8') as fout:
    for label, emb in zip(data_label, data_emb):
        piece = {"label": label, "sentence": emb}
        print(json.dumps(piece), file=fout)
#
# out_data = []
# # head = ['target']
# # print('dim: ', len(data_json[0]))
# # for i in range(len(data_json[0])):
# #     head.append(str(i+1))
# out_data.append(head)
# for a,b in tqdm(zip(data_xlsx, data_json)):
#     out_data.append([a[0]] + b)
#
# df = pd.DataFrame(out_data)
# df.to_csv(file_out, index=False, header=False)
