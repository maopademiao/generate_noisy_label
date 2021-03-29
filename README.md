# generate_noisy_label
In this github，we use five single noises type (random,uniform, symmetric, pair and white noise) and their combination types to generate noisy labels.

We introduce the details in English and Chinese in this readme file.

在此github中，我们通过五种单噪音模式及其组合模式进行噪音标签生成。

我们在这个readme文件中采用两种语言（英语和汉语）进行说明。

[toc]

## English Introduction

### Quick way to use

```shell script
sh generate_all_noisy_data.sh
```

### Data file location  introduction

We use four datasets, including chn,chngolden,trec and agnews.

Their directory is as follows

- /data/chnsenticorp/chn/
- /data/chnsenticorp/chngolden/

- /data/chnsenticorp/trec/
- /data/chnsenticorp/agnews/

In each directory, they include two files--train.json and test.json. They are training dataset and test dataset.

We also have a directory /white_text/white, it includes the white noise data. 

where  chinese3.txt is added chinese white noise，middlemarch_agnews.txt is English text for agnews sample；english_agnews.txt is English text for agnews white noise; middlemarch_trec is English text for trec white noise.

### Way to use

For single noise type

```python
python datanoisy.py --rate 0.1  --dataset chn --result_dir noisy_data --type pairflip --seed 0 --random_state 0
```
For multi moise type

```python
python datanoisy.py --rate 0.2  --dataset chn --result_dir noisy_data --multitype u_0.5_p_0.5 --seed 0 --random_state 0
```

all parameters：

| parameter    | meaning          | default     | optional                                                     |
| ------------ | ---------------- | ----------- | ------------------------------------------------------------ |
| rate         | noise rate       | 0.1         | value in zero to one                                         |
| dataset      | dataset name     | agnews      | agnews, trec, chn, chngolden                                 |
| result_dir   | result's dir     | noisedata   |                                                              |
| type         | noise type       | pairflip    | pairflip, symmetric, uniform, random, white                  |
| seed         | random seed      | 0           |                                                              |
| random_state | random seed      | 0           |                                                              |
| multitype    | multi type noise | u_0.5_p_0.5 | A_rate1_B_rate2, <br />where A/B can be p,s,u,r,w(first character for type)]<br />rate1 and rate2 The sum of A and B is equal to 1<br />where rate1 denotes A noise type rate, rate2 denotes B |

default result files will be in : 

-  for single noise: /result_dir/dataset_type_rate/train.json
- for multi noise: /result_dir/dataset_rate_A_rate1_B_rate2/train.json

for example: 

- for single noise: /noisedata/agnews_symmetric_0.1/train.json
- for multi noise: /noisedata/agnews_0.1_symmetric_0.4_uniform_0.6/train.json

the format of results is like this:

{"label": 2, "sentence": "Wall St. Bears Claw Back Into the Black (Reuters).Reuters - Short-sellers, Wall Street's dwindling\\band of ultra-cynics, are seeing green again."}

### Dataset details
| dataset             | train  | test | classes | details of classes                                           |
| ------------------- | ------ | ---- | ------- | ------------------------------------------------------------ |
| chnsenticorp        | 6989   | 483  | 2       | 0--negative, 1--positive                                     |
| chnsenticorp-golden | 4158   | 483  | 2       | 0--negative, 1--positive                                     |
| agnews              | 120000 | 7600 | 4       | *{'World': 0, 'Sports': 1, 'Business': 2, 'Sci/Tech': 3}*    |
| trec                | 5452   | 500  | 6       | *{'DESC': 0, 'ENTY': 1, 'ABBR': 2, 'HUM': 3, 'NUM': 4, 'LOC': 5}* |

| white noise                                                  | Used for dataset                      | total sentences |
| ------------------------------------------------------------ | ------------------------------------- | --------------- |
| Chinese history texts                                        | chn&chngolden                         | 7001            |
| Collection of English Novels<br />middlemarch<br />to kill a mockingbird<br/>Pride and Prejudice<br/>A Dance with Dragons <br/>Fifty Shades of Grey<br/>The Grapes of Wrath<br/>Walden<br/>Little Prince | agnews                                | 120467          |
| English novels《middlemarch》                                | trec(sentence length shorter than 15) | 5825            |

| dataset   | Noise rate calculation formula | Maximum noise |
| --------- | ------------------------------ | ------------- |
| chn       | 7001/6989                      | 1             |
| chngolden | 4246/4158                      | 1             |
| agnews    | 120467/120000                  | 1             |
| trec      | 5825/5452                      | 1             |

*********

## Chinese Introduction
### 快速使用

```shell script
sh generate_all_noisy_data.sh
```
### 数据位置说明

我们一共使用四个数据集, 包括chn,chngolden,trec and agnews.

目录如下

- /data/chnsenticorp/chn/
- /data/chnsenticorp/chngolden/
- /data/chnsenticorp/trec/
- /data/chnsenticorp/agnews/

在每一个目录下，都包括两个文件train.json和test.json，分别是训练数据集和测试数据集

另外有一目录 /white_text/white，包含白噪声数据文件

其中chinese3.txt添加的中文白噪声文本，middlemarch_agnews.txt用于agnews的英文白噪声文本样本；english_agnews是用于agnews的英文白噪声数据，middlemarch_trec.txt用于trec的英文白噪声文本

### 使用方法：

对于单模式噪声生成
```
python datanoisy.py --rate 0.1  --dataset chn --result_dir noisy_data --type pairflip --seed 0 --random_state 0
```
对于多模式混合噪声生成

```python
python datanoisy.py --rate 0.2  --dataset chn --result_dir noisy_data --multitype u_0.5_p_0.5 --seed 0 --random_state 0
```

全部参数设置：

| 参数名       | 含义           | 默认值    | 可选参数                                    |
| ------------ | -------------- | --------- | ------------------------------------------- |
| rate         | 噪声率         | 0.1       | 0~1之间的值                                 |
| dataset      | 数据集名称     | agnews    | agnews, trec, chn, chngolden                |
| result_dir   | 生成结果的目录 | noisedata |                                             |
| type         | 噪声生成方式   | pairflip  | pairflip, symmetric, uniform, random, white |
| seed         | 随机种子       | 0         |                                             |
| random_state | 随机种子       | 0         |                                             |
| multitype    | multi type noise | u_0.5_p_0.5 | A_rate1_B_rate2, <br />where A/B can be p,s,u,r,w(first character for type)]<br />rate1 and rate2 The sum of A and B is equal to 1<br />where rate1 denotes A noise type rate, rate2 denotes B |

默认结果文件会在目录: 

-  对于单一模式噪声: /result_dir/dataset_type_rate/train.json
- 对于多噪声混合: /result_dir/dataset_rate_A_rate1_B_rate2/train.json

比如: 

- 对于单一模式噪声: /noisedata/agnews_symmetric_0.1/train.json
- 对于多噪声混合: /noisedata/agnews_0.1_symmetric_0.4_uniform_0.6/train.json

生成格式为{"label": 2, "sentence": "Wall St. Bears Claw Back Into the Black (Reuters).Reuters - Short-sellers, Wall Street's dwindling\\band of ultra-cynics, are seeing green again."}

***

### 数据集详情

| 数据集              | train  | test | 类别数 |                                                              |
| ------------------- | ------ | ---- | ------ | ------------------------------------------------------------ |
| chnsenticorp        | 6989   | 483  | 2      | 0负1正                                                       |
| chnsenticorp-golden | 4158   | 483  | 2      | 0负1正                                                       |
| agnews              | 120000 | 7600 | 4      | *{'World': 0, 'Sports': 1, 'Business': 2, 'Sci/Tech': 3}*    |
| trec                | 5452   | 500  | 6      | *{'DESC': 0, 'ENTY': 1, 'ABBR': 2, 'HUM': 3, 'NUM': 4, 'LOC': 5}* |



| white noise                                                  | 用于数据集           | 总数   |
| ------------------------------------------------------------ | -------------------- | ------ |
| 中文历史文本                                                 | chn&chngolden        | 7001   |
| 英文小说合集<br />middlemarch<br />to kill a mockingbird<br/>Pride and Prejudice<br/>A Dance with Dragons <br/>Fifty Shades of Grey<br/>The Grapes of Wrath<br/>Walden<br/>Little Prince | agnews               | 120467 |
| 英文小说《middlemarch》                                      | trec(句子长度小于15) | 5825   |



| 数据集    | 计算公式      | 最高噪声 |
| --------- | ------------- | -------- |
| chn       | 7001/6989     | 1        |
| chngolden | 4246/4158     | 1        |
| agnews    | 120467/120000 | 1        |
| trec      | 5825/5452     | 1        |

