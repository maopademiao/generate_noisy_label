# generate_noisy_label
nlp generate noisy label

data-|chnsenticorp  其中newtest.json 金标测试集；newtrain1.json原始训练集；newtrain2.json金标准训练集

​		-|agnews_csv	其中train.json 训练集；test.json测试集

​		-|trec	其中train.json 训练集；test.json测试集

​		-|white 其中chinese2.txt添加的中文白噪声文本，middlemarch_agnews.txt用于agnews的英文白噪声文本；middlemarch_trec.txt用于trec的英文白噪声文本

### 使用方法：

```
python datanoisy.py --rate 0.1  --dataset chn --result_dir noisedata --type pairflip --seed 0 --random_state 0
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

结果文件会在 result_dir/dataset/dataset_type_rate.json

比如 noisedata/agnews/agnews_symmetric_0.1.json

生成格式为{"label": 2, "sentence": "Wall St. Bears Claw Back Into the Black (Reuters).Reuters - Short-sellers, Wall Street's dwindling\\band of ultra-cynics, are seeing green again."}



***

### 注意

使用white的方式生成noise时，因为噪声数据有的不够，有的数据集noise rate有限制

| 数据集              | train  | test | 类别数 |                                                              |
| ------------------- | ------ | ---- | ------ | ------------------------------------------------------------ |
| chnsenticorp        | 6989   | 483  | 2      | 0负1正                                                       |
| chnsenticorp-golden | 4158   | 483  | 2      | 0负1正                                                       |
| agnews              | 120000 | 7600 | 4      | *{'World': 0, 'Sports': 1, 'Business': 2, 'Sci/Tech': 3}*    |
| trec                | 5452   | 500  | 6      | *{'DESC': 0, 'ENTY': 1, 'ABBR': 2, 'HUM': 3, 'NUM': 4, 'LOC': 5}* |



| white noise             | 用于数据集           | 总数  |
| ----------------------- | -------------------- | ----- |
| 中文历史文本            | chn&chngolden        | 4246  |
| 英文小说《middlemarch》 | agnews               | 14612 |
| 英文小说《middlemarch》 | trec(句子长度小于15) | 5825  |



| 数据集    | 计算公式     | 最高噪声 |
| --------- | ------------ | -------- |
| chn       | 4246/6989    | 0.6075   |
| chngolden | 4246/4158    | 1        |
| agnews    | 14612/120000 | 0.12177  |
| trec      | 5825/5452    | 1        |

