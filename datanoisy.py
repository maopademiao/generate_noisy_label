import numpy as np
import random
import torch
import sys
import json
import argparse
import os
import time
from numpy.testing import assert_array_almost_equal


parser = argparse.ArgumentParser()
parser.add_argument('--rate', type=float, help='corruption rate, should be less than 1', default=0.4)
parser.add_argument('--dataset', type=str, help='agnews,trec,chn,chngolden', default='trec')
parser.add_argument('--result_dir', type=str, help='dir to save result txt files', default='noisedata')
parser.add_argument('--type', type=str, help='[pairflip, symmetric, uniform, random, white]', default='uniform')
parser.add_argument('--multitype', type=str, help='[A_rate_B_rate,A/B can be p,s,u,r,w(short for type)]', default=None)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--random_state', type=int, default=0)
parser.add_argument('--peer', dest='peer', action='store_true', default=False)

args = parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    np.random.seed(seed)              # Numpy module
    random.seed(seed)                 # Python random module

import pandas as pd
def reform_data_for_peer(file_path, folder_dir):
    csv_path = os.path.join(folder_dir, 'data.csv')
    with open(file_path, 'r', encoding='utf8') as fin:
        data = [json.loads(l.strip()) for l in fin.readlines()]
    csv_data = []
    head = ['target']
    print('dim: ', len(data[0]['sentence']))
    for i in range(len(data[0]['sentence'])):
        head.append(str(i+1))
    csv_data.append(head)
    for d in data:
        label = d['label']
        sent = d['sentence']
        csv_data.append([label] + sent)
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_path, index=False, header=False)

def normalize(matrix): # 使概率和为1，每个值除以每一行的和
    m, n = np.shape(matrix)
    for i in range(0,m):
        row = matrix[i,:]
        rowsum = row.sum()
        for j in range(0,n):
            matrix[i][j] /= rowsum
    return  matrix

def noisify_random(nb_classes, noise_rate):
    """
    P = (1-p)I+p*△，p is noise_rate
    where I is the identity matrix,单位矩阵
    and ∆ is a matrix with zeros along the diagonal 对角线为0
    and remaining entries of each column are drawn uniformly and independently from the K−1-dimensional unit simplex.
    其余每列元素均匀 独立于k-1维unit simplex

    unit simplex: https://zh.wikipedia.org/wiki/%E5%8D%95%E7%BA%AF%E5%BD%A2
    (也称单纯形采点) 单位K-1-单纯形中产生有效产生均匀分布的采样方法。
    基于从K-1维单位单纯形采样等价于从参数α = (α1, ..., αK)都等于1的狄利克雷分布中采样的事实。
        确切的流程为：
        1. 产生K个服从单位指数分布的随机数x1, ..., xK.
        2. 这可以通过产生K个在开区间(0,1)中均匀分布的随机数yi然后取 xi=-ln(yi).
        3. 令S为xi之和。
            单位单纯形中的点的K个坐标t1, ..., tK由ti=xi/S给出
    :param nb_classes: 类别数量
    :param noise_rate:
    :return: P
    """
    I = np.eye(nb_classes,nb_classes)
    Tri = np.random.rand(nb_classes,nb_classes)
    Tri = -np.log(Tri)
    for i in range(0, nb_classes):
        row = Tri[i,:]
        rowsum = row.sum()
        for j in range(0,nb_classes):
            if i==j:
                Tri[i][j] = 0
            else:
                Tri[i][j] /= rowsum

    tmpa = normalize(Tri)*noise_rate
    tmpb = I*(1-noise_rate)+tmpa
    tmpb = normalize(tmpb)
    P = I*(1-noise_rate) + noise_rate*Tri
    # print("未使和为1")
    # print(P)
    P = normalize(P)
    # print(sys._getframe().f_code.co_name)
    # print("classes:", nb_classes, " noise rate:", noise_rate)
    # print("noise matrix:\n",P)
    return tmpb

def noisify_uniform(nb_classes, noise_rate):
    """
    P = (1-p)I + pⅡ, p is noise_rate, K is nb_classes
    Here, I represents the identity matrix
    and
    II denotes a matrix with zeros along the diagonal 对角线为0，其余元素符合均匀分布(0,1)
    :param nb_classes:
    :param noise_rate:
    :return:
    """
    I = np.eye(nb_classes, nb_classes)
    II = np.ones((nb_classes,nb_classes))
    II = np.linalg.matrix_power(II, nb_classes)
    tmpa = np.random.uniform(0,1,(nb_classes,nb_classes-1))
    tmpf = np.zeros(nb_classes)
    tmpa = np.insert(tmpa, nb_classes-1, values=tmpf, axis=1)# 随意添加最后一列，变成kXk矩阵
    tmpe = np.zeros((nb_classes,nb_classes))
    for i in range(0,nb_classes):
        for j in range(0,nb_classes):
            if i>j:
                tmpe[i][j]=tmpa[i][j]
            if i<j:
                tmpe[i][j] = tmpa[i][j-1]
            if i==j:
                continue
    # np.fill_diagonal(tmpa, 0)
    tmpb = normalize(tmpe)
    tmpc = tmpb*noise_rate+(1-noise_rate)*I
    tmpd = normalize(tmpc)
    # P = (1-noise_rate)*I + II/nb_classes*noise_rate
    # P = normalize(P)
    return tmpd

# noisify_pairflip call the function "multiclass_noisify"
def noisify_pairflip(nb_classes, noise_rate):
    """
    P =
    :param nb_classes:
    :param noise_rate:
    :return:
    """
    P = np.eye(nb_classes, nb_classes)
    n = noise_rate
    # 0 -> 1
    P[0, 0], P[0, 1] = 1. - n, n
    for i in range(1, nb_classes-1):
        P[i, i], P[i, i + 1] = 1. - n, n
    P[nb_classes-1, nb_classes-1], P[nb_classes-1, 0] = 1. - n, n

    return P

def noisify_symmetric(nb_classes, noise_rate):

    P = np.ones((nb_classes, nb_classes))
    n = noise_rate
    P = (n / (nb_classes - 1)) * P
    # 0 -> 1
    P[0, 0] = 1. - n
    for i in range(1, nb_classes-1):
        P[i, i] = 1. - n
    P[nb_classes-1, nb_classes-1] = 1. - n

    return P


def noisify_white(nb_classes, args, totallinesnum, datasetpath, writejsonpath, dictTypeRate=None):
    """
    :param nb_classes: 总类别数 chn 2, chngolden 2, trec 6, agnews 4
    :param args:
    :param totallinesnum: args.dataset的训练行数
    :param datasetpath: args.dataset的路径
    :param writejsonpath: 写入json
    :return:
    """
    list = []
    for i in range(0,nb_classes):
        list.append(i)
    if args.dataset=='chn' or args.dataset=='chngolden':
        whiteNoisyFile = './white_text/white/chinese3.txt'
    elif args.dataset=='agnews':
        whiteNoisyFile = './white_text/white/english_agnews.txt'
    elif args.dataset=='trec':
        whiteNoisyFile = './white_text/white/middlemarch_trec.txt'
    print(dictTypeRate)
    if dictTypeRate is None:
        addlinesnum = int(args.rate * totallinesnum)
    elif "white" in dictTypeRate:
        addlinesnum = int(args.rate * float(dictTypeRate['white'])*totallinesnum)
    # print(totallinesnum)
    # print(addlinesnum)
    # print(int(addlinesnum))
    noisyTotalLineNum = 0
    with open(whiteNoisyFile, 'r', encoding='utf-8') as f:
        for lines in f:
            noisyTotalLineNum+=1
    # print(noisyTotalLineNum)

    assert addlinesnum <= noisyTotalLineNum,'noise_rate is high for this dataset'
    with open(datasetpath, 'r', encoding='utf-8') as f:
        with open(writejsonpath, 'w', encoding='utf-8') as writef:
            for line in f:
                writef.write(line)

    # 对白噪声数据随机分配标签，按照类别数顺序分配，如4类，按0,1,2,3，0,1,2,3,0,1,2,3...顺序依次分配
    with open(whiteNoisyFile, 'r', encoding='utf-8') as f:
        with open(writejsonpath, 'a', encoding='utf-8') as writef:
            num = 0
            lines = f.readlines()
            for linenum in range(0,addlinesnum):
                line = lines[linenum]
                text = line.strip('\n')
                label = num
                num += 1
                if num%nb_classes==0:
                    num=0
                # print(label,text)
                dict = {}
                dict["label"] = label
                dict["sentence"] = text
                ddata = json.dumps(dict, ensure_ascii=False)
                writef.write(ddata)
                writef.write('\n')
    print("already add",addlinesnum,"white noisy data and label")
    realNoise = addlinesnum/(addlinesnum+totallinesnum)
    return realNoise,addlinesnum


# basic function
def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    # print(np.max(y), P.shape[0])
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    # print("total lines:",m)
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        temp1 = P[i, :]
        temp2 = P[i, :][0]
        temp3 = flipper.multinomial(1, P[i, :][0], 1)
        flipped = flipper.multinomial(1, P[i, :][0], 1)[0]
        temp4 = np.where(flipped==1)
        temp5 = temp4[0]
        new_y[idx] = np.where(flipped == 1)[0]

    # for i in range(0,10):
    #     print(y[:10,][i][0],end=' ')
    # print("\n")
    # for i in range(0,10):
    #     print(new_y[:10,][i][0],end=' ')
    # print("\n")

    new_y_list = []
    for i in range(0,len(new_y)):
        new_y_list.append(new_y[:len(new_y),][i][0])
    # print(new_y_list[:10])
    return new_y_list

def dealdataset(path): # 返回原始标签，该数据集类别数，文本内容
    # path = "./data/ChnSentiCorp/newtrain1.json"
    labels = []
    classes = []
    texts = []
    labelslist = []
    with open(path, 'r', encoding='utf8')as fp:
        lines = fp.readlines()
        for line in lines:
            line = json.loads(line)
            label = int(line.get('label'))
            labels.append(label)
            if label not in classes:
                classes.append(label)
            text = line.get('sentence')
            # print(labels)
            texts.append(text)
            # break
    labellist = labels
    labels = np.array(labels)
    train_labels = torch.from_numpy(labels).long()
    train_labels = np.asarray([[train_labels[i]] for i in range(len(train_labels))])
    # print(train_labels.shape)
    nb_classes = len(classes)
    return train_labels,nb_classes,texts, labellist


def  writeOutputFile(labels,texts,path):
    with open(path, 'w', encoding='utf-8') as f:
        assert len(labels)==len(texts)
        for i in range(0,len(labels)):
            dict = {}
            dict['label'] = labels[i]
            dict['sentence'] = texts[i]
            dict['label'] = int(dict['label'])
            ddata = json.dumps(dict, ensure_ascii=False)
            f.write(ddata)
            f.write('\n')

def calRealNoisy(truelabels,noisylabels):
    assert len(truelabels) == len(noisylabels)
    different = 0
    for i in range(0,len(truelabels)):
        if truelabels[i]!=noisylabels[i]:
            different+=1
    realNoise = different/len(truelabels)
    return realNoise

def changename(char):
    str=""
    incharlist = ['p','r','s','u','w']
    assert char in incharlist
    if char=='p': str='pairflip'
    elif char=='r': str='random'
    elif char=='s': str='symmetric'
    elif char == 'u': str='uniform'
    elif char =='w':str='white'
    return str


from shutil import copyfile
def copy_test_data(file_path, folder_dir):
    target = os.path.join(folder_dir, os.path.basename(file_path))
    copyfile(file_path, target)

def exchangeNoisyName(c):
    if c == 'pairflip':
        str = noisify_pairflip
    elif c == 'random':
        str = noisify_random
    elif c == 'symmetric':
        str = noisify_symmetric
    elif c == 'uniform':
        str = noisify_uniform
    elif c == 'white':
        str = noisify_white
    return str

def MultiTypeNoise(dictTypeRate,nb_classes,noise_rate,y):
    listP =[]
    arr = np.zeros(len(dictTypeRate))
    num = 0
    for type in dictTypeRate:
        arr[num] = float(dictTypeRate[type])
        num+=1
        if type!='white':
            functionName = exchangeNoisyName(type)
            P=functionName(nb_classes,noise_rate)
            print(type+"_"+str(dictTypeRate[type]),"noisy matrix:\n",P)
            # print("noisy matrix:\n", P)
            assert P.shape[0] == P.shape[1]
            assert np.max(y) < P.shape[0]
            assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
            assert (P >= 0.0).all()
        else:
            P=np.eye(nb_classes,nb_classes)
        listP.append(P)

    m = y.shape[0]
    new_y = y.copy()

    flipper = np.random.RandomState(args.random_state)
    flipper2 = np.random.RandomState(args.random_state)
    for idx in np.arange(m):
        flipped = flipper.multinomial(1, arr, 1)[0]
        whereEqualsOne = np.where(flipped == 1)[0][0]
        P = listP[whereEqualsOne]
        i = y[idx]
        flipped2 = flipper2.multinomial(1, P[i, :][0], 1)[0]
        new_y[idx] = np.where(flipped2 == 1)[0]

    new_y_list = []
    for i in range(0,len(new_y)):
        new_y_list.append(new_y[:len(new_y),][i][0])
    return new_y_list


if __name__ == '__main__':
    noise_rate = args.rate
    noise_type = args.type
    multi_type = args.multitype
    random_state = args.random_state
    seed = args.seed
    set_seed(seed)
    dataset = args.dataset
    result_dir = args.result_dir
    save_dir = result_dir + '/' + dataset + '_%s' % noise_type + '_%s' % noise_rate + '/' #''_%s' % noise_type + '_%s' % noise_rate
    if multi_type is None:
        save_dir = result_dir + '/' + dataset + '_%s' % noise_type + '_%s' % noise_rate + '/'  # ''_%s' % noise_type + '_%s' % noise_rate
    else:
        save_dir = result_dir + '/' + dataset + '_%s' % noise_rate + '_%s' % multi_type + '/'  # save_dir: noisedata/trec_0.4_p_0.5_r_0.5/
    save_file = 'train.json'
    print("save_dir:", save_dir)

    if not os.path.exists(save_dir):
        os.system('mkdir -p %s' % save_dir)
    print("output_file:",save_dir+save_file)

    if dataset == 'chn':
        path = "./data/chn/train.json"
        test_file = './data/chn/test.json'
    elif dataset == 'chngolden':
        path = "./data/chngolden/train.json"
        test_file = './data/chngolden/test.json'
    elif dataset == 'agnews':
        path = "./data/agnews/train.json"
        test_file = './data/agnews/test.json'
    elif dataset =='trec':
        path = "./data/trec/train.json"
        test_file = './data/trec/test.json'
    else:
        print("no %s dataset",dataset)

    if args.peer:
        path = path.replace('train.json', 'train_peer.json')

    flag = 0
    if noise_type == 'pairflip':#, , unifrom, random, white
        functionName = noisify_pairflip
    elif noise_type == 'symmetric':
        functionName = noisify_symmetric
    elif noise_type == 'uniform':
        functionName = noisify_uniform
    elif noise_type == 'random':
        functionName = noisify_random
    elif noise_type == 'white':
        flag = 1
    else:
        print("no %s noise type", noise_type)

    labels, nb_classes, texts, labelslist = dealdataset(path) # 原始标签array，类别总数， 原始文本,原始标签list
    print("classes:", nb_classes)
    print("dataset lines:", labels.shape[0])
    print("noisy type:", noise_type)
    print("noisy rate:", noise_rate)
    if multi_type is None:
        if flag == 0:
            P = functionName(nb_classes,noise_rate) #噪声转移矩阵 nb_classes*nb_classes
            y_train_noisy = multiclass_noisify(labels, P=P, random_state=random_state) #转移后的标签list
            print("noisy matrix:\n",P)
            writeOutputFile(y_train_noisy,texts,save_dir+save_file)
            realNoise = calRealNoisy(labelslist,y_train_noisy)
            print("realnoise:",realNoise)
        else:
            realNoise,_ = noisify_white(nb_classes, args, len(labelslist), path, save_dir+save_file)
            print("realnoise:", realNoise)
    else:
        dictTypeRate = {}
        typelist = multi_type.split('_')
        for i in range(0,len(typelist),2):
            typename = changename(typelist[i])
            dictTypeRate[typename] = float(typelist[i+1])
        print(dictTypeRate)
        y_train_noisy = MultiTypeNoise(dictTypeRate,nb_classes,noise_rate,labels)
        if "white" not in dictTypeRate:
            writeOutputFile(y_train_noisy, texts, save_dir + save_file)
            realNoise = calRealNoisy(labelslist, y_train_noisy)
            print("realnoise:", realNoise)
        else:
            tmppath = "tmp.json"
            writeOutputFile(y_train_noisy, texts, tmppath)
            realNoise1 = calRealNoisy(labelslist, y_train_noisy)
            realNoise2,addsum = noisify_white(nb_classes, args, len(labelslist), tmppath, save_dir + save_file,dictTypeRate)
            print("realnoise:", (realNoise1*labels.shape[0]+addsum)/(labels.shape[0]+addsum))
            os.system('rm %s' % tmppath)

        # time.sleep(300)
    copy_test_data(test_file, save_dir)
    if args.peer:
        reform_data_for_peer(path, save_dir)
    else:
        copy_test_data(test_file, save_dir)



# 0 1 1 0 0 1 1 1 1 1
# pair
# [[0.8 0.2]
#  [0.2 0.8]]
# 0 1 1 1 0 1 0 1 0 0

# 0 1 1 0 0 1 1 1 1 1
# symmetric
# 0 1 1 1 0 1 0 1 0 0

# 0 1 1 0 0 1 1 1 1 1
# uniform
# 0 1 1 1 0 1 0 1 1 0


# 0 1 1 0 0 1 1 1 1 1
# random
# 0 1 1 0 0 1 0 1 1 1