import re
def dealwhite():
    path = "./data/white/chinese1.txt"
    path2 = "./data/white/chinese2.txt"
    with open(path,'r',encoding='utf-8') as f1:
        with open(path2,'w',encoding='utf-8') as f2:
            i=0
            for lines in f1:
                i+=1
                line = lines.strip().strip('\n')
                newline = line.split('。')
                for j in newline:
                    if len(j)<=10:
                        continue
                    else:
                        f2.write(j+'。')
                        f2.write('\n')
                # print(newline)
                # print(line)
                # break

def dealwhiteenglish():
    path = "./data/white/middlemarch.txt"
    path2 = "./data/white/middlemarch_agnews.txt"
    path3 = "./data/white/middlemarch_trec.txt" #句子长度小于15，用于trec
    with open(path,'r',encoding='utf-8') as f1:
        with open(path2,'w',encoding='utf-8') as f2:
            for lines in f1:
                # lines = "apple is a a a good fruit. Is it a a right? Yes,it a a is."
                line = lines.strip().strip('\n')
                newline = line.split('.')
                # print(line)
                # print(newline)
                for j in newline:
                    j = j.strip()
                    if len(j.split(' '))<5:
                        continue
                    else:
                        # print(j)
                        k = j.split('?')
                        # print(k)
                        if len(k)>1:
                            for each in k:
                                each = each.strip()
                                if len(each.split(' ')) < 5:
                                    continue
                                else:
                                    f2.write(each + '?')
                                    f2.write('\n')
                        else:
                            f2.write(j + '.')
                            f2.write('\n')
    num = 0
    with open(path2, 'r', encoding='utf-8') as f:
        with open(path3,'w',encoding='utf8') as f3:
            for lines in f:
                line = lines.strip('\n').split()
                if len(line)<14:
                    num+=1
                    f3.write(lines)
    print(num)
import json
def test():
    path = './noisedata/trec/trec_white_0.4.json'
    with open(path,'r',encoding='utf-8') as f:
        lines = f.readlines()
        line = json.loads(lines[7621])
        text = line["sentence"]
        print(line)
        print(text)

# dealwhite() #4246
# dealwhiteenglish() #14612(add? for agnews)  5825(for trec length<15
# test()

