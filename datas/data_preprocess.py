import random

import numpy as np

from Community import config
import re
from transformers import BertTokenizer, BertModel
import pickle
from tqdm import tqdm
import torch
import pandas as pd
import os

tokenizer = BertTokenizer.from_pretrained(config.bert_path)

def get_nick_name(dataset):
    nick_names = []
    with open('{}.txt'.format(dataset), 'r') as rf:
        datas = [each.strip().split(':')[1] for each in rf.readlines()]
        for data in datas:
            words = data.lower().split()
            for w in words:
                if '@' in w:
                    w = w.split('@')[1]
                    if w not in tokenizer.vocab:
                        nick_name = w
                        if nick_name not in nick_names:
                            nick_names.append(nick_name)

    return nick_names

def clean_str(string):
    other_char = re.compile(r"[^A-Za-z0-9(),!?\']", flags=0)
    string = re.sub(other_char, " ", string)
    string = re.sub(" https.+? ", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r" ", " ", string)
    string = string.replace('\\', '')
    return string.strip().lower()

def data_clean(dataset):
    print('clean data')
    nick_names = get_nick_name(dataset)
    communities = {}    # save the index of nickname
    for each in nick_names:
        communities[each] = []
    with open('{}.txt'.format(dataset), 'r') as rf:
        datas = [each.strip().lower() for each in rf.readlines()]
    data_without_nick = []
    for i in tqdm(range(len(datas))):
        data = datas[i].split(':')
        words = data[1].replace('@', ' ').split()
        sen_wo_nick = []
        for w in words:
            if w in nick_names:
                communities[w].append(i)
            else:
                sen_wo_nick.append(w)
        sen_wo_nick = clean_str(' '.join(sen_wo_nick))
        data_without_nick.append(data[0]+':'+sen_wo_nick)

    with open('{}/clean_data.txt'.format(dataset), 'w') as wf:
        wf.write('\n'.join(data_without_nick))
    with open('{}/community.pkl'.format(dataset), 'wb') as pkl:
        pickle.dump(communities, pkl)

def get_vocabs():
    vocabs = []
    with open('{}/clean_data.txt'.format(config.dataset), 'r') as rf:
        datas = [each.strip().lower().split(':')[1] for each in rf.readlines()]
        for data in datas:
            tokens = data.split(' ')
            for token in tokens:
                if token not in vocabs:
                    vocabs.append(token)
    with open(config.dataset + '/vocabs.txt', 'w') as wf:
        wf.write('\n'.join(vocabs))

def get_fold():
    Yes = []
    No = []
    with open('{}/clean_data.txt'.format(config.dataset), 'r') as rf:
        datas = [each.strip() for each in rf.readlines()]
        for data in datas:
            if data[0] != '0':
                Yes.append(data)
            else:
                No.append(data)
    print(len(Yes), len(No))
    # prepare data for 10 folds
    yes10 = int(len(Yes) / 10)
    no10 = int(len(No) / 10)
    SplitYes, SplitNo = [], []
    for i in range(9):
        SplitYes.append(Yes[i * yes10:(i + 1) * yes10])
    SplitYes.append(Yes[yes10 * 9:])
    for i in range(9):
        SplitNo.append(No[i * no10:(i + 1) * no10])
    SplitNo.append(No[no10 * 9:])
    for i in range(10):
        train = []
        test = []
        for j in range(10):
            if i == j:
                test.extend(SplitYes[j])
                test.extend(SplitNo[j])
            else:
                train.extend(SplitYes[j])
                train.extend(SplitNo[j])
        if 'fold' not in os.listdir('{}'.format(config.dataset)):
            os.makedirs('{}/fold'.format(config.dataset))
        with open('{}/fold/fold{}train.txt'.format(config.dataset, str(i)), 'w') as wf:
            Trains = []
            for each in train:
                idx = datas.index(each)  # 获取每个句子的id
                Trains.append(str(idx) + ':' + each)
            # random.shuffle(Trains)
            wf.write('\n'.join(Trains))
        with open('{}/fold/fold{}test.txt'.format(config.dataset, str(i)), 'w') as wf:
            Tests = []
            for each in test:
                idx = datas.index(each)  # 获取每个句子的id
                Tests.append(str(idx) + ':' + each)
            # random.shuffle(Tests)
            wf.write('\n'.join(Tests))

def get_random():
    Yes = []
    No = []
    with open('{}/clean_data.txt'.format(config.dataset), 'r') as rf:
        datas = [each.strip() for each in rf.readlines()]
        for data in datas:
            if data[0] != '0':
                Yes.append(data)
            else:
                No.append(data)
    # print(len(Yes), len(No))
    SplitYes, SplitNo = int(0.7*len(Yes)), int(0.7*len(No))
    if 'fold' not in os.listdir('{}_rand'.format(config.dataset)):
        os.makedirs('{}_rand/fold'.format(config.dataset))
    for i in range(10):
        train = Yes[:SplitYes] + No[:SplitNo]
        test = Yes[SplitYes:] + No[SplitNo:]
        print(len(train), len(test))
        random.shuffle(train)
        random.shuffle(test)
        with open('{}_rand/fold/fold{}train.txt'.format(config.dataset, str(i)), 'w') as wf:
            Trains = []
            for each in train:
                idx = datas.index(each)  # 获取每个句子的id
                Trains.append(str(idx) + ':' + each)
            # random.shuffle(Trains)
            wf.write('\n'.join(Trains))
        with open('{}_rand/fold/fold{}test.txt'.format(config.dataset, str(i)), 'w') as wf:
            Tests = []
            for each in test:
                idx = datas.index(each)  # 获取每个句子的id
                Tests.append(str(idx) + ':' + each)
            # random.shuffle(Tests)
            wf.write('\n'.join(Tests))


if __name__ == '__main__':
    # if config.dataset == 'HatEval':
    #     datas = []
    #     frame = pd.read_csv('HatEval.tsv', sep='\t')
    #     contexts = list(frame['text'].values)
    #     labels = list(frame['HS'].values)
    #     for context, label in zip(contexts, labels):
    #         context = context.replace(':', '')
    #         datas.append(str(label)+":"+context)
    #     with open('HatEval.txt', 'w') as wf:
    #         wf.write('\n'.join(datas))
    # if config.dataset == 'Davids':
    #     with open('Davids_ori.txt', 'r') as rf:
    #         datas = [each.strip().split('\t') for each in rf.readlines()]
    #     outs = []
    #     for data in datas:
    #         outs.append(str(data[0]) + ":" + data[1].replace(':', ''))
    #     with open('Davids.txt', 'w') as wf:
    #         wf.write('\n'.join(outs))

    data_clean(config.dataset)
    # get_vocabs()
    # get_fold()
    # get_random()   # get random split for the datasets
    pass