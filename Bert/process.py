import numpy as np
from transformers import BertModel, BertConfig, BertTokenizer, DistilBertTokenizer, DistilBertModel
import config
import random
from tqdm import tqdm
import torch
import config
import pickle

class DataIter():
    def __init__(self, is_train, fold=0):
        super(DataIter).__init__()
        self.is_train = is_train
        # init data
        self.fold = str(fold)
        self.init(is_train)
        self.ipts = 0
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_path)

    def __iter__(self):
        return self

    def get_steps(self, is_train):
        if is_train:
            return len(self.trains) // config.batch_size
        else:
            return len(self.tests) // config.batch_size

    def load_data(self):
        with open('../datas/{}/fold/fold{}train.txt'.format(config.dataset, self.fold), 'r') as rf:
            trains = [each.strip() for each in rf.readlines()]
        with open('../datas/{}/fold/fold{}test.txt'.format(config.dataset, self.fold), 'r') as rf:
            tests = [each.strip() for each in rf.readlines()]
        random.shuffle(trains)
        random.shuffle(tests)
        Trains = []
        Tests = []
        for i in range(len(trains)):
            if trains[i] == '':
                print(i)
            train = trains[i].split(':')
            Trains.append(train)
        for i in range(len(tests)):
            if tests[i] == '':
                print(i)
            test = tests[i].split(':')
            Tests.append(test)
        return Trains, Tests


    def init(self, is_train):
        print('init DataIter {}'.format(self.is_train))
        self.trains, self.tests = self.load_data()
        if is_train == 'train':
            self.data_iter = iter(self.trains)
        if is_train == 'test':
            self.data_iter = iter(self.tests)

    def reset(self, is_train):
        if is_train == 'train':
            self.data_iter = iter(self.trains)
        if is_train == 'test':
            self.data_iter = iter(self.tests)

    def get_batch(self):
        batch_data = []
        for i in self.data_iter:
            batch_data.append(i)
            if len(batch_data) == config.batch_size:
                break
        if len(batch_data) < 1:
            return None

        input_ids = []
        attention_masks = []
        lables = []
        s_labels = []
        for data in batch_data:   # ['0', '1', 'barryswallows merkel would never say no']
            # data = data.split('\t')
            lables.append(int(data[1]))  # int labels
            words = self.tokenizer.tokenize(data[2])[:config.max_sen_len-1]
            words = ['[CLS]'] + words
            ids = self.tokenizer.convert_tokens_to_ids(words)
            masks = [1] * len(ids) + [0] * (config.max_sen_len - len(ids))  # attention mask
            ids += [self.tokenizer.pad_token_id] * (config.max_sen_len - len(ids))  # input_ids
            input_ids.append(ids)
            attention_masks.append(masks)
        return_data = {"input_ids": input_ids, "attention_mask": attention_masks,"labels": lables}
        return_data = {k: torch.tensor(v, dtype=torch.long) for k, v in return_data.items()}
        return_data['s_labels'] = torch.FloatTensor(s_labels)
        return return_data

    # next operation for Python iterator, it's necessary for your own iter
    def __next__(self):
        '''
        You must be familiar with python class 'iterator'.
        When the iter has no __next__, that's cursor of the iterator is at the end,
        so there is no addition data for a new __next__, then iter returns None
        So when an iteration terminates, you should re-init the iter.
        But the init() function takes a very long time because of the magnitude of data,
        So I define a new function 'reset()' to get a new iter from self.data_ids
        instead of reading from data.pkl.
        '''
        if self.ipts is None:
            self.reset(self.is_train)
        self.ipts = self.get_batch()  # each iter get a batch
        if self.ipts is None:
            raise StopIteration
        else:
            return self.ipts

if __name__ == "__main__":
    Iter = DataIter(is_train='test', fold=0)
    # print(iter.tokenizer.vocab)
    for i, iter in enumerate(Iter):   # test just
        print(i, iter)