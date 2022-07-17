import torch.nn as nn
from transformers import BertModel, BertConfig, BertTokenizer, DistilBertTokenizer, DistilBertModel
import torch
from os.path import join, exists
import torch.functional as F
import config
import os
import json

class HBert(nn.Module):
    def __init__(self):
        super(HBert, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for name, param in self.bert.named_parameters():
            if 'layer.11' in name or 'layer.10' in name or 'layer.9' in name or 'layer.8' in name or 'pooler.dense' in name:
                # if 'layer.11' in name or 'pooler.dense' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.tokenizer = BertTokenizer.from_pretrained(config.bert_path)
        self.dropout = nn.Dropout(0.1)
        self.clf = torch.nn.Linear(768,out_features=config.nclass)

    def forward(self, input_ids, attention_mask, *args, **kwargs):
        token_embeddings,cls = self.bert(input_ids=input_ids, attention_mask=attention_mask)[:2] # DistilBertTokenizer只返回一个值
        token_embeddings = self.dropout(token_embeddings)
        # Read ou
        self.encoded = torch.max(token_embeddings, dim=1)[0]
        out = self.clf(self.encoded)
        return out

    def save(self, save_dir: str):
        self.bert.save_pretrained(save_dir)
        # self.tokenizer.save_pretrained(save_dir)