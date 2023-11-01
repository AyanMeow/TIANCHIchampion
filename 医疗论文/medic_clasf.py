import jieba
import transformers as ts
import pandas as pd
import numpy as np
import torch
import json
from sklearn.model_selection import train_test_split
from ark_nlp.model.ner.global_pointer_bert import (Dataset,Tokenizer,GlobalPointerBertConfig,
                                                   GlobalPointerBert,get_default_model_optimizer,Task,Predictor)
from model import GlobalPointer,GlobalPointerCrossEntropy,GlobalPointerNERPredictor
from torch.utils.data import Dataset,DataLoader
import os
import logging
import torch.nn as nn

class clf(nn.Module):
    """docstring for clf."""
    def __init__(self, bert_path,cat_num):
        super(clf, self).__init__()
        bcfg=ts.BertConfig.from_pretrained(bert_path)
        self.bert=ts.BertModel.from_pretrained(bert_path,config=bcfg)
        self.mlp=nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(768,512,),
            nn.GELU(),
            nn.Linear(512,cat_num),
            nn.GELU(),
            nn.Softmax(dim=1)
        )
        
    def forward(self,input_ids,attention_mask,token_type_ids):
        last_hidden_state=self.bert(input_ids,attention_mask,token_type_ids)
        logits=last_hidden_state[0]
        logits=self.mlp(logits)
        return logits
    

class medData(Dataset):
    def __init__(self,texts,labels,tokenizer,maxlen) -> None:
        super(medData,self).__init__()
        self.text=tokenizer(texts,
                            padding='max_length',
                            max_length = maxlen)
        self.labels=labels
        
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, index):
        text=torch.LongTensor(self.text[index])
        label=torch.LongTensor(self.labels[index])
        return text,label
    
def main():
    pass