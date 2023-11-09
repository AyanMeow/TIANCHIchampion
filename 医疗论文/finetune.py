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
import math
from tqdm import tqdm
    
class wwm_pretrain_set(Dataset):
    """docstring for wwm_pretrain_set."""
    def __init__(self, file_p,tokenizer,vocab,dynamic=5,maxlen=200):
        super(wwm_pretrain_set, self).__init__()
        self.sentences=[]
        jieba.load_userdict(vocab)
        with open(file_p,'r',encoding='utf-8') as f:
            totaldata=f.readlines()
            with tqdm(totaldata * dynamic) as loader:
                for data in loader:
                    data = data.replace('\n', '').replace('\r', '').replace('\t','').replace(' ','').replace('　', '')
                    chinese_ref=self.get_chinese_ref(data)
                    input_ids=tokenizer.encode_plus(data,truncation=True,max_length=maxlen).input_ids
                    self.sentences.append({
                        'input_ids':input_ids,
                        'chinese_ref':chinese_ref
                    })

    def get_chinese_ref(self,sentence):
        s_sprit=[w for w in jieba.cut(sentence)]
        chinese_ref=[]
        index=1
        for word in s_sprit:
            for i,char in enumerate(word):
                if i>0:
                    chinese_ref.append(index)
                index+=1
        return chinese_ref
    
    def __getitem__(self, index):
        return self.sentences[index]
    
    def __len__(self,):
        return len(self.sentences)
    

def main():
    logging.getLogger().setLevel(logging.INFO)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    logging.info('loading tokenizer')
    tokenizer=ts.BertTokenizer.from_pretrained('./pretrained_models')
    
    vocab_path='./datasets/med_word.txt'
    m_path='/home/xyy/models/macbert-base-chinese-medical-collation'
    #m_path='E:/自然语言处理/model/macbert-base-chinese-medical-collation'
    #m_path='I:/自然语言学习/model/macbert-base-chinese-medical-collation'
    
    logging.info('building model')
    #bcfg=ts.BertConfig.from_pretrained(m_path)
    bertmodel=ts.BertForMaskedLM.from_pretrained(m_path)
    #bertmodel.resize_token_embeddings(len(tokenizer))
    
    logging.info('loading data')
    train_dataset=wwm_pretrain_set(file_p='./datasets/sentence.txt',
                                   tokenizer=tokenizer,
                                   vocab=vocab_path)
    
    data_collar=ts.DataCollatorForWholeWordMask(tokenizer=tokenizer,
                                                mlm=True,
                                                mlm_probability=0.15)
    
    
    logging.info('initing training')
    save_step=len(train_dataset)*10
    training_args = ts.TrainingArguments(
        output_dir='./pretrained_models',
        overwrite_output_dir=True,
        num_train_epochs=100,
        per_device_train_batch_size=16,
        save_steps=save_step,
        save_total_limit=2,
        prediction_loss_only=True,
    )
    
    trainer = ts.Trainer(
        model=bertmodel,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collar,
    )
    
    logging.info('start training')
    trainer.train()
    
    
    logging.info('saving pretrained bert')
    trainer.save_model('./pretrained_models')
    
    logging.info('-------complete-------')
       
if __name__ == "__main__":
    main()