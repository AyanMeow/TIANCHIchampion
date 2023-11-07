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

class Fmodel(nn.Module):
    """docstring for Fmodel."""
    def __init__(self, bert_path):
        super(Fmodel, self).__init__()
        bcfg=ts.BertConfig.from_pretrained(bert_path)
        self.bert=ts.BertModel.from_pretrained(bert_path,config=bcfg)
        self.vocab_size=bcfg.vocab_size
        self.hidden_size=bcfg.hidden_size
        self.biffine=nn.Sequential(nn.Linear(self.hidden_size,self.vocab_size),
                                   nn.GELU())

    def forward(self,input_ids,attention_mask,token_type_ids):
        output=self.bert(input_ids,attention_mask,token_type_ids)
        logits=output['last_hidden_state']
        logits=self.biffine(logits)
        return logits
    
class Fset(Dataset):
    def __init__(self,data,tokenizer,device,maxlen=200):
        super(Fset,self).__init__()
        text=[d['text'] for d in data]
        mtext=[d['masked'] for d in data]
        self.inputs=[tokenizer(t,
                            padding='max_length',
                            max_length = maxlen) for t in mtext]
        self.labels=[tokenizer(t,
                            padding='max_length',
                            max_length = maxlen) for t in text]
        self.device=device
        
    def __getitem__(self, idx):
        input_ids=torch.LongTensor(self.inputs[idx]['input_ids']).to(self.device)
        attention_mask=torch.LongTensor(self.inputs[idx]['attention_mask']).to(self.device)
        token_type_ids=torch.LongTensor(self.inputs[idx]['token_type_ids']).to(self.device)
        label_ids=torch.LongTensor(self.labels[idx]['input_ids']).to(self.device)
        
        return {'input_ids':input_ids,'attention_mask':attention_mask,'token_type_ids':token_type_ids},label_ids
    
    def __len__(self,):
        return len(self.inputs)
    
def main():
    logging.getLogger().setLevel(logging.INFO)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    logging.info('loading data')
    with open('./datasets/fintune.json','r',encoding='utf-8') as f:
        f_data=json.load(f)
        f.close()
    m_path='/home/xyy/models/macbert-base-chinese-medical-collation'
    #m_path='E:/自然语言处理/model/macbert-base-chinese-medical-collation'
    #m_path='I:/自然语言学习/model/macbert-base-chinese-medical-collation'
    logging.info('loading tokenizer')
    tokenizer=ts.BertTokenizer.from_pretrained(m_path)
    
    logging.info('building dataset')
    fdataset=Fset(data=f_data,
                  tokenizer=tokenizer,
                  device=device,
                  maxlen=200)
    fdl=DataLoader(dataset=fdataset,batch_size=32)
    
    logging.info('building model')
    bertmodel=Fmodel(m_path).to(device)
    
    logging.info('initing training')
    opt=torch.optim.Adam(bertmodel.parameters(),lr=5e-5,weight_decay=0.99)
    loss_fn=nn.CrossEntropyLoss()
    epoch=50
    vocab_size=bertmodel.vocab_size
    
    logging.info('start training')
    bertmodel.train()
    for e in range(0,epoch):
        for step,(bd,l) in enumerate(fdl):
            logits=bertmodel(**bd)
            loss=loss_fn(logits.view(-1,vocab_size),l.view(-1))
            
            opt.zero_grad()
            loss.backward()
            opt.step()
    
        if step%50 == 0:
            logging.info('epoch:'+str(e)+'  loss:'+str(loss.item()))
    
    
    logging.info('saving pretrained bert')
    torch.save(bertmodel.state_dict(),'./finetune_bert.pth')
    
    logging.info('-------complete-------')
       
if __name__ == "__main__":
    main()