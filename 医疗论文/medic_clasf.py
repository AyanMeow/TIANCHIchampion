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

label_list=['C-对比选项',
 'C-研究方法',
 'C-研究目的',
 'I-其他干预',
 'I-干预',
 'I-教育/行为干预',
 'I-药物干预',
 'I-非药物干预',
 'O-定性结论',
 'O-定量结论',
 'O-结论',
 'O-结论主语',
 'P-人群/患者类型',
 'P-条件',
 'P-研究对象',
 'P-评估项',
 'S-因素(病因/风险)分析',
 'S-指南标准建议',
 'S-治疗',
 'S-病因学',
 'S-统计分析',
 'S-诊断',
 'S-预后']

class clf(nn.Module):
    """docstring for clf."""
    def __init__(self, bert_path,cat_num):
        super(clf, self).__init__()
        bcfg=ts.BertConfig.from_pretrained(bert_path)
        self.bert=ts.BertModel.from_pretrained(bert_path,config=bcfg)
        self.lstm=nn.LSTM(self.bert.config.hidden_size,self.bert.config.hidden_size//2,num_layers=1,
                                batch_first=True,bidirectional=True)
        self.dropout=nn.Dropout(p=0.5)
        self.mlp=nn.Sequential(
            nn.Linear(self.bert.config.hidden_size,cat_num),
            nn.GELU()
        )
        
    def forward(self,input_ids,attention_mask,token_type_ids):
        last_hidden_state=self.bert(input_ids,attention_mask,token_type_ids)
        logits=last_hidden_state[0][:,0,:]
        logits=self.dropout(logits)
        logits,(cell,_)=self.lstm(logits)
        logits=self.mlp(logits)
        return logits
    

class medData(Dataset):
    def __init__(self,texts,labels,tokenizer,maxlen,label_list,device) -> None:
        super(medData,self).__init__()
        self.text=[tokenizer(t,
                            padding='max_length',
                            max_length = maxlen) for t in texts]
        self.labels=[label_list.index(l) for l in labels]
        self.labels=nn.functional.one_hot(torch.tensor(self.labels))
        self.device=device
        
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, index):
        input_ids=torch.LongTensor(self.text[index]['input_ids']).to(self.device)
        attention_mask=torch.LongTensor(self.text[index]['attention_mask']).to(self.device)
        token_type_ids=torch.LongTensor(self.text[index]['token_type_ids']).to(self.device)
        label=torch.tensor(self.labels[index],dtype=torch.float).to(self.device)
        
        return {'input_ids':input_ids,'attention_mask':attention_mask,'token_type_ids':token_type_ids},label
    
def main():
    #os.environ["CUDA_VISIBLE_DEVICES"]='0,1,2'
    logging.getLogger().setLevel(logging.INFO)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info('load tokenizer')
    m_path='/home/xyy/models/macbert-base-chinese-medical-collation'
    #m_path='E:/自然语言处理/model/macbert-base-chinese-medical-collation'
    tokenizer=ts.BertTokenizer.from_pretrained(m_path)
    
    logging.info('load data')
    ctrain_df=pd.read_json('./datasets/context_title_train.json')
    ctrain_dataset=medData(ctrain_df['text'].__array__(),ctrain_df['label'].__array__(),tokenizer,200,label_list,device)
    ctrain_dl=DataLoader(ctrain_dataset,batch_size=32)
    
    logging.info('load bert model')
    classfier=clf(m_path,cat_num=len(label_list)).to(device)
    #bcfg=ts.BertConfig.from_pretrained(m_path,num_labels=len(label_list))
    #classfier=ts.BertForSequenceClassification.from_pretrained(m_path,config=bcfg,ignore_mismatched_sizes=True).to(device)
    
    logging.info('init training')
    epoch=500
    opt=torch.optim.Adam(classfier.parameters(),lr=1e-5)
    loss_fn=torch.nn.CrossEntropyLoss()
    best_score=1
    logging.info('start training')
    classfier.train()
    for e in range(0,epoch):
        train_loss=0
        for step,(bd,l) in enumerate(ctrain_dl):
            logits=classfier(bd['input_ids'],
                             bd['attention_mask'],
                             bd['token_type_ids'])
            loss=loss_fn(logits,l)
            train_loss+=loss.item()
            # output=classfier(**bd,labels=l)
            # loss=output['loss']
            # train_loss+=loss.item()
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
        train_loss=train_loss/step
        logging.info('epoch:'+str(e)+'  loss:'+str(train_loss))
        
        if train_loss < best_score:
            best_score=train_loss
            torch.save(classfier.state_dict(),'./best_classfier.pth')
            logging.info('-----------best model saved')


if __name__ == "__main__":
    main()