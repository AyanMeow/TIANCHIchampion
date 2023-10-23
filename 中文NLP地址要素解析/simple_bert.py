import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import transformers as ts
from torch.utils.data import Dataset,DataLoader
import logging
import numpy as np
from torchcrf import CRF
import os

class RoBERTa_CRF(nn.Module):
    def __init__(self, bert_path,num_labels) -> None:
        super(RoBERTa_CRF,self).__init__()
        self.bert=ts.AutoModelForTokenClassification.from_pretrained(bert_path,num_labels=num_labels)
        self.crf=CRF(num_tags=num_labels,batch_first=True)
        
    def forward(self,batch_data):
        output=self.bert(**batch_data)
        loss=-self.crf(output.logits,batch_data['labels'],batch_data['attention_mask'].bool())
        out=self.crf.decode(output.logits,batch_data['attention_mask'].bool())
        return out,loss

def load_data(path,train=True):
    if train:
        name='train.conll'
    else:
        name='dev.conll'
    texts=[]
    labels=[]
    with open(path+'/'+name) as file:
        for line in file:
            if line == '' or line == '\n':
                if texts:
                    yield{
                        'text':texts,
                        'label':labels
                    }
                    texts=[]
                    labels=[]
            else:
                sprilts=line.split()
                texts.append(sprilts[0])
                labels.append(sprilts[1])
        if texts:
            yield{
                'text':texts,
                'label':labels
            }
    file.close()

def get_entities(text,label):
    entities=[]
    cur_entities={}
    for t,l in zip(text,label):
        if l[0] in 'BOS' and cur_entities:
            entities.append(cur_entities)
            cur_entities={}
        if l[0] in 'BS':
            cur_entities={
                'text':t,
                'entities':[l[2:]]
            }
        elif l[0] in 'IE':
            cur_entities['text']+=t
            cur_entities['entities'].append(l[2:])
    if cur_entities:
        entities.append(cur_entities)
    return entities

def makedata(data):
    sentences=[]
    labels=[]
    for _,d in enumerate(data):
        entities=get_entities(d['text'],d['label'])
        sentence=''
        label=[]
        for e in entities :
            sentence+=e['text']
            label.extend(e['entities'])
        sentences.append([sentence])
        labels.append(label)
    return {'text':sentences,'label':labels}

def label2int(labels):
    ldict=['O']
    for l in labels:
        for i in l:
            if i not in ldict:
                ldict.append(i)
    return ldict

class Addr(Dataset):
    """docstring for Addr."""
    def __init__(self,data,tokenizer,ldict,device):
        self.text=[]
        for t in data['text']:
            self.text.append(list(t[0]))
        self.encodings=tokenizer(self.text,is_split_into_words=True,padding=True)
        labels=[]
        for i,l in enumerate(data['label']):
            label=[0,]
            for t in l:
                label.append(ldict.index(t))
            for _ in range(0,len(self.encodings['input_ids'][i])-len(label)):
                label.append(0)
            labels.append(label)
        self.labels=labels
        self.device=device
        
    def __getitem__(self, idx):
        input_ids = torch.LongTensor(self.encodings['input_ids'][idx]).to(self.device)
        attention_mask = torch.LongTensor(self.encodings['attention_mask'][idx]).to(self.device)
        labels = torch.LongTensor(self.labels[idx]).to(self.device)
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

    def __len__(self):
        return len(self.text)

def load_test_data(path):
    texts=[]
    with open(path+'/final_test.txt') as file:
        for line in file:
            splits=line.split('\x01')
            texts.append(list(splits[1].rsplit()[0]))
        file.close()
    return texts

class AddrTest(Dataset):
    def __init__(self,data,tokenizer,device):
        self.text=data
        self.encodings=tokenizer(data,is_split_into_words=True,padding=True)
        self.device=device
    def __len__(self):
        return len(self.text)
    def __getitem__(self, idx):
        input_ids = torch.LongTensor(self.encodings['input_ids'][idx]).to(self.device)
        attention_mask = torch.LongTensor(self.encodings['attention_mask'][idx]).to(self.device)
        return {'input_ids': input_ids, 'attention_mask': attention_mask}
    

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]='1'
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)
    logging.getLogger().setLevel(logging.INFO)
    
    base_root='/home/xyy'
    models_path='/models'
    m_bert='/bert-base-chinese'
    m_robert='/chinese-roberta-wwm-ext'
    
    m_use=base_root+models_path+m_robert
    
    logger.info('start load data')
    
    logging.info('start load data')
    train_data=load_data('./datasets/')
    train_data=makedata(train_data)
    val_data=load_data('./datasets/',False)
    val_data=makedata(val_data)
    test_data=load_test_data('./datasets/')
    logger.info('data load complete')
    
    ldict=label2int(train_data['label']+val_data['label'])
    
    logger.info('initiate tokenlizer')
    tokenizer=ts.AutoTokenizer.from_pretrained(m_use)
    
    logger.info('create datasets')
    train_datasets=Addr(train_data,tokenizer,ldict,device) 
    val_datasets=Addr(val_data,tokenizer,ldict,device)
    test_datasets=AddrTest(test_data,tokenizer,device)

    logger.info('load bert model')
    classfier=RoBERTa_CRF(m_use,len(ldict))
    logger.info('use device ',device)
    classfier=classfier.to(device)
    
    epoch=500;
    bs=128
    
    train_dl=DataLoader(train_datasets,batch_size=bs)
    val_dl=DataLoader(val_datasets,batch_size=bs)
    
    logger.info('-----------start training---------')
    params=classfier.parameters()
    optimizer=torch.optim.Adam(params=params,lr=5e-5)
    classfier.train()
    for e in range(0,epoch):
        for step,batch in enumerate(train_dl):
            out,loss=classfier(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        logger.info('epoch'+str(e)+'  loss:'+str(loss.item()))
        
        
    logger.info('-----------start evaluting---------')
    accuracy=[]
    for step ,batch in enumerate(val_dl):
        with torch.no_grad():
            out,loss=classfier(batch)
        labels=batch['labels'].tolist()
        acc=[]
        for logits,label in zip(out,labels):
            lo=np.array(logits)
            la=np.array(label)
            lo=np.pad(lo,(0,la.shape[0]-lo.shape[0]))
            acc.append((lo==la).sum()/la.shape[0])
        accuracy.append(np.mean(acc))
    accuracy=np.mean(accuracy)
    
    logger.info('accuracy:'+str(accuracy)+'   loss:'+str(loss.item()))
    
    torch.save(classfier.state_dict(),'./roberta_crf.pth')
    logger.info('-----------model state saved---------')