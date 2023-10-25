import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import transformers as ts
from torch.utils.data import Dataset,DataLoader
import numpy as np
from simple_bert import RoBERTa_CRF
import os

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
        labels=torch.LongTensor(np.zeros_like(self.encodings['attention_mask'][idx])).to(self.device)
        return {'input_ids': input_ids, 'attention_mask': attention_mask,'labels':labels}

def create_line(idx,data,preds):
    line=str(idx)+'\u0001'+''.join(data)+'\u0001 '
    def split_same(preds):
        result = []
        current_group = [preds[0]]
        for i in range(1, len(preds)):
            if preds[i] == preds[i - 1]:
                current_group.append(preds[i])
            else:
                result.append(current_group)
                current_group = [preds[i]]
        result.append(current_group)
        return result
    rpreds=split_same(preds)
    rpreds=rpreds[1:-1]
    bio=''
    for s in rpreds:
        if s[0] == 0:
            bio=bio+(ldict[s[0]]+' ')*(len(s))
        elif len(s) == 1:
            bio=bio+'B-'+ldict[s[0]]+' '
        else:
            lens=len(s)-2
            bio=bio+'B-'+ldict[s[0]]+' '+('I-'+ldict[s[0]]+' ')*lens+'E-'+ldict[s[0]]+' '
    bio=bio.rstrip()
    return line+bio

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]='1'
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    ldict=['O',
    'prov',
    'city',
    'district',
    'town',
    'community',
    'poi',
    'road',
    'roadno',
    'subpoi',
    'devzone',
    'houseno',
    'intersection',
    'assist',
    'cellno',
    'floorno',
    'distance',
    'village_group']
    
    base_root='/home/xyy'
    models_path='/models'
    m_bert='/bert-base-chinese'
    m_robert='/chinese-roberta-wwm-ext'
    
    m_use=base_root+models_path+m_robert
    
    test_data=load_test_data('./datasets/')
    device=torch.device('cuda')
    tokenizer=ts.AutoTokenizer.from_pretrained(m_use)
    test_datasets=AddrTest(test_data,tokenizer,device)
    test_dl=DataLoader(test_datasets,batch_size=1)
    
    
    model=RoBERTa_CRF(bert_path=m_use,num_labels=len(ldict))
    state_dict=torch.load('./roberta_crf.pth')
    model.load_state_dict(state_dict=state_dict)
    model=model.to(device)
    
    with open('./Akiyan_addr_parsing_runid.txt',"w",encoding="utf-8") as file:
        for step ,batch in enumerate(test_dl):
            with torch.no_grad():
                out,loss=model(batch)
            indexs = batch['input_ids'].tolist()[0].index(102)
            sentence = tokenizer.decode(batch['input_ids'][0][1:indexs]).replace(" ","")
            line=create_line(step+1,test_data[0],out[0])
            print(line)
            file.write(line+'\n')
        file.close()