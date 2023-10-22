import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import transformers as ts
from torch.utils.data import Dataset,DataLoader
import logging

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
    def __init__(self,data,tokenizer,ldict):
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
        
    def __getitem__(self, idx):
        input_ids = torch.LongTensor(self.encodings['input_ids'][idx])
        attention_mask = torch.LongTensor(self.encodings['attention_mask'][idx])
        labels = torch.LongTensor(self.labels[idx])
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
    def __init__(self,data,tokenizer):
        self.text=data
        self.encodings=tokenizer(data,is_split_into_words=True,padding=True)
    def __len__(self):
        return len(self.text)
    def __getitem__(self, idx):
        input_ids = torch.LongTensor(self.encodings['input_ids'][idx])
        attention_mask = torch.LongTensor(self.encodings['attention_mask'][idx])
        return {'input_ids': input_ids, 'attention_mask': attention_mask}
    

if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)
    logging.getLogger().setLevel(logging.INFO)
    
    base_root='/home/xyy'
    models_path='/models'
    m_bert='/bert-base-chinese'
    m_robert='/chinese-roberta-wwm-ext'
    
    m_use=base_root+models_path+m_bert
    
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
    train_datasets=Addr(train_data,tokenizer,ldict) 
    val_datasets=Addr(val_data,tokenizer,ldict)
    test_datasets=AddrTest(test_data,tokenizer)

    logger.info('load bert model')
    classfier=ts.AutoModelForTokenClassification.from_pretrained(m_use,num_labels=len(ldict))
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logger.info('use device ',device)
    classfier=classfier.to(device)
    
    
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        acc = (labels == preds).sum()/len(labels)
        return {
            'accuracy': acc,
        }
        
    training_args = ts.TrainingArguments(
        output_dir='./results',         # output directory 结果输出地址
        num_train_epochs=1,             # total # of training epochs 训练总批次
        per_device_train_batch_size=64,  # batch size per device during training 训练批大小
        per_device_eval_batch_size=64,   # batch size for evaluation 评估批大小
        logging_dir='./logs/',    # directory for storing logs 日志存储位置
        learning_rate=1e-4,             # 学习率
        save_steps=500,               # 不保存检查点
    )
    training_args.device
    trainer=ts.Trainer(model=classfier,
                    args=training_args,
                    train_dataset=train_datasets,
                    eval_dataset=val_datasets,
                    compute_metrics=compute_metrics,
                    tokenizer=None)
    logger.info('start training')
    trainer.train()
    logger.info('start evaluating')
    trainer.evaluate()
    trainer.save_model()
    logger.info('trained model saved')
    #preds=trainer.predict(test_dataset=test_datasets)