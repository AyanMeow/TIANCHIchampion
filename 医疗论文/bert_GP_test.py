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
from torch.utils.data import DataLoader
import os
import logging
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


def main():
    #os.environ["CUDA_VISIBLE_DEVICES"]='0'
    logging.getLogger().setLevel(logging.INFO)
    
    device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    print('load data')
    logging.info('load data')
    
    train_data=pd.read_json('./datasets/title_train.json')
    train_set,val_set=train_test_split(train_data,test_size=0.2,random_state=1234)
    train_set=train_set.reset_index(drop=True)
    val_set=val_set.reset_index(drop=True)
    train_set['label']=train_set['label'].apply(lambda x: str(x))
    val_set['label']=val_set['label'].apply(lambda x: str(x))
    
    trainds=Dataset(train_set,categories=label_list)
    valds=Dataset(val_set,categories=label_list)
    
    model_rope='/home/xyy/models/macbert-base-chinese-medical-collation'
    
    print('load tokenizer')
    logging.info('load tokenizer')
    
    tokenizer=ts.BertTokenizer.from_pretrained(model_rope)
    ttokenier=Tokenizer(vocab=tokenizer,max_seq_len=200)
    
    trainds.convert_to_ids(ttokenier)
    valds.convert_to_ids(ttokenier)
    
    print('load model')
    logging.info('load model')
    args={
        'bert_dir':model_rope,
        'last_4_bert':True,
        'use_bilstm':True
    }
    bertModel=GlobalPointer(args,len(label_list),64).to(device)

    traindl=DataLoader(trainds.dataset,batch_size=64)
     
    opt=torch.optim.Adam(bertModel.parameters(),lr=5e-5)
    loss_fn=GlobalPointerCrossEntropy()
    
    print('strat_training')
    logging.info('strat_training')
    epoch=100   
    best_score=1
    for e in range(0,epoch):
        train_loss=0
        for step ,bd in enumerate(traindl):
            logits=bertModel(bd['input_ids'].to(device),bd['attention_mask'].to(device),bd['token_type_ids'].to(device))
            loss=loss_fn(logits,bd['label_ids'].to(device).to_dense())
            train_loss+=loss.item()
            
            opt.zero_grad()
            loss.backward()
            opt.step()
        
        tloss=train_loss/step
        print('epoch:',e,'  loss:',tloss)
        logging.info('epoch:'+str(e)+'  loss:'+str(tloss))
        
        if tloss < best_score:
            best_score=tloss
            torch.save(bertModel.state_dict(),'./title_best(lstm).pth')
            print('---------save best model')
            logging.info('---------save best model')
    
    print('try_preding')
    logging.info('try_preding')
    
    test_text='常染色体显性多囊肾病的临床问题及其肾脏替代治疗的选择'
    
    mpred=GlobalPointerNERPredictor(bertModel,ttokenier,trainds.cat2id,tokenizer)
    
    result=mpred.predict_one_sample(test_text)
    
    print('pred:',result)
    logging.info('----------pred:'+str(result))
    
    
if __name__ == "__main__":
    main()