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
from tricks import EMA,PGD

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
    
    logging.info('load tokenizer')
    
    tokenizer=ts.BertTokenizer.from_pretrained(model_rope)
    ttokenier=Tokenizer(vocab=tokenizer,max_seq_len=200)
    
    trainds.convert_to_ids(ttokenier)
    valds.convert_to_ids(ttokenier)
    
    logging.info('load model')
    args={
        'bert_dir':model_rope,
        'last_4_bert':True,
        'use_bilstm':True
    }
    bertModel=GlobalPointer(args,len(label_list),64).to(device)

    traindl=DataLoader(trainds.dataset,batch_size=64)
    valdl=DataLoader(valds.dataset,batch_size=64)
     
    opt=torch.optim.Adam(bertModel.parameters(),lr=4e-5)
    loss_fn=GlobalPointerCrossEntropy()
    
    pgd=PGD(model=bertModel)
    K=3
    
    ema=EMA(model=bertModel,decay=0.999)
    ema.register()
    
    logging.info('strat_training')
    epoch=300   
    best_score=1
    for e in range(0,epoch):
        train_loss=0
        for step ,bd in enumerate(traindl):
            logits=bertModel(bd['input_ids'].to(device),bd['attention_mask'].to(device),bd['token_type_ids'].to(device))
            loss=loss_fn(logits,bd['label_ids'].to(device).to_dense())
            train_loss+=loss.item()
            
            opt.zero_grad()
            loss.backward()
            
            
            token_id=bd['input_ids']
            at_mask=bd['attention_mask']
            token_type_ids=bd['token_type_ids']
            label_id=bd['label_ids']
            pgd.backup_grad()
            for t in range(K):
                pgd.attack(is_first_attack=(t == 0))
                if t != K - 1:
                    bertModel.zero_grad()
                else:
                    pgd.restore_grad()
                    
                outputs = bertModel(token_id.to(device), at_mask.to(device), token_type_ids.to(device))
                loss_pgd = loss_fn(outputs, label_id.to(device).to_dense()).mean()
                loss_pgd.backward()
            pgd.restore()
            
            opt.step()
            
            ema.update()
        
        tloss=train_loss/step
        logging.info('epoch:'+str(e)+'  loss:'+str(tloss))
        
        with torch.no_grad():
            vloss=0
            for vstep,bd in enumerate(valdl):
                vlogits=bertModel(bd['input_ids'].to(device),bd['attention_mask'].to(device),bd['token_type_ids'].to(device))
                vloss+=loss_fn(vlogits,bd['label_ids'].to(device).to_dense()).item()
            vloss=vloss/vstep
        logging.info('epoch:'+str(e)+'  vloss:'+str(vloss))
        if vloss < best_score:
            best_score=vloss
            torch.save(bertModel.state_dict(),'./title_best(lstm).pth')
            logging.info('---------save best model')
    
    logging.info('try_preding')
    
    test_text='常染色体显性多囊肾病的临床问题及其肾脏替代治疗的选择'
    
    mpred=GlobalPointerNERPredictor(bertModel,ttokenier,trainds.cat2id,tokenizer)
    
    result=mpred.predict_one_sample(test_text)
    
    logging.info('----------pred:'+str(result))
    
    
if __name__ == "__main__":
    main()