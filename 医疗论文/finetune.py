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
    
def main():
    logging.getLogger().setLevel(logging.INFO)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    logging.info('loading tokenizer')
    tokenizer=ts.BertTokenizer.from_pretrained('./pretrained_models')
    
    m_path='/home/xyy/models/macbert-base-chinese-medical-collation'
    #m_path='E:/自然语言处理/model/macbert-base-chinese-medical-collation'
    #m_path='I:/自然语言学习/model/macbert-base-chinese-medical-collation'
    
    logging.info('building model')
    #bcfg=ts.BertConfig.from_pretrained(m_path)
    bertmodel=ts.BertForMaskedLM.from_pretrained(m_path)
    #bertmodel.resize_token_embeddings(len(tokenizer))
    
    logging.info('loading data')
    train_dataset=ts.LineByLineTextDataset(tokenizer=tokenizer,
                                           file_path='./datasets/sentence.txt',
                                           block_size=200)
    eval_dataset=ts.LineByLineTextDataset(tokenizer=tokenizer,
                                           file_path='./datasets/sentence.txt',
                                           block_size=200)
    data_collar=ts.DataCollatorForWholeWordMask(tokenizer=tokenizer,
                                                mlm=True,
                                                mlm_probability=0.15)
    
    
    logging.info('initing training')
    training_args = ts.TrainingArguments(
        output_dir='./pretrained_models',
        overwrite_output_dir=True,
        num_train_epochs=50,
        per_device_train_batch_size=16,
        save_steps=1000,
        save_total_limit=2,
        prediction_loss_only=True,
    )
    
    trainer = ts.Trainer(
        model=bertmodel,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collar,
    )
    
    logging.info('start training')
    trainer.train()
    
    
    logging.info('saving pretrained bert')
    trainer.save_model('./pretrained_models')
    eval_results = trainer.evaluate()
    logging.info(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
    
    logging.info('-------complete-------')
       
if __name__ == "__main__":
    main()