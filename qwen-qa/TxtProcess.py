import os
from prettytable import PrettyTable
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS

from config import QWEN_CONFIG


def load_dir_and_split(file_dir_path=None,config:QWEN_CONFIG = None):
    '''
    处理txt文件并向量化,持久化到本地存储
    '''
    assert file_dir_path != None,'empty file dir'
    
    if config.online_emb[0]:
        embeddings=HuggingFaceEmbeddings(model_name=config.online_emb[1])
    elif config.local_emb[0]:
        embeddings=HuggingFaceEmbeddings(model_name=config.local_emb[1])
    else:
        return False
    
    filenames=os.listdir(file_dir_path)
    textsplitter=RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
    
    if not os.path.exists(config.vec_store_path):
        os.mkdir(config.vec_store_path)
    
    pt=PrettyTable(['file_name','splited_len','vec_idx'])
    doc_split=[]
    
    for file in filenames:
        loader=TextLoader(file_path=file_dir_path+'/'+file,
                          encoding='utf-8')
        doc=loader.load()
        doc_split+=textsplitter.split_documents(doc)
        
    vec_store=FAISS.from_documents(documents=doc_split,
                                    embedding=embeddings,
                                    )
    index_name='financial'
    vec_store.save_local(folder_path=config.vec_store_path,
                            index_name=index_name)
    
    pt.add_row([file,str(len(doc_split)),index_name])
    
    print('vec_store_path:'+config.vec_store_path)
    print(pt)
    
    return 'done'

def load_vec_and_similarty_search(query = None,vec_idx = 'financial',config:QWEN_CONFIG = None):
    '''
    加载向量库,并进行相似度搜索,返回topK个结果。
    '''
    assert query != None,'empty vector index'
    
    if config.online_emb[0]:
        embeddings=HuggingFaceEmbeddings(model_name=config.online_emb[1])
    elif config.local_emb[0]:
        embeddings=HuggingFaceEmbeddings(model_name=config.local_emb[1])
    else:
        return False
    
    vec_store=FAISS.load_local(folder_path=config.vec_store_path,
                               embeddings=embeddings,
                               index_name=vec_idx)
    
    searchs=vec_store.similarity_search(query=query,
                                        k=config.vec_search_topK)
    
    return searchs