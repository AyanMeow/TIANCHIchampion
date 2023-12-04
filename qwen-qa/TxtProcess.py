import os
import json
from prettytable import PrettyTable
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.document_loaders import TextLoader
from langchain.embeddings import DashScopeEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from typing import List,Dict
from tqdm import tqdm
import uuid

from config import QWEN_CONFIG
from Container import g_container
from Tools.keywords_extra import LLMKeywordsEXChain

#TODO:单个关键词对应一段文本可能太冗余，应该更适合小切片，
# 例如1-2句话的切片，每个切片对应1-2个关键词，降低冗余存取
def do_ds_keywords_embedding(
    docs:List[Document],
    emb_model:DashScopeEmbeddings,
    chunk_size:int=25,
    kw_ex:LLMKeywordsEXChain = None,
) -> Dict[str,any]:
    texts = [doc.page_content.replace('\n','') for doc in docs]
    meta_datas = [doc.metadata for doc in docs]
    
    texts_t = tqdm(texts)
    extrations=[]
    for text in texts_t:
        extration=kw_ex._call(inputs={'inputs':text})
        extration=','.join(extration)
        extrations.append(extration)
    
    chunk_texts=[extrations[i:i+chunk_size] for i in range(0,len(extrations),chunk_size)]
    embeddings=[]
    chunk_tqdm = tqdm(chunk_texts)
    for chunk in chunk_tqdm:
        emb=emb_model.embed_documents(chunk)
        embeddings=embeddings+emb
    return {
        "texts":texts,
        "embeddings":embeddings,
        "meta_datas":meta_datas
    }


def do_ds_embedding(
    docs:List[Document],
    emb_model:DashScopeEmbeddings,
    chunk_size:int=25,
    **kwargs
) -> Dict[str,any]:
    texts = [doc.page_content.replace('\n','') for doc in docs]
    meta_datas = [doc.metadata for doc in docs] 
    embeddings = emb_model.embed_documents(texts)
    return {
        "texts":texts,
        "embeddings":embeddings,
        "meta_datas":meta_datas
    }
   

    
class faiss_kb_ds(object):
    def __init__(
        self,
        vs_path:str = None,
        kb_path:str = None,
        embs:Embeddings = None
        ) -> None:
        self.vs_path = vs_path
        self.kb_path = kb_path
        self.embs = embs
        init=Document(page_content='init',metadata={})
        self.kb=FAISS.from_documents(
            documents=[init],embedding=self.embs,normalize_L2=True
        )
        ids=list(self.kb.docstore._dict.keys())
        self.kb.delete(ids)
        self.keywords_ex=LLMKeywordsEXChain.from_llm(llm=g_container.MODEL)
    
    def do_add_doc(
        self,
        docs:List[Document]
    ):
        data=do_ds_embedding(docs=docs,emb_model=self.embs,kw_ex=self.keywords_ex)
        ids=self.kb.add_embeddings(
            text_embeddings=zip(data['texts'],data['embeddings']),
            metadatas=data['meta_datas'])
        doc_infos=[{"id":id,"meta_data":d.metadata} for id,d in zip(ids,docs)]
        return doc_infos 
    
    def do_search(
        self,
        query: str,
        top_k: int = 5,
        threshold: int = 0.5
        ) -> List[Document]:
        query_emb=self.embs.embed_query(query)
        result=self.kb.similarity_search_with_score_by_vector(
            embedding=query_emb,k=top_k,score_threshold=threshold
        )
        return result
    
    
    def do_save(self):
        self.kb.save_local(folder_path=self.vs_path)
        
    def do_load(self):
        self.kb.load_local(folder_path=self.vs_path)