import os
import json
from prettytable import PrettyTable
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.embeddings import DashScopeEmbeddings
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from typing import List,Dict


from config import QWEN_CONFIG
from Container import g_container


def do_ds_embedding(
    docs:List[Document],
    emb_model:DashScopeEmbeddings,
    chunk_size:int=25
) -> Dict[str,any]:
    texts = [doc.page_content for doc in docs]
    meta_datas = [doc.metadata for doc in docs]
    chunk_texts=[texts[i:i+chunk_size] for i in range(0,len(texts),chunk_size)]
    embeddings=[]
    for chunk in chunk_texts:
        emb=emb_model.embed_documents(chunk)
        embeddings=embeddings+emb
    return {
        "texts":texts,
        "embeddings":embeddings,
        "meta_datas":meta_datas
    }

# class DS_Embeddings(Embeddings):
    
#     def __init__(self,model_name :str) -> None:
#         self.model_name = model_name

#     def embed_documents(
#         self, 
#         texts: List[str]
#         ) -> List[List[float]]:
#         return super().embed_documents(texts)
    
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
    
    def do_add_doc(
        self,
        docs:List[Document]
    ):
        data=do_ds_embedding(docs=docs,emb_model=self.embs)
        ids=self.kb.add_embeddings(
            text_embeddings=zip(data['texts'],data['embeddings']),
            metadatas=data['meta_datas'])
        doc_infos=[{"id":id,"meta_data":d.metadata} for id,d in zip(ids,docs)]
        return doc_infos 
    
    def do_search(
        self,
        query: str,
        top_k: int,
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