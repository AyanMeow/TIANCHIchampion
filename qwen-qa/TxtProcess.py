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
from langchain.prompts import PromptTemplate
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
    
    
    def do_save(self,index:str = 'index'):
        self.kb.save_local(folder_path=self.vs_path,index_name=index)
        
    def do_load(self):
        self.kb.load_local(folder_path=self.vs_path)
        
import pdfplumber
from langchain.chains import LLMChain
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
def load_dir_txt_process(txt_dir:str,pdf_dir:str,cfg:QWEN_CONFIG):
    _PROMPT_TEM="""
    你需要在用户给出的内容中找到所有不包含“证券股份”和“证券”的所有公司、机构名称。
    你需要以字符串数组的形式返回符合要求的答案。
    在>>>>和>>>>之间是一个例子：

    >>>>
    
    用户输入：创业板公司具有创新投入大、新旧产业融合成功与否存在不确定性、\n尚处于成长期、经营风险高、业绩不稳定、退市风险高等特点，投资者面临较\n上海真兰仪表科技股份有限公司\n大的市场风险。投资者应充分了解创业板市场的投资风险及本公司所披露的风\n险因素，审慎作出投资决定。\nZenner Metering Technology（Shanghai）Ltd.\n（上海市青浦区盈港东路 6558 号 4 幢）\n首次公开发行股票并在创业板上市\n招股意向书\n保荐机构（主承销商）\n（福州市鼓楼区鼓屏路 27号 1#楼 3层、4 层、5 层）华泰联合证券有限公司
    你的答案：["上海真兰仪表科技股份有限公司"]

    >>>>
    
    你的回答格式应该按照下面的内容，请注意---output等标记都必须输出，这是我用来提取答案的标记。
    不要输出中文的逗号，不要输出引号。如果你找不到符合要求的答案，则返回[]。
    你不可以编造答案，也不可以输出要求以外的文字。
    
    content:${{用户的输入}}
    
    ---output
    ["你的答案1","你的答案2",...]
    
    现在，我们开始作答：
    content:{content}
    """
    PROMPT=PromptTemplate(
        input_variables=['content'],
        template=_PROMPT_TEM
    )
    def find_company(inputs:str):#规则方法筛选
        import re
        inputsl=re.split(r'，|。|；|：|\n',inputs)
        ans=[
            com for com in inputsl if '公司' in com
        ]
        return ans
    llm_chain = LLMChain(llm=g_container.MODEL, prompt=PROMPT)
    file_names = [f for f in os.listdir(txt_dir) if os.path.isfile(os.path.join(txt_dir, f))]
    file_names = tqdm(file_names)
    for file in file_names:
        pdf=file.split('.')[0]+'.PDF'
        with pdfplumber.open(pdf_dir+'/'+pdf) as p:
            first_page=p.pages[0].extract_text()
            companys=find_company(first_page)
            companys=','.join(companys)
        llmout=llm_chain.predict(
            content=companys,
            stop=['---output'],
            callbacks=CallbackManagerForChainRun.get_noop_manager()
            )
        print(llmout)