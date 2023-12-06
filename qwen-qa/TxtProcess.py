import os
import json
from prettytable import PrettyTable
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.document_loaders import TextLoader
from langchain.embeddings.base import Embeddings
from langchain.embeddings.base import Embeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from typing import List,Dict
from tqdm import tqdm_notebook as tqdm
import uuid
from langchain.prompts import PromptTemplate
from config import QWEN_CONFIG
from Container import g_container
from Tools.keywords_extra import LLMKeywordsEXChain

#TODO:单个关键词对应一段文本可能太冗余，应该更适合小切片，
# 例如1-2句话的切片，每个切片对应1-2个关键词，降低冗余存取
def do_ds_keywords_embedding(
    docs:List[Document],
    emb_model:Embeddings,
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
    emb_model:Embeddings,
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
    ) -> List[dict]:
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
        ) -> dict:
        #query_emb=self.embs.embed_query(query)
        result=self.kb.similarity_search_with_relevance_scores(
            query=query,k=top_k,score_threshold=threshold
        )
        documents=[res[0] for res in result]
        scores=[res[1] for res in result]
        return {
            "documents":documents,
            "scores":scores
        }
    
    
    def do_save(self,index:str = 'index'):
        self.kb.save_local(folder_path=self.vs_path,index_name=index)
        
    def do_load(self,index : str = 'index'):
        self.kb=self.kb.load_local(folder_path=self.vs_path,
                           embeddings=self.embs,
                           index_name=index
                           )
    def do_view(self):
        return self.kb.__sizeof__()
        
import pdfplumber
from langchain.chains import LLMChain
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from prettytable import PrettyTable
def load_dir_txt_idx_process(txt_dir:str,pdf_dir:str,cfg:QWEN_CONFIG):
    _PROMPT_TEM="""
    你需要在用户给出的内容中找到所有公司、机构的名称。
    你需要以字符串数组的形式返回符合要求的答案。
    在>>>>和>>>>之间是一些例子：

    >>>>
    
    用户输入：创业板公司具有创新投入大、新旧产业融合成功与否存在不确定性、\n尚处于成长期、经营风险高、业绩不稳定、退市风险高等特点，投资者面临较\n上海真兰仪表科技股份有限公司\n大的市场风险。投资者应充分了解创业板市场的投资风险及本公司所披露的风\n险因素，审慎作出投资决定。\nZenner Metering Technology（Shanghai）Ltd.\n（上海市青浦区盈港东路 6558 号 4 幢）\n首次公开发行股票并在创业板上市\n招股意向书\n保荐机构（主承销商）\n（福州市鼓楼区鼓屏路 27号 1#楼 3层、4 层、5 层）华泰联合证券有限公司
    你的答案：上海真兰仪表科技股份有限公司

    用户输入：'海尔施生物医药股份有限公司\n（宁波市小港街道前进村半港河西 159 号）\n首次公开发行股票招股意向书\n（封卷稿）\n保荐人（主承销商）：瑞信方正证券有限责任公司\n（北京市西城区金融大街甲 9号金融街中心南楼 15层）'
    你的答案：海尔施生物医药股份有限公司
    
    >>>>
    
    注意：
    你的回答格式应该按照下面的内容，请注意---output等标记都必须输出，这是我用来提取答案的标记。
    不要输出中文的逗号，不要输出引号。如果你找不到符合要求的答案，则输出无。
    你输出的答案只能使用简体中文，如果不是简体中文，你需要将其转换为简体中文再输出。
    你不可以编造答案，不可以输出重复答案，也不可以输出要求以外的文字。
    
    content:${{用户的输入}}
    
    ---output
    ${{你的答案1}},${{你的答案2}},...
    
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
    kb_idx=faiss_kb_ds(
        vs_path=cfg.vec_store_path,
        kb_path='./',
        embs=g_container.EMBEDDING
    )
    table=PrettyTable(['source','company','doc_len','chunk_size','meta_data','save_index'])
    for file in tqdm(file_names,desc='handle files:',position=0,leave=False):
        pdf=file.split('.')[0]+'.PDF'
        with pdfplumber.open(pdf_dir+'/'+pdf) as p:
            first_page=p.pages[0].extract_text().replace(' ','')
            if first_page == '':
                first_page=p.pages[1].extract_text().replace(' ','')
            companys=find_company(first_page)
            companys=','.join(companys)
        llmout=llm_chain.predict(
            content=companys,
            stop=['---output'],
            callbacks=CallbackManagerForChainRun.get_noop_manager().get_child()
            )
        llmout=llmout.split('\n')[1]
        llmout=llmout.split(',')

        # print(llmout)
        for company in llmout:
            doc=Document(
                page_content=company,
                metadata={"source":file.split('.')[0]}
                )
            info=kb_idx.do_add_doc([doc])
            table.add_row([file,company,'-','-',str(doc.metadata),'index'])
            # print(info)
        kb=faiss_kb_ds(
            vs_path=cfg.vec_store_path,
            kb_path='./',
            embs=g_container.EMBEDDING
        )
        spliter=RecursiveCharacterTextSplitter(
            chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap
        )
        docs=TextLoader(file_path=txt_dir+'/'+file).load_and_split(text_splitter=spliter)
        first_page=first_page.replace('\n','')
        first_page_doc=Document(
            page_content=first_page,
            metadata=docs[0].metadata
        )
        docs.insert(0,first_page_doc)
        
        #处理表格
        with pdfplumber.open(pdf_dir+'/'+pdf) as p:
            for page in tqdm(p.pages,desc='handle tables:',position=1,leave=False):
                dtables=page.extract_tables()
                if type(dtable) != list : continue
                for dtable in dtables:
                    t=PrettyTable([str(i) for i in range(len(dtable[0]))])
                    t.add_rows(dtable)
                    table_doc=Document(page_content=t.get_formatted_string(),metadata=docs[0].metadata)
                    docs.append(table_doc)
                
        kb.do_add_doc(docs)
        kb.do_save(index=file.split('.')[0])
        table.add_row([file,str(llmout),len(docs),cfg.chunk_size,str(docs[0].metadata),file.split('.')[0]])
        #print(table)
        
    kb_idx.do_save()
    
    print(table)
    
    with open(cfg.vec_store_path+'/description.csv','w',encoding='utf-8') as f:
        f.write(table.get_csv_string())
    
    return kb_idx
        