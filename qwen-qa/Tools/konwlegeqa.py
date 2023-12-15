import json
from typing import Any, Type,List,Dict
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.schema import BasePromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import Optional
from Container import g_container
from config import QWEN_CONFIG
from TxtProcess import faiss_kb_ds

_PROMPT_TEMPLATE = """
你是一个帮我分析用户提问中重要内容的人工助手。
对于用户提出的问题，你应该对问题进行理解和拆解，从中找出能够作为关键词的名词。
你所找出的关键词必须是名词或名词组合，可以是主语、宾语、时间日期或用于修饰的名词。
如果存在并列结构，你需要将它拆分、重组成为单独的名词组合，例如：中信公司和大成公司的税收指标。===>中信公司的税收指标，大成公司的税收指标
在>>>> >>>>之间是一些示例：

>>>>

Question:2008年南京中电联环保股份有限公司营业收入增长主要来源是什么？
Answer:
    "company":["南京中电联环保股份有限公司"],
    "date":["2008年"],
    "keywords":["营业收入来源"]

Question:云南沃森生物技术股份有限公司先后承担了多少国家级、省级、市级课题项目？
Answer:
    "company":["云南沃森生物技术股份有限公司"],
    "date":[],
    "keywords":["国家级课题项目","省级课题项目","市级课题项目","课题项目"]

>>>>

你的回答应严格按照下面的格式，请注意---output 等标记都必须输出，这是我用来提取答案的标记。
不要输出中文的逗号，不要输出引号。
不可以编造答案，不可以输出要求以外的任何文字。

Question: ${{用户的问题}}

---output
${{"company":${{你的回答}},"date"${{你的回答}},:"keywords":${{你的回答}}}}

现在，我们开始作答
Question: {question}

"""

PROMPT=PromptTemplate(
    input_variables=['question'],
    template=_PROMPT_TEMPLATE
)

_PROMPT_TEMPLATE2 = """
你是一个帮我回答用户提问的人工助手。
对于用户提出的问题，你需要从相关内容中找到答案，并与问题进行组合作为答案，不可以只返回答案。
你所找到的答案的上下文中必须包含至少1个关键词。
在>>>> >>>>之间是一些示例：

>>>>

Question:黄山胶囊厂设立时的账面总资产和评估价值是多少？
Keywords:账面总资产;评估价值
Content:一）设立时的资产评估情况 1995 年 9 月，以 1995 年 7 月 31 日为评估基准日，安徽东南资产评估公司对安徽省旌德县黄山胶囊厂整体资产进行评估，并出具了《资产评估结果报告书》（皖东南资评字（1995）033 号）。资产评估结果如下：总资产账面价值6,649,648.36 元，评估价值 7,710,308.39 元，评估增值 1,060,660.03 元，增值率 15.95%。总负债账面价值 5,805,132.43 元，评估价值 5,860,996.17 元，评估增值额-55,863.74元，增值率-0.96%。净资产账面价值844,515.93元，评估价值1,849,312.22元，评估增值额1,004,796.29元，增值率118.98%。（二）整体变更时的资产评估情况 有限公司整体变更为股份有限公司时，总资产账面价值为 18,299.16 万元，评估价值27,508.94万元，增值额9,209.78万元。

Answer:黄山胶囊厂设立时的账面总资产是6,649,648.36元，评估价值是1,849,312.22元。

>>>>

在你的回答中不要输出中文的逗号，不要输出引号。
你不可以编造答案，不可以输出要求以外的任何多余文字。如果你找不到答案，直接回答我不知道。
你的回答应严格按照下面的格式，请注意---output等标记都必须输出，这是我用来提取答案的标记。

Question: ${{用户的问题}}
Keywords: ${{用户问题的关键词}}
Content: ${{相关内容}}

---output
${{你的回答}}

现在，我们开始作答
Question: {question}
Keywords: {keywords}
Content: {content}
"""

PROMPT2=PromptTemplate(
    input_variables=['question','content','keywords'],
    template=_PROMPT_TEMPLATE2
)

class LLMKnowledgeChain(LLMChain):
    llm_chain:LLMChain
    llm_chain2:LLMChain
    llm:Optional[BaseLanguageModel] = None
    stopwords :List[str] = ['---output']
    prompt:BasePromptTemplate=PROMPT
    prompt2:BasePromptTemplate=PROMPT2
    input_key:str='inputs'
    output_key:str='outputs'
    
    config : QWEN_CONFIG
    
    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]
    
    
    def _process_llm_out(self,llm_out:str,stop:str)->str:
        llm_out=llm_out.split('\n')
        idx=llm_out.index(stop)+1
        return str(llm_out[idx])
        
    
    def _call(
        self,
        inputs:dict,
        run_manager:Optional[CallbackManagerForChainRun] = None
        )->dict:
        _run_manager=run_manager or CallbackManagerForChainRun.get_noop_manager()
        _run_manager.on_text(text=self.input_key)
        
        kb_idx=faiss_kb_ds(vs_path=self.config.vec_store_path,
                           kb_path='./',
                           embs=g_container.EMBEDDING)
        kb_idx.do_load(index='index')
        
        llm_output=self.llm_chain.predict(
            question=inputs[self.input_key],
            stop=['---output'],
            callbacks=_run_manager.get_child()
        )
        llm_output=self._process_llm_out(llm_output,'---output')
        query_info = json.loads(llm_output)
        
        companys=query_info['company']
        doc_ids=[]
        pagelist=[]
        for com in companys:
            doc=kb_idx.do_search(query=com,top_k=1)['documents'][0]
            if doc.metadata['source'] not in doc_ids:
                doc_ids.append(doc.metadata['source'])
        for did in doc_ids:
            kb=faiss_kb_ds(vs_path=self.config.vec_store_path,
                           kb_path='./',
                           embs=g_container.EMBEDDING)
            kb.do_load(index=did)
            for keyword in query_info['keywords']:
                docs_list=kb.do_search(query=keyword,top_k=5,threshold=0.5)['documents']
                doc_pages=[d.page_content for d in docs_list]
                pagelist.extend(doc_pages)
        pagelist=list(set(pagelist))
        
        ans=[]
        for page in pagelist:
            llm_output=self.llm_chain2.predict(
                question = inputs[self.input_key],
                content = page,
                keywords = ';'.join(query_info['date'])+';'.join(query_info['keywords']),
                stop=['---output'],
                callbacks=_run_manager.get_child()
            )
            llm_output = self._process_llm_out(llm_output,'---output')
            ans.append(llm_output)
        
        llm_output=self.llm_chain2.predict(
            question = inputs[self.input_key],
            content = '。'.join(ans),
            keywords = ';'.join(query_info['date'])+';'.join(query_info['keywords']),
            stop=['---output'],
            callbacks=_run_manager.get_child()
        )
        llm_output = self._process_llm_out(llm_output,'---output')
        
        final_ans={
            self.input_key:inputs[self.input_key],
            self.output_key:llm_output.strip(),
            'relent_docs':doc_ids,
        }
        return final_ans[self.output_key]
            
    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        cfg: QWEN_CONFIG,
        prompt: BasePromptTemplate = PROMPT,
        prompt2: BasePromptTemplate = PROMPT2,
        **kwargs: Any,
    ):
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        llm_chain2 = LLMChain(llm=llm,prompt=prompt2)
        return cls(llm_chain=llm_chain, llm_chain2=llm_chain2,llm=llm,config=cfg,verbose=True,**kwargs)


def run_knowledge_qa(query:str):
    llm_model=g_container.MODEL
    llm_chain=LLMKnowledgeChain.from_llm(llm=llm_model,cfg=g_container.GCONFIG)
    ans=llm_chain._call({llm_chain.input_key:query})
    return ans

from pydantic import BaseModel, Field
class KnowledgeSearchInput(BaseModel):
    question: str = Field(description="需要在知识库中搜索相关答案的问题。")