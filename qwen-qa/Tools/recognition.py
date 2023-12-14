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
import sqlite3

_PROMPT_TEMPLATE="""
请你思考用户的问题，并按照下面的格式判断用户问题的意图。
>>>>
Question:用户的问题。
Thought:用户的真实意图。
Answer:[chat,calculate,knowledge_QA,sql_search]中的某一个。
>>>>

下面是一些判断用户意图的示例：
>>>>
Question:你可以帮助我吗？
Thought:用户的输入属于对话行为
Answer:chat

Question:广东银禧科技股份有限公司注册资本是多少？
Thought:用户需要问题相关的资料，应查询知识库
Answer:knowledge_QA

Question:5的平方减去1，答案是多少？
Thought:用户需要进行数学计算
Answer:calculate

Question:帮我看一下在20190502,代码为00292的港股日价格涨幅是多少？
Thought:有具体字段的值，用户需要查询SQL数据库
Answer:sql_search
>>>>

你的回答应严格按照示例格式。不要输出中文的逗号。

现在，我们开始作答：

Question:{question}
"""

PROMPT=PromptTemplate(
    input_variables=['question'],
    template=_PROMPT_TEMPLATE
)

class LLMRecChain(LLMChain):
    llm_chain:LLMChain
    llm:Optional[BaseLanguageModel] = None
    stopwords :List[str] = None
    prompt:BasePromptTemplate=PROMPT
    input_key:str='inputs'
    output_key:str='outputs'
    config : QWEN_CONFIG
    
    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]
    
    def _process_llm_out(self,llm_out:str,run_manager: CallbackManagerForChainRun = None)->str:
        
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        _run_manager.on_text(text=llm_out)
        outs=llm_out.split('Answer:')
        return outs[1]
    
    def _call(
        self, 
        inputs: Dict[str, Any], 
        run_manager: CallbackManagerForChainRun = None,
        ) -> Dict[str, str]:
        query=inputs[self.input_key]
        _run_manager=run_manager or CallbackManagerForChainRun.get_noop_manager()
        _run_manager.on_text(text=query)
        llm_out=self.llm_chain.predict(
            question=query,
            database_info=json.dumps(g_container.DATABASE,ensure_ascii=False),
            callbacks=_run_manager.get_child()
        )

        llm_out=self._process_llm_out(llm_out=llm_out,run_manager=_run_manager)
          
        final_ans = {
            self.input_key:query,
            self.output_key:llm_out
        }
        
        return final_ans[self.output_key]
        
    @classmethod
    def from_llm(
            cls,
            llm: BaseLanguageModel,
            cfg: QWEN_CONFIG,
            prompt: BasePromptTemplate = PROMPT,
            **kwargs: Any,
        ):
            llm_chain = LLMChain(llm=llm, prompt=prompt)
            return cls(llm_chain=llm_chain,llm=llm,config=cfg,verbose=True,**kwargs)
        
        
def run_recognition(query:str)->dict:
    llmchain=LLMRecChain.from_llm(llm=g_container.MODEL,cfg=g_container.GCONFIG)
    ans=llmchain._call({llmchain.input_key:query})
    return ans

import pydantic
class RecognitionInput(pydantic.BaseModel):
    question: str = pydantic.Field()