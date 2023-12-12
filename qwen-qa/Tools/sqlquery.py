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
你是一个帮助用户查询sql数据库的助手。
下面是你能访问的数据库的信息:

>>>>
{database_info}
>>>>

请你思考用户的要求，并按照要求，写出能够使用python的sqlite3库执行的SQL查询语句。
你的回答应严格按照示例格式。不要输出中文的逗号，不要输出引号。
你所使用的数据库表项、字段等，都必须严格按照提供给你的数据库信息。
你的回答使用如下的示例：

>>>>
Question:用户的查询需求。
Thought:应该使用的数据库中的表项和相应字段。
Answer:你编写的能够执行的SQL查询语句。
>>>>

现在，我们开始作答：

Question:{question}

"""

PROMPT=PromptTemplate(
    input_variables=['question','database_info'],
    template=_PROMPT_TEMPLATE
)

_PROMPT_TEMPLATE2="""
你是一个帮助用户纠正SQL查询语句的助手。
请你纠正用户的SQL语句中的错误，包括表、字段名称错误，语法错误等。
SQL查询语句所使用的表项、字段名称应与下面数据库信息中的内容完全相同。
表、字段中包含的特殊符号等也不能省略。
下面是用户使用的数据库的信息:

>>>>
{database_info}
>>>>

你的回答应严格按照示例格式。不要输出中文的逗号，不要输出引号。
你的回答使用如下的示例：

>>>>
UserSQL:用户的SQL查询语句。
Thought:用户SQL语句存在的错误。
Answer:你纠错后的SQL查询语句。
>>>>

现在，我们开始作答：

UserSQL:{SQLstring}

"""

PROMPT2=PromptTemplate(
    input_variables=['SQLstring','database_info'],
    template=_PROMPT_TEMPLATE2
)

class LLMSQLChain(LLMChain):
    llm_chain:LLMChain
    llm_chain2:LLMChain
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
        outs=llm_out.split('\n')
        _run_manager=run_manager or CallbackManagerForChainRun.get_noop_manager()
        result=' '
        for out in outs:
            if out.startswith('Answer'):
                _run_manager.on_text(text=out)
                result=out.split(':')[1].strip()
            else:
                _run_manager.on_text(text=out)
        return result 
    
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
            database_info=g_container.DATABASE,
            callbacks=_run_manager.get_child()
        )
        llm_out=self._process_llm_out(llm_out=llm_out,run_manager=_run_manager)
        
        llm_crt=self.llm_chain2.predict(
            SQLstring=llm_out,
            database_info=g_container.DATABASE,
            callbacks=_run_manager.get_child()
        )
        llm_crt=self._process_llm_out(llm_crt)
        
        conn=sqlite3.connect(g_container.DATABASE['db_path'])
        cursor=conn.cursor()
        sql_result=cursor.execute(llm_crt)
        conn.close()
        print(sql_result)
    
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
            llm_chain2 = LLMChain(llm=llm, prompt=prompt2)
            return cls(llm_chain=llm_chain,llm_chain2=llm_chain2,llm=llm,config=cfg,**kwargs)
        
        
def run_sql_query(query:str)->str:
    llmchain=LLMSQLChain.from_llm(llm=g_container.MODEL,cfg=g_container.GCONFIG)
    ans=llmchain._call({llmchain.input_key:query})
    return ans