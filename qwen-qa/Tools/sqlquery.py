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
你所使用的数据库的table_name、column_name等，都必须严格与提供给你的数据库信息中的table_name、column_name一致。
table_name、column_name应被""包裹。
你的输出应严格按照如下示例。
你的回答使用如下的示例：

>>>>
Question:用户的查询需求。
Thought:应该使用的数据库中的表项和相应字段。
Answer:你编写的能够执行的SQL查询语句。
>>>>

注意:Question及其内容也需要一并输出。现在，我们开始作答：

Question:{question}

"""

PROMPT=PromptTemplate(
    input_variables=['question','database_info'],
    template=_PROMPT_TEMPLATE
)

_PROMPT_TEMPLATE2="""
你是一个帮助用户纠正SQL查询语句的助手。
请你根据错误信息纠正SQL语句中的错误，包括table_name、column_name错误，语法错误等。如果不存在错误，则不需要进行修改。
SQL查询语句所使用的table_name、column_name应与下面数据库中的table_name、column_name完全一致。
table_name、column_name中包含的特殊符号等也不能省略。
table_name、column_name应被""包裹。
下面是用户使用的数据库的信息:

>>>>
{database_info}
>>>>

你的回答应严格按照示例格式。不要输出中文的逗号。
你的回答使用如下的示例：

>>>>
History:用户与其他LLM交互的历史。
UserSQL:需要纠错的SQL查询语句。
ERROR:用户SQL语句执行时产生的错误。
Thought:用户SQL语句存在的错误。
Answer:你纠错后的SQL查询语句。
>>>>

现在，我们开始作答：

History:{history}
UserSQL:{SQLstring}
ERROR:{error}

"""

PROMPT2=PromptTemplate(
    input_variables=['SQLstring','history','error','database_info'],
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
        history=[]
        for out in outs:
            if out.startswith('Question') or out.startswith('User'):
                _run_manager.on_text(text=out)
                user=out.split(':')[1].strip()
                history.append('User:'+user)
            elif out.startswith('Answer'):
                _run_manager.on_text(text=out)
                result=out.split(':')[1].strip()
                history.append("Assistant:"+result)
            else:
                _run_manager.on_text(text=out)
                ass=out.split(':')[1].strip()
                history.append("Assistant:"+ass)
        return result , history
    
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

        llm_out,history=self._process_llm_out(llm_out=llm_out,run_manager=_run_manager)
        
        conn=sqlite3.connect(g_container.DATABASE['db_path'])
        cursor=conn.cursor()
        while True:
            error=None
            try:
                sql_result=cursor.execute(llm_out)
            except sqlite3.OperationalError as e:
                error=str(e)
                _run_manager.on_text(text=error)
                llm_out=self.llm_chain2.predict(
                    SQLstring=llm_out,
                    history=history,
                    error=error,
                    database_info=json.dumps(g_container.DATABASE,ensure_ascii=False),
                    callbacks=_run_manager.get_child()
                )
                llm_out,_=self._process_llm_out(llm_out)
            if error == None:break
        answers=[data for data in sql_result]
        conn.close()
        
        final_ans = {
            self.input_key:query,
            "SQLquery":llm_out.strip(),
            self.output_key:answers
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
            llm_chain2 = LLMChain(llm=llm, prompt=prompt2)
            return cls(llm_chain=llm_chain,llm_chain2=llm_chain2,llm=llm,config=cfg,verbose=True,**kwargs)
        
        
def run_sql_query(query:str)->dict:
    llmchain=LLMSQLChain.from_llm(llm=g_container.MODEL,cfg=g_container.GCONFIG)
    ans=llmchain._call({llmchain.input_key:query})
    return ans

from pydantic import BaseModel, Field
class SQLSearchInput(BaseModel):
    question: str = Field(description="需要在SQL数据库中搜索相关答案的问题。")