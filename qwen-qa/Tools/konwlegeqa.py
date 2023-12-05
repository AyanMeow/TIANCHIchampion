
from typing import Any, Type
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


_PROMPT_TEMPLATE = """
你是一个帮我分析用户提问中重要内容的人工助手。
对于用户提出的问题，你应该对问题进行理解和拆解，从中找出能够作为关键词的名词。
你所找出的关键词必须是名词或名词组合，可以是主语、宾语、时间日期或用于修饰的名词。
如果存在并列结构，你需要将它拆分、重组成为单独的名词组合，例如：中信公司和大成公司的税收指标。===>中信公司的税收指标，大成公司的税收指标

对于用户的问题，你的输出应该按照下面例子的格式：

question:中国铁路通信信号股份有限公司提供轨道交通控制系统全产业链上的产品及服务，主要业务包括哪些？
answer:中国铁路通信信号股份有限公司,轨道交通控制系统全产业链的主要业务

下面这些数据库是你能访问的，冒号之前是他们的编号，冒号之后是他们的描述，你应该参考每个知识库的描述来帮助你思考：

{database_subscription}

你的回答格式应该按照下面的内容，请注意```output 等标记都必须输出，这是我用来提取答案的标记。
不要输出中文的逗号，不要输出引号。

Question: ${{用户的问题}}

```output
知识库的编号

现在，我们开始作答
问题: {question}
"""

PROMPT=PromptTemplate(
    input_variables=['question','database_subscription'],
    template=_PROMPT_TEMPLATE
)

class LLMKnowledgeChain(LLMChain):
    llmchain:LLMChain
    llm_model:Optional[BaseLanguageModel] = None
    database_subscription:dict=g_container['knowledge']
    prompt:BasePromptTemplate=PROMPT
    input_key:str='question'
    output_key:str='answer'
    
    def _process_llm_out(self,llm_out,llm_in,run_manager):
        run_manager.on_text(llm_out, color="green")
        
    
    def _call(
        self,
        inputs:dict,
        run_manager:Optional[CallbackManagerForChainRun] = None
        ):
        _run_manager=run_manager or CallbackManagerForChainRun.get_noop_manager()
        _run_manager.on_text(text=self.input_key)
        database_name=[f'{k}:{v}' for k,v in self.database_subscription.items()]
        llm_output=self.llmchain(database_subscription=database_name,
                                 question=inputs[self.input_key],
                                 stop=['```output'],
                                 callbacks=_run_manager.get_child())
        return self._process_llm_out(llm_output,inputs[self.input_key],_run_manager)
        
    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: BasePromptTemplate = PROMPT,
        **kwargs: Any,
    ):
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        return cls(llm_chain=llm_chain, **kwargs)

def run_knowledge_qa(query:str):
    llm_model=g_container.MODEL
    llm_chain=LLMKnowledgeChain.from_llm(llm=llm_model,prompt=PROMPT)
    ans=llm_chain.run({'question':query})
    return ans