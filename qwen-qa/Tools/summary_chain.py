from typing import Any, Coroutine, Dict, List, Type
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.pydantic_v1 import Extra, root_validator
from langchain.schema import BasePromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from typing import Optional



_PROMPT_TEMPLATE = """
请对下面段落的内容进行简要总结,对于其中存在的公司名称、机构名称、人名等，将其单独罗列一段。
需要总结的内容：
{content}

注意：
你应该用几句话描述你所总结的内容，禁止将照搬原句。
你的回答不需要分段。你的回答格式应该按照下面的内容，请注意---output 等标记都必须输出，这是我用来提取答案的标记。
除了你的回答之外不要输出多余的文字。

---output
${{"summary":你对段落内容的总结,"entity":[存在的公司名称、机构名称、人名等]}}
"""

PROMPT=PromptTemplate(
    input_variables=['content'],
    template=_PROMPT_TEMPLATE
)

class LLMSummaryChain(LLMChain):
    llm_chain:LLMChain
    llm:Optional[BaseLanguageModel] = None
    prompt:BasePromptTemplate=PROMPT
    input_key:str='content'
    output_key:str='summary'
    
    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid
        arbitrary_types_allowed = True
    
    @property
    def input_keys(self) -> List[str]:
        return self.input_keys
    
    @property
    def output_keys(self) -> List[str]:
        return self.output_keys
    
    def _call(
        self, 
        inputs: Dict[str, Any], 
        run_manager = None
        ) -> Dict[str, str]:
        _run_manager=run_manager or CallbackManagerForChainRun.get_noop_manager()
        _run_manager.on_text(text=self.input_key)
        llm_output=self.llm_chain.predict(
            content=inputs[self.input_key],
            stop=['---output'],
            callbacks=_run_manager.get_child())
        return llm_output
    
    async def _acall(
        self, 
        inputs: Dict[str, Any], 
        run_manager: AsyncCallbackManagerForChainRun = None
        ) -> Coroutine[Any, Any, Dict[str, str]]:
        _run_manager=run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        await _run_manager.on_text(text=self.input_key)
        llm_output=await self.llm_chain.apredict(
            content=inputs[self.input_key],
            stop=['```output'],
            callbacks=_run_manager.get_child())
        return llm_output
    
    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: BasePromptTemplate = PROMPT,
    ):
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        return cls(llm_chain=llm_chain)