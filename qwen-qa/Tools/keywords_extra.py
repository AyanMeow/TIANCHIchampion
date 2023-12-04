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
from jieba import analyse

import json



_PROMPT_TEMPLATE = """
请你提取下面段落的关键词，要求所提取的关键词能够鲜明表示内容性质。
如果有必要，你可以使用金融领域关键词进行概括。
你所提取的关键词应该是名词短语，不要包含谓语、助词等。
你所提取的关键词应不少于{topk}个。
同时，对于内容中存在的公司、机构名称，地址信息，人名等，也作为关键词。

待处理的内容：
{content}

注意：你的回答不需要分段。你的回答格式应该按照下面的内容，请注意---output 等标记都必须输出，这是我用来提取答案的标记。
除了你的回答之外不要输出多余的文字。不要编造答案。

---output
${{"keywords":[你提取的关键词]}}
"""

PROMPT=PromptTemplate(
    input_variables=['content','topk'],
    template=_PROMPT_TEMPLATE
)

class LLMKeywordsEXChain(LLMChain):
    llm_chain:LLMChain
    llm:Optional[BaseLanguageModel] = None
    stopwords :List[str] = None
    prompt:BasePromptTemplate=PROMPT
    input_key:str='inputs'
    output_key:str='outputs'
    
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
    
    def _process_output_with_stop(self,outputs:str,stop:str):
        if stop in outputs:
            outputs=outputs.split("\n")[-1]
        outputs=outputs.strip()
        return outputs    
        
        
    def _call(
        self, 
        inputs: Dict[str, Any],
        top_k: int=20, 
        run_manager = None
        ) -> Dict[str, str]:
        _run_manager=run_manager or CallbackManagerForChainRun.get_noop_manager()
        _run_manager.on_text(text=self.input_key)
        
        #TF-IDF-效果不好
        texts=inputs[self.input_key]
        extration=analyse.extract_tags(texts,topK=top_k, withWeight=False, allowPOS=())
        extration_tf=[e for e in extration if e not in self.stopwords]
        
        #Hanlp
        
        
        #LLM
        stop='---output'
        llm_output=self.llm_chain.predict(
            content=inputs[self.input_key],
            topk=top_k,
            stop=[stop],
            callbacks=_run_manager.get_child())
        llm_output=self._process_output_with_stop(llm_output,stop)
        llm_outdict=json.loads(llm_output)
        extration_llm=llm_outdict['keywords']
        
        extration = list(dict.fromkeys(extration_tf+extration_llm))
        
        return extration
    
    async def _acall(
        self, 
        inputs: Dict[str, Any], 
        top_k: int=20, 
        run_manager: AsyncCallbackManagerForChainRun = None
        ) -> Coroutine[Any, Any, Dict[str, str]]:
        _run_manager=run_manager or CallbackManagerForChainRun.get_noop_manager()
        await _run_manager.on_text(text=self.input_key)
        
        #TF-IDF
        texts=inputs[self.input_key]
        extration=analyse.extract_tags(texts,topK=top_k, withWeight=False, allowPOS=())
        extration_tf=[e for e in extration if e not in self.stopwords]
        
        #LLM
        llm_output=await self.llm_chain.apredict(
            content=inputs[self.input_key],
            topk=top_k,
            stop=['---output'],
            callbacks=_run_manager.get_child())
        llm_outdict=json.dumps(llm_output)
        extration_llm=llm_outdict['keywords']
        
        extration = list(dict.fromkeys(extration_tf+extration_llm))
        
        return extration
    
    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: BasePromptTemplate = PROMPT,
        stopwords_path :str = './stopwords.txt'
    ):
        with open(stopwords_path,'r',encoding='utf-8') as f:
            stopw=f.readlines()
        stopw=[s.strip() for s in stopw]
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        return cls(llm_chain=llm_chain,stopwords=stopw)