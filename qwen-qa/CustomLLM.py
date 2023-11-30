from typing import Any, List, Mapping, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
import transformers as ts
from transformers import PreTrainedModel,PreTrainedTokenizer,PretrainedConfig
from typing import Optional,Literal,Dict

class LocalLLM(LLM):
    model :PreTrainedModel = None
    tokenizer :PreTrainedTokenizer = None
    model_cfg :Dict[Any,Any] = None
    maxlen :int = 1024
 
    
    @property
    def _llm_type(self) -> str:
        return self.model_cfg['model_type']
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return self.model_cfg
    
    def _call(
        self, 
        query: str, 
        stop: List[str] = None, 
        run_manager: CallbackManagerForLLMRun  = None, 
        **kwargs: Any) -> str:
        inputs=self.tokenizer(query,
                              padding=True,
                              max_length=self.maxlen,
                              truncation=True,
                              return_tensors='pt')
        inputs=inputs.to(self.model.device)
        outputs=self.model.generate(**inputs)
        outputs=self.tokenizer.decode(outputs.cpu()[0],skip_special_tokens=True)
        _run_manager = run_manager  or CallbackManagerForLLMRun.get_noop_manager()
        _run_manager.on_text(text=outputs,color="green")
        return outputs
    
    @classmethod
    def from_pretrain(
        cls,
        repo_path:str=None,
        maxlen:int=1024,
        device_map:str='auto',
        bf16:bool=False,
        fp16:bool=True,
        trust_remote_code: bool=True,
    ):
        if bf16:fp16=False
        
        tokenizer=ts.AutoTokenizer.from_pretrained(repo_path,
                                                   trust_remote_code=trust_remote_code,
                                                   padding_side="left",
                                                   pad_token="<|endoftext|>")
        cfg=ts.AutoConfig.from_pretrained(repo_path,trust_remote_code=trust_remote_code)
        model=ts.AutoModelForCausalLM.from_pretrained(repo_path,
                                                      config=cfg,
                                                      device_map=device_map,
                                                      trust_remote_code=trust_remote_code,
                                                      ).eval()
        model.generation_config = ts.generation.GenerationConfig.from_pretrained(repo_path,trust_remote_code=trust_remote_code)
        cfg=ts.AutoConfig.from_pretrained(repo_path,trust_remote_code=trust_remote_code).to_dict()
        return cls(model=model,tokenizer=tokenizer,maxlen=maxlen,model_cfg=cfg)
    

import dashscope
from dashscope import Generation
from http import HTTPStatus

class QWENonlie(LLM):
    model_name :str = Generation.Models.qwen_max
    temperature :float = 0.5
    max_token :int = 1500
    api_key :str = None
    
    @property
    def _llm_type(self) -> str:
        return self.model_name
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model_name":self.model_name,
            "temperature":self.temperature,
            "max_token":self.max_token
        }
        
    def _call(
        self, 
        prompt: str, 
        stop: List[str] = None, 
        run_manager: CallbackManagerForLLMRun = None, 
        **kwargs: Any
        ) -> str:
        dashscope.api_key=self.api_key
    
    @classmethod
    def from_name(
        cls,
        api_key :str = None,
        model_name :str = Generation.Models.qwen_max,
        temperature :float = 0.5,
        max_token :int = 1500,
    ):
        if not api_key :
            return 'empty api key'
        return cls(model_name,temperature,max_token,api_key)