from __future__ import annotations
from langchain.agents import Tool, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from typing import List
from langchain.schema import AgentAction, AgentFinish

from Container import g_container
import json
class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]

    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

class CustomOutputParser(AgentOutputParser):
    begin: bool = False
    def __init__(self):
        super().__init__()
        self.begin = True

    def parse(self, llm_output: str) -> AgentFinish | tuple[dict[str, str], str] | AgentAction:
        if not g_container.MODEL and self.begin:
            self.begin = False
            stop_words = ["Observation:"]
            min_index = len(llm_output)
            for stop_word in stop_words:
                index = llm_output.find(stop_word)
                if index != -1 and index < min_index:
                    min_index = index
                llm_output = llm_output[:min_index]

        if "Final Answer" in llm_output :
            self.begin = True
            fa = llm_output.split("Final Answer", 1)[-1].split('action_input":',1)[-1].split("}",1)[0].strip()
            return AgentFinish(
                return_values={"output": fa},
                log=llm_output,
            )
        if  "FinalAnswer" in llm_output:
            self.begin = True
            fa = llm_output.split("FinalAnswer", 1)[-1].split('action_input":',1)[-1].split("}",1)[0].strip()
            return AgentFinish(
                return_values={"output": fa},
                log=llm_output,
            )
        llm_output=llm_output.replace('\n','').replace(' ','')
        parts = llm_output.split("Action:")
        if len(parts) < 2:
            return AgentFinish(
                return_values={"output": f"调用agent工具失败，该回答为大模型自身能力的回答:\n\n `{llm_output}`"},
                log=llm_output,
            )
        # action = parts[1].split('"action_input":')[0].strip()
        # action_input = parts[1].split('"action_input":')[1].strip()
        action_text = parts[1].replace('`','').replace('\{\{','{').replace('\}\}','}')
        print(action_text)
        action_dict = json.loads(action_text)
        action = action_dict['action']
        action_input = str(action_dict['action_input'])
        try:
            ans = AgentAction(
                tool=action,
                tool_input=action_input.strip(" ").strip('"'),
                log=llm_output
            )
            return ans
        except:
            return AgentFinish(
                return_values={"output": f"调用agent失败: `{llm_output}`"},
                log=llm_output,
            )