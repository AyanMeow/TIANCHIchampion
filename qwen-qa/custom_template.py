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

        if "Final Answer:" in llm_output:
            self.begin = True
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:", 1)[-1].strip()},
                log=llm_output,
            )
        # 'Question:我想知道股票600887在申万行业分类下的二级行业是什么？用最新的数据。Thought:用户想了解股票600887在申万行业分类下的二级行业，需要使用知识库工具查找最新的数据。Action:{"action": "knowledegQA","action_input": "股票600887在申万行业分类下的二级行业是什么"}'
        llm_output=llm_output.replace('\n','').replace(' ','')
        parts = llm_output.split("Action:")
        if len(parts) < 2:
            return AgentFinish(
                return_values={"output": f"调用agent工具失败，该回答为大模型自身能力的回答:\n\n `{llm_output}`"},
                log=llm_output,
            )
        # action = parts[1].split('"action_input":')[0].strip()
        # action_input = parts[1].split('"action_input":')[1].strip()
        action_dict = json.loads(parts[1])
        action = action_dict['action']
        action_input = action_dict['action_input']
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