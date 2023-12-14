
_AGENT_PROMPT_TEMPLATE="""
你是一个会使用工具来回答用户问题的助手。
对于常识性问题或对话，你可以不使用工具，直接回答用户的问题；
你应该尽可能精准且全面地回答用户的问题。
下面是一些你能够使用的工具：
{tools}

使用$JSON_BLOB的形式来选择你要使用的工具,action用于存贮工具的名字(tool name);
而action_input用于存贮工具的输入。
action的值必须是 "Final Answer"或{tool_names} 中的某一个。
每一个$JSON_BLOB中只能使用一个action,如下所示:

>>>>
{{{{
    "action":$工具名称,
    "action_input":$工具输入
}}}}
>>>>

整个回答过程应如下所示例:

>>>>
Question:用户的提问
Thought:结合之前的步骤,考虑接下来该做什么
Action:
```
$JSON_BLOB
```
Observation:执行action得到的结果
....(必要时,重复上述 Thought/Action/Observation 过程n次)
Thought:现在我有了所有回答问题所需要的东西。
Action:
```
{{{{
    "action":"Fincal Answer",
    "action_input":"$问题的最终答案"
}}}}
```
>>>>

注意,你必须时刻按照示例的格式来进行每一步的思考,并在必要的时候使用工具。
现在,让我们开始作答。下面的内容必须全部输出。

history: {history}
Question: {input}
Thought: {agent_scratchpad}
Action:
"""
from langchain.agents import LLMSingleActionAgent, AgentExecutor
from custom_template import CustomPromptTemplate,CustomOutputParser
from history import History
from typing import List,Iterable
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from tool_select import tool_names,tools
from langchain.chains import LLMChain
from Container import g_container
from langchain.memory import ConversationBufferWindowMemory

def agent_qa(
    query:str,
    history:List[History]= [],
    run_manager:CallbackManagerForChainRun = None
)->Iterable[str]:
    _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
    
    prompt_tmplate=CustomPromptTemplate(
        template=_AGENT_PROMPT_TEMPLATE,
        tools=tools,
        input_variables=["input", "intermediate_steps", "history"]
    )
    output_parser = CustomOutputParser()
    llm_chain = LLMChain(
        llm=g_container.MODEL,
        prompt=prompt_tmplate
    )
    memory=ConversationBufferWindowMemory(k=g_container.MEMORY_WINDOW*2)
    
    for message in history:
        # 检查消息的角色
        if message.role == 'user':
            # 添加用户消息
            memory.chat_memory.add_user_message(message.content)
        else:
            # 添加AI消息
            memory.chat_memory.add_ai_message(message.content)
    
    agent=LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        stop=["\nObservation:", "Observation"],
        allowed_tools=tool_names,
    )
    
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=memory
    )
    AgentExecutor.acall
    res=agent_executor._call(
        {
            "input":query,
            #"intermediate_steps":[],
            "history":[]
        },
        run_manager=_run_manager)
    print(res)