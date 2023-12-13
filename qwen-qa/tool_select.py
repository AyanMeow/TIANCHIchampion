from langchain.tools import Tool
from Tools.konwlegeqa import run_knowledge_qa,KnowledgeSearchInput
from Tools.calculate import run_calculate,CalculatorInput
from Tools.sqlquery import run_sql_query,SQLSearchInput
from Tools.recognition import run_recognition,RecognitionInput

tools = [
    Tool.from_function(
        func=run_calculate,
        name='calculate',
        description="使用这个工具来进行数学计算。",
        #args_schema=CalculatorInput
    ),
    Tool.from_function(
        func=run_knowledge_qa,
        name='knowledegQA',
        description="使用这个工具来为用户的提问搜索本地文档知识库中的答案。工具的输入为<用户问题>。",
        #args_schema=KnowledgeSearchInput
    ),
    Tool.from_function(
        func=run_sql_query,
        name="SQLsearch",
        description="使用这个工具来为用户的提问搜索本地SQL数据库中的内容。工具的输入为<用户问题>。",
        #args_schema=SQLSearchInput
    ),
    Tool.from_function(
        func=run_recognition,
        name="IntentRecognition",
        description="用户问题意图不明确时，使用这个工具分析用户问题的真实意图。工具的输入为<用户问题>。",
        #args_schema=RecognitionInput
    )
]

tool_names = [tool.name for tool in tools]