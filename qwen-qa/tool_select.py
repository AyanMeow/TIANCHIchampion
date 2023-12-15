from langchain.tools import Tool
from Tools.konwlegeqa import run_knowledge_qa,KnowledgeSearchInput
from Tools.calculate import run_calculate,CalculatorInput
from Tools.sqlquery import run_sql_query,SQLSearchInput
from Tools.recognition import run_recognition,RecognitionInput

#工具的输入为<$Questiion>。

tools = [
    Tool.from_function(
    func=run_recognition,
    name="IntentRecognition",
    description="你应该**首先**使用这个工具分析用户问题的真实意图。需要的输入:<用户原问题>。",
    args_schema=RecognitionInput
    ),
    Tool.from_function(
        func=run_calculate,
        name='calculate',
        description="使用这个工具来进行数学计算。",
        args_schema=CalculatorInput
    ),
    Tool.from_function(
        func=run_knowledge_qa,
        name='knowledgeQA',
        description="使用这个工具来为用户的提问搜索本地文档知识库中的答案。",
        args_schema=KnowledgeSearchInput
    ),
    Tool.from_function(
        func=run_sql_query,
        name="DBsearch",
        description="使用这个工具来为用户的提问搜索本地SQL数据库中的内容。",
        args_schema=SQLSearchInput
    ),

]

tool_names = [tool.name for tool in tools]