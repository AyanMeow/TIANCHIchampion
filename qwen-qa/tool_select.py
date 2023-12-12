from langchain.tools import Tool
from Tools.konwlegeqa import run_knowledge_qa,KnowledgeSearchInput
from Tools.calculate import run_calculate,CalculatorInput

tools = [
    Tool.from_function(
        func=run_knowledge_qa,
        name='knowledegQA',
        description="使用这个工具来为用户的提问搜索本地文档知识库中的答案。例如：云南沃森生物技术股份有限公司负责产品研发的是什么部门？",
        args_schema=KnowledgeSearchInput
    ),
    Tool.from_function(
        func=run_calculate,
        name='calculate',
        description="使用这个工具来进行数学计算。",
        args_schema=CalculatorInput
    )
]

tool_names = [tool.name for tool in tools]