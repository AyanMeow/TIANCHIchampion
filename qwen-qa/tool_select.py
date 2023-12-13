from langchain.tools import Tool
from Tools.konwlegeqa import run_knowledge_qa,KnowledgeSearchInput
from Tools.calculate import run_calculate,CalculatorInput
from Tools.sqlquery import run_sql_query,SQLSearchInput

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
        description="使用这个工具来为用户的提问搜索本地文档知识库中的答案。工具的输入为<用户问题>,例如：云南沃森生物技术股份有限公司负责产品研发的是什么部门？",
        #args_schema=KnowledgeSearchInput
    ),
    Tool.from_function(
        func=run_sql_query,
        name="SQLsearch",
        description="使用这个工具来为用户的提问搜索本地SQL数据库中的内容。工具的输入为<用户问题>,例如:20210618日，一级行业为基础化工的股票的成交量合计是多少？取整。",
        #args_schema=SQLSearchInput
    )
]

tool_names = [tool.name for tool in tools]