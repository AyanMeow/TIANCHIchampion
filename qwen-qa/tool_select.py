from langchain.tools import Tool
from Tools.konwlegeqa import run_knowledge_qa,KnowledgeSearchInput


tools = [
    Tool.from_function(
        func=run_knowledge_qa,
        name='knowledegQA',
        description="Use this tool to search local knowledgebase and get information for questions",
        args_schema=KnowledgeSearchInput
    )
]

tool_names = [tool.name for tool in tools]