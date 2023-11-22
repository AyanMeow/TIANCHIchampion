
from langchain.agents import tool
from langchain import PromptTemplate
import sqlite3


def gen_task_deifne_prompt():
    template="""我将给你一些示例，你需要按照示例将<input>进行分类。
    注意，你的分类结果只能是[database_query,document_query]中的某一个。
    请按照示例格式，以json的形式返回你对task的分类结果。
    注意，你只需要返回结果，不需要什么其他的解释或是多余的文字。

    example:

    input:湖南长远锂科股份有限公司变更设立时作为发起人的法人有哪些？
    category:document_query

    input:20210304日，一级行业为非银金融的股票的成交量合计是多少？取整。
    category:database_query

    input:云南沃森生物技术股份有限公司负责产品研发的是什么部门？
    category:document_query

    input:在20201022，按照中信行业分类的行业划分标准，哪个一级行业的A股公司数量最多？
    category:database_query

    task:

    input:{task_query}
    category:"""
    prompt=PromptTemplate(template=template, input_variables=["task_query"])
    return prompt