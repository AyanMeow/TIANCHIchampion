from langchain.llms.base import LLM
from sqlite3 import Cursor
from langchain.embeddings.base import Embeddings
from config import QWEN_CONFIG

class GlobalContainer(object):
    """docstring for GlobalContainer."""
    def __init__(self, ):
        super(GlobalContainer, self).__init__()
        self.MODEL: LLM = None
        #self.CURSOR: Cursor = None
        self.DATABASE: dict = None
        self.EMBEDDING: Embeddings = None
        self.GCONFIG: QWEN_CONFIG = None
        
        self.MEMORY_WINDOW: int = 7
        
g_container=GlobalContainer()

    