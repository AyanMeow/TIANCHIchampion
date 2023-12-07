from dashscope import Generation,TextEmbedding

class QWEN_CONFIG(object):
    """QWEN_CONFIGs.
    
    """
    def __init__(self,
                 model_name = Generation.Models.qwen_max,
                 embedding_name = TextEmbedding.Models.text_embedding_v2,
                 api_key = None,
                 vec_store_path = './vec_store',
                 vec_search_topK = 5,
                 chunk_size = 700,
                 chunk_overlap_rate = 0.2,
                 ):
        super(QWEN_CONFIG, self,).__init__()
        #标书等文本向量设置
        self.model_name = model_name
        self.embedding_name = embedding_name
        self.api_key = api_key
        self.vec_store_path = vec_store_path
        self.vec_search_topK = vec_search_topK
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap_rate * self.chunk_size
        #sqlite数据库设置
        