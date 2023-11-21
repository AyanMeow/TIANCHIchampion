

class QWEN_CONFIG(object):
    """QWEN_CONFIGs.
    
    """
    def __init__(self,
                 online_emb = (False,''),
                 local_emb = (True,''),
                 vec_store_path = './vec_store',
                 vec_search_topK = 5,
                 chunk_size = 1000,
                 chunk_overlap_rate = 0.1,
                 online_model_pipe = (False,''),
                 local_model_path = (True,'')
                 ):
        super(QWEN_CONFIG, self,).__init__()
        #标书等文本向量设置
        self.online_emb = online_emb
        self.local_emb = local_emb
        self.vec_store_path = vec_store_path
        self.vec_search_topK = vec_search_topK
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap_rate * self.chunk_size
        self.online_model_pipe = online_model_pipe
        self.local_model_path = local_model_path,
        #sqlite数据库设置
        