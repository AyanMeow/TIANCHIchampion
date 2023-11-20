

class QWEN_CONFIG(object):
    """QWEN_CONFIGs.
    
    """
    def __init__(self, ):
        super(QWEN_CONFIG, self,
              ).__init__()
        self.chunk_size = 2000
        self.chunk_overlap = 0.1 * self.chunk_size
        self.online_model_pipe = (False,'')
        self.local_model_path = None
    