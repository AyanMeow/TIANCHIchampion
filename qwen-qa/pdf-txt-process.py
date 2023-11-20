from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader

def load_dir_and_split(file_dir_path=None):
    '''
    handle files in file_dir
    split and vectorization
    '''
    assert file_dir_path == None,'empty file dir'
    
    