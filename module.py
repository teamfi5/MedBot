from generation import QAmodel
from retrieve import retrieve
from rerank import reranker

class Bot:
    def __init__(self):
        self.re = retrieve()
        
        self._reranker = reranker()

        self.model = QAmodel()
        
    def generate(self, input_text):
        
        # Truy xuất 16 context 
        _docs = self.re.similarity_search(query=input_text,top_k=16)
        docs = [i.page_content for i in _docs]
        
        # Xếp hạng lại
        result = self._reranker.rerank(query=input_text, contexts = docs)
        
        # TỔng hợp 3 context tốt nhất
        contexts = ''.join(result[0][:3])
        
        # Trả về liên kết của 3 context
        links_pdf = set([self.re.search_doc_pdf(_docs[result[1][i]]) for i in range(3)])
        
        text = self.model.generation(input_text, contexts)
        return (text, links_pdf)
        
        

