from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os
import requests
from transformers import logging
logging.set_verbosity(logging.CRITICAL)

file_path = os.path.abspath(__file__)
PROJECT_PATH = os.path.dirname(file_path)

class retrieve:
    def __init__(self, vector_store_name = 'vector_store', embedding_model_name = 'bge-m3-ft-triplet'):
        print('Load embedding model...', end = '  ')
        EMBEDDING_MODEL_PATH = PROJECT_PATH + '/model/embedding_model/' + embedding_model_name
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)
        print('Done')
        
        print('Load vector store...', end = '  ')
        self.vector_store = FAISS.load_local(
            PROJECT_PATH + f'/{vector_store_name}', self.embeddings, allow_dangerous_deserialization=True
        )
        print('Done')
        
        self.list_id = set((i.split('-')[0], i) for i in os.listdir(PROJECT_PATH + '/pdf'))
        
    # Trả về file pdf có chứa dữ liệu trong doc 
    def search_doc_pdf(self, doc, pdf_path = PROJECT_PATH + r'\pdf'):
        page_id = str(doc.metadata['source'])
        if page_id.split('-')[0] == 'n_wk':
            return (pdf_path + '/' + page_id, page_id[3:-4].replace('_', ' '))
        for _page_id, pdf in self.list_id:
            if page_id == _page_id:
                path = pdf_path + '/' + pdf
                name = ''.join(pdf.split('-')[1:])[:-4].replace('_', ' ')
                return (path, name)
        
        url = "https://vi.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "pageids": page_id,
            "format": "json"
        }
        
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            title = data["query"]["pages"][str(page_id)]["title"]
            output_file = pdf_path + f'/{page_id}-{title.replace(' ', '_').replace('/', '_')}.pdf'
            pdf_url = f"https://vi.wikipedia.org/api/rest_v1/page/pdf/{title.replace(' ', '_').replace('/', r'%2f')}"
            response = requests.get(pdf_url)
            if response.status_code == 200:
                with open(output_file, "wb") as file:
                    file.write(response.content)
                self.list_id.add((page_id, output_file.split('/')[-1]))
                print(f"Tải xuống thành công: {output_file}")
                return (output_file, title)
            else:
                print(f"Lỗi tải xuống PDF cho {title}. Mã lỗi: {response.status_code}")
                return None
        else:
            print(f"Lỗi khi lấy title cho page_id {page_id}. Mã lỗi: {response.status_code}")
            return None
    
    # Tìm kiếm top_k tài liệu tương đồng nhất được tính bằng similarity cosin
    def similarity_search(self, query, top_k = 8):
        return self.vector_store.similarity_search(query=query, k = top_k)
        
    
    
        