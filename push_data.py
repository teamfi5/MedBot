# Đẩy dữ liệu mới lên CSDL
# Dữ liệu mới được đặt ở push_folder dưới dạng pdf

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from uuid import uuid4
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
import fitz
import os

prj_path = r'D:\Huan\Project\KHDL'
push_folder = 'push_data'
EMBEDDING_MODEL_PATH = r'D:\Huan\Project\KHDL\model\embedding_model\bge-m3-ft-triplet'

# Tạo đối tượng TextLoader để load tài liệu
class DataframeLoader(TextLoader):
    def __init__(self, folder_path):
        self.documents = []
        
        for data in os.listdir(folder_path):
            docs = []
            if (data[-4:] != '.pdf'):
                print(f'{data} không phải file pdf')
                continue
            with fitz.open(folder_path + '/' + data) as doc:
                # Đọc nội dung từng trang
                for page in doc:
                    docs.append(page.get_text())
                document = ''.join(docs)
            self.documents.append((data, document))
        
        self.exists_pdf = set(os.listdir(prj_path + '/pdf'))
        
        print(f'Tìm thấy {len(self.documents)} files')

    def load(self):
        _documents = []
        for document in self.documents:
            if 'n_wk-' + document[0].replace(' ', '_') in self.exists_pdf:
                print('n_wk-' + document[0].replace(' ', '_') + ' đã tồn tại trong CSDL')
            else:
                _documents.append(Document(page_content=document[1], metadata={"source": 'n_wk-' + document[0].replace(' ', '_')}))
        return _documents
    
# Load tài liệu và chia thành các chunk
loader_data = DataframeLoader(prj_path + '/' + push_folder)

documents_data = loader_data.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                chunk_overlap=400)

texts_data = text_splitter.split_documents(documents_data)

# Load Embedding model
print('Load embedding model...')
embeddings = HuggingFaceBgeEmbeddings(model_name=EMBEDDING_MODEL_PATH)
print('Done')

# Tải vector store cũ
vector_store = FAISS.load_local(
    folder_path = 'vector_store', embeddings = embeddings, allow_dangerous_deserialization = True
)

# Tạo vector store mới
_index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
_uuids = [str(uuid4()) for _ in range(len(texts_data))]
_vector_store = FAISS(
    embedding_function=embeddings,
    index=_index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)
_vector_store.add_documents(documents=texts_data, ids=_uuids)

# Trộn 2 vector store rồi lưu lại
vector_store.merge_from(_vector_store)
vector_store.save_local('vector_store')

print(f"Loaded {len(texts_data)} documents and saved")