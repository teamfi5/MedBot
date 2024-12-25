from transformers import (
    logging
)
from sentence_transformers import CrossEncoder
import os
file_path = os.path.abspath(__file__)
PROJECT_PATH = os.path.dirname(file_path)

logging.set_verbosity(logging.CRITICAL)



class reranker:
    def __init__(self):
        print("Load reranker model... ", end = '')
        self.model = CrossEncoder(PROJECT_PATH + "/model/rerank_model")
        print("Done")
    
    def rerank(self, query, contexts):
        scores = self.model.predict([(query, context) for context in contexts])
        # Sắp xếp contexts theo scores giảm dần, lưu chỉ số cũ
        indexed_data = list(enumerate(zip(scores, contexts)))  # Gắn index vào dữ liệu
        sorted_data = sorted(indexed_data, key=lambda x: x[1][0], reverse=True)

        # Tách dữ liệu ra: chỉ số cũ, scores, và contexts
        old_indices, sorted_scores_contexts = zip(*sorted_data)
        sorted_scores, sorted_contexts = zip(*sorted_scores_contexts)
        return (sorted_contexts, old_indices, sorted_scores)
