from FlagEmbedding import FlagLLMReranker
class reranker:
    def __init__(self, RR_PATH = r'D:\Huan\Project\KHDL\reranker'):
        print("Load reranker model... ", end = '')
        self.reranker = FlagLLMReranker(RR_PATH, use_fp16=True)
        print("Done")
    def rerank(self, query, contexts, normalize = False):
        reranker = self.reranker
        scores = reranker.compute_score([[query, context] for context in contexts], max_length=1000, normalize = normalize)
        
        # Sắp xếp contexts theo scores giảm dần, lưu chỉ số cũ
        indexed_data = list(enumerate(zip(scores, contexts)))  # Gắn index vào dữ liệu
        sorted_data = sorted(indexed_data, key=lambda x: x[1][0], reverse=True)

        # Tách dữ liệu ra: chỉ số cũ, scores, và contexts
        old_indices, sorted_scores_contexts = zip(*sorted_data)
        sorted_scores, sorted_contexts = zip(*sorted_scores_contexts)
        return (sorted_contexts, old_indices, sorted_scores)
