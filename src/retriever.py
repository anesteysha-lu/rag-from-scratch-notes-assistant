# Слой поиска: ищет нужные куски по запросу в двух режимах, semantic and tfidf

from .vectorizers import retrieve_tfidf


class Retriever:
    def __init__(self, index, embedder):
        self.index = index
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 5, mode: str = "semantic"):
        if mode == "semantic":
            query_vector = self.embedder.encode(query)
            return self.index.search(query_vector, k=top_k)
        elif mode == "tfidf":
            documents = [it["text"] for it in self.index.items]
            tfidf_results = retrieve_tfidf(query, documents, top_k=top_k)
            return [(score, self.index.items[idx]) for idx, score in tfidf_results]
        else:
            raise ValueError(f"Неизвестный режим: {mode}")
