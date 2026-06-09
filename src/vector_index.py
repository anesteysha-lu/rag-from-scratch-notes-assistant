# Поиск по смыслу, косинусная близость векторов и хранилище VectorIndex.

import numpy as np


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms


def cosine_scores(query_vector: np.ndarray, doc_vectors: np.ndarray) -> np.ndarray:
    query_norm = np.linalg.norm(query_vector)
    if query_norm == 0:
        return np.zeros(doc_vectors.shape[0])
    normalized_query = query_vector / query_norm
    normalized_docs = normalize_rows(doc_vectors)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return normalized_docs @ normalized_query


def top_k(scores: np.ndarray, k: int) -> list[int]:
    return np.argsort(scores)[::-1][:k].tolist()


class VectorIndex:
    def __init__(self):
        self.vectors = None
        self.items = []

    def add(self, vectors, items):
        if self.vectors is None:
            self.vectors = vectors
        else:
            self.vectors = np.vstack([self.vectors, vectors])
        self.items.extend(items)

    def search(self, query_vector, k=5, min_score=None):
        scores = cosine_scores(query_vector, self.vectors)
        best = top_k(scores, k)
        results = []
        for i in best:
            if min_score is None or scores[i] >= min_score:
                results.append((scores[i], self.items[i]))
        return results
