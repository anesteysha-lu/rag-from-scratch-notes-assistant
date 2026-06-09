# TF-IDF: поиск документов по совпадению слов (без учёта смысла).

import re
import math
from collections import Counter


def tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^а-яa-z0-9\- ]", " ", text)
    return text.split()


def compute_df(tokenized_documents: list[list[str]]) -> dict[str, int]:
    df = Counter()
    for tokens in tokenized_documents:
        for term in set(tokens):
            df[term] += 1
    return dict(df)


def compute_idf(df: dict[str, int], total_doc: int) -> dict[str, float]:
    return {term: math.log(total_doc / df[term]) for term in df}


def compute_tfidf(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    tf = Counter(tokens)
    return {term: tf[term] * idf[term] for term in tf if term in idf}


def dot_product(vec1: dict[str, float], vec2: dict[str, float]) -> float:
    return sum(weight * vec2.get(term, 0.0) for term, weight in vec1.items())


def retrieve_tfidf(query: str, documents: list[str], top_k: int = 3) -> list[tuple[int, float]]:
    tokenized_documents = [tokenize(doc) for doc in documents]
    df = compute_df(tokenized_documents)
    idf = compute_idf(df, len(documents))
    doc_vectors = [compute_tfidf(tokens, idf) for tokens in tokenized_documents]
    query_vector = compute_tfidf(tokenize(query), idf)
    scores = [(i, dot_product(query_vector, dv)) for i, dv in enumerate(doc_vectors)]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]
