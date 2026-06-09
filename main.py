# Запуск RAG из терминала: вводишь вопрос — получаешь ответ по документам.

from sentence_transformers import SentenceTransformer

from src.document_loader import load_documents
from src.vector_index import VectorIndex
from src.retriever import Retriever
from src.prompt_builder import build_prompt
from src.generator import Generator

DATA_PATH = "data/raw/ml_notes.txt"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


def build_rag():
    documents = load_documents(DATA_PATH)
    model = SentenceTransformer(EMBED_MODEL)
    doc_matrix = model.encode(documents)
    items = [{"text": d, "source": "ml_notes.txt", "chunk_id": i} for i, d in enumerate(documents)]
    index = VectorIndex()
    index.add(doc_matrix, items)
    retriever = Retriever(index, model)
    generator = Generator()
    return retriever, generator


def main():
    print("Загружаю модель и документы...")
    retriever, generator = build_rag()
    print("Готово. Задай вопрос (или 'exit' для выхода).\n")
    while True:
        query = input("Вопрос: ").strip()
        if query.lower() in {"exit", "quit", ""}:
            print("Пока!")
            break
        print("Думаю...", flush=True)
        try:
            results = retriever.retrieve(query, top_k=5, mode="semantic")
            answer = generator.answer(build_prompt(query, results))
            print("\nОтвет:", answer, "\n")
        except Exception as e:
            print(f"\nЧто-то пошло не так ({type(e).__name__}), попробуй ещё раз.\n")


if __name__ == "__main__":
    main()
