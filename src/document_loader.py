# Загружает документы из текстового файла

from pathlib import Path


def load_documents(path: str) -> list[str]:
    text = Path(path).read_text(encoding="utf-8")
    documents = [line.strip() for line in text.splitlines() if line.strip()]
    return documents
