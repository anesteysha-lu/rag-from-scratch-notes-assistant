# RAG From Scratch — Notes Assistant

Минимальная RAG-система, собранная с нуля, без готовых фреймворков (LangChain / LlamaIndex).
Все основные части — TF-IDF, vector search, chunking, retriever, сборка промпта и оценка
качества — реализованы вручную. Генерация ответа выполняется локальной моделью через Ollama.

Система отвечает на вопрос **только по загруженным документам**, а если ответа в них нет —
честно сообщает об этом.

## Что реализовано

Код находится в `src/`:

| Модуль | Что делает |
|---|---|
| `document_loader.py` | загрузка документов из текстового файла |
| `text_splitter.py` | разбиение текста на чанки (по словам с overlap и по абзацам) |
| `vectorizers.py` | TF-IDF: токенизация, df, idf, tf-idf, поиск по словам |
| `vector_index.py` | косинусная близость, top-k и хранилище `VectorIndex` (вектор + метаданные) |
| `retriever.py` | слой поиска `Retriever` в двух режимах: `semantic` (эмбеддинги) и `tfidf` |
| `prompt_builder.py` | сборка промпта из вопроса и найденного контекста |
| `generator.py` | генерация ответа локальной LLM (Ollama) |
| `evaluation.py` | оценка retrieval: `hit_rate_at_k` и разбор промахов |

## Структура проекта

```
rag-from-scratch-notes-assistant/
  data/
    raw/ml_notes.txt        # исходные документы (по одному на строку)
  notebooks/                # черновики и эксперименты по блокам
  src/                      # чистая реализация (см. таблицу выше)
  main.py                   # интерактивный запуск из терминала
  requirements.txt
  README.md
```

## Требования

- Python 3.9
- Пакеты из `requirements.txt` (numpy, pandas, scikit-learn, jupyter, sentence-transformers, ollama)
- Для генерации ответа — [Ollama](https://ollama.com) с моделью `qwen3:8b`

## Установка

```bash
# 1. создать и активировать виртуальное окружение
python3 -m venv .venv
source .venv/bin/activate

# 2. установить зависимости
pip install -r requirements.txt

# 3. установить Ollama (https://ollama.com/download) и скачать модель
ollama pull qwen3:8b
```

Эмбеддинг-модель `paraphrase-multilingual-MiniLM-L12-v2` скачается автоматически при первом запуске.

## Запуск

Запускать из корня проекта при запущенном Ollama:

```bash
source .venv/bin/activate
python main.py
```

Скрипт загрузит модель и документы, после чего можно задавать вопросы в терминале:

```
Вопрос: как защитить данные от утечки
Думаю...
Ответ: Система предотвращения утечек данных контролирует передачу конфиденциальной информации.

Вопрос: exit
Пока!
```

Для выхода — введите `exit`, `quit` или пустую строку.

## Возможные улучшения

- **Hybrid retrieval** — объединить режимы `semantic` и `tfidf` (например, через
  Reciprocal Rank Fusion), чтобы и точные термины, и смысловые запросы находились
  одинаково хорошо. Сейчас режимы работают по отдельности.
- Расширить набор документов и тест-сет для более надёжной оценки качества.
