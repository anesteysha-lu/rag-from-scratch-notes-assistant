# Собирает промпт для модели из вопроса и найденного контекста


def build_prompt(query: str, results: list[tuple]) -> str:
    context = ""
    for score, item in results:
        context += f"[{item['source']}, chunk {item['chunk_id']}]\n{item['text']}\n\n"

    prompt = f"""Ты ассистент. Отвечай ТОЛЬКО на основе контекста ниже.
Если в контексте нет ответа, честно скажи: "В документах нет ответа."

Контекст:
{context}
Вопрос: {query}

Ответ:"""
    return prompt
