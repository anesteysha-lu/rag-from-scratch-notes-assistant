# Отправляет промпт локальной модели (Ollama) и возвращает ответ

import ollama


class Generator:
    def __init__(self, model: str = "qwen3:8b"):
        self.model = model

    def answer(self, prompt: str) -> str:
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]
