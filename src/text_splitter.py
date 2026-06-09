# Режет текст на куски (chunks) по словам с перекрытием или по абзацам


def split_by_words(text: str, chunk_size: int, overlap: int) -> list[str]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size должен быть больше overlap")
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk_text = " ".join(words[i:i + chunk_size])
        chunks.append(chunk_text)
        if i + chunk_size >= len(words):
            break
    return chunks


def split_by_paragraph(text: str, chunk_size: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size должен быть больше 0")
    paragraphs = text.split("\n\n")
    chunks = []
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if len(p.split()) <= chunk_size:
            chunks.append(p)
        else:
            sentences = p.split(". ")
            current_chunk_sentences = []
            current_words_count = 0
            for sentence in sentences:
                if not sentence.endswith("."):
                    sentence += "."
                sentence_words = len(sentence.split())
                if sentence_words > chunk_size:
                    raise ValueError(
                        f"Предложение слишком длинное ({sentence_words} слов) "
                        f"для chunk_size={chunk_size}"
                    )
                if current_words_count + sentence_words <= chunk_size:
                    current_chunk_sentences.append(sentence)
                    current_words_count += sentence_words
                else:
                    if current_chunk_sentences:
                        chunks.append(" ".join(current_chunk_sentences))
                    current_chunk_sentences = [sentence]
                    current_words_count = sentence_words
            if current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))
    return chunks
