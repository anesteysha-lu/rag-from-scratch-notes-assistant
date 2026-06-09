# Измеряет качество поиска (как часто нужный документ попадает в top-k)


def hit_rate_at_k(retriever, eval_set, k, mode="semantic"):
    hits = 0
    for example in eval_set:
        results = retriever.retrieve(example["question"], top_k=k, mode=mode)
        found_ids = [item["chunk_id"] for score, item in results]
        if example["expected_chunk_id"] in found_ids:
            hits += 1
    return hits / len(eval_set)


def show_misses(retriever, eval_set, k, mode):
    for example in eval_set:
        results = retriever.retrieve(example["question"], top_k=k, mode=mode)
        found_ids = [item["chunk_id"] for score, item in results]
        if example["expected_chunk_id"] not in found_ids:
            print(f"[{mode}] промах: «{example['question']}»")
            print(f"   ждали chunk {example['expected_chunk_id']}, нашли {found_ids}")
