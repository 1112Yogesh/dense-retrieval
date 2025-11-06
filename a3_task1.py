import time
from typing import List, Dict, Tuple

import json
import os
from pyserini.search.lucene import LuceneSearcher

import torch


FINAL_TOP_K = 10
MAX_LENGTH = 512

BERT_MODEL_NAME = 'amberoad/bert-multilingual-passage-reranking-msmarco'
BEST_K = 110

DEVICE = torch.device("cpu")
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")


def setup_bert_reranker():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)
    # The model must be loaded for sequence classification (binary relevance)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_NAME).to(DEVICE)
    model.eval()
    
    return model, tokenizer

def get_queries(query_path: str) -> list[tuple[str, str]]:
    queries: list[tuple[str, str]] = []
    with open(query_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            queries.append((data['query_id'], data['text']))
    return queries

def bm25_retrieval(queries: list[tuple[str, str]]) -> list[dict]:
    try:
        searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')
        
        if not searcher:
            print("ERROR: Pyserini searcher failed to initialize. Check Java configuration again.")
            return

    except Exception as e:
        print(f"\n--- FATAL INITIALIZATION ERROR ---")
        print(f"Pyserini failed to initialize (likely Java/JVM issue): {e}")
        print("Please ensure your JAVA_HOME environment variable correctly points to JDK 21.")
        return

    query_results = []

    for q_id, q_text in queries:
        hits = searcher.search(q_text, k=FINAL_TOP_K)
        query_results.append({
            'query_id': q_id,
            'query': q_text,
            'hits': [{'docid': hit.docid, 'score': hit.score} for hit in hits]
        })

    return query_results

def save_results_to_trec(bm25_output_file: str, results: list[dict]) -> None:
    if (not os.path.exists(os.path.dirname(bm25_output_file)) and bm25_output_file.find('/') != -1):
        os.makedirs(os.path.dirname(bm25_output_file), exist_ok=True)
        
    with open(bm25_output_file, 'w') as f:
        for result in results:
            q_id = result['query_id']
            hits = result['hits']
            for rank, hit in enumerate(hits):
                f.write(f"{q_id} {hit['docid']} {rank + 1} {hit['score']}\n")

def bm25_with_bert_retrieval(queries: list[tuple[str, str]], top_k: int) -> list[dict]:
    model, tokenizer = setup_bert_reranker()

    try:
        searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')
        if not searcher:
            print("ERROR: Pyserini searcher failed to initialize. Check Java configuration again.")
            return

    except Exception as e:
        print(f"\n--- FATAL INITIALIZATION ERROR ---")
        print(f"Pyserini failed to initialize (likely Java/JVM issue): {e}")
        print("Please ensure your JAVA_HOME environment variable correctly points to JDK 21.")
        return
    
    query_results = []

    for q_id, q_text in queries:
        hits = searcher.search(q_text, k=top_k)

        pairs_to_rerank = []
        doc_ids = []

        for hit in hits:
            try:
                doc_json = searcher.doc(hit.docid).raw()
                doc_text = json.loads(doc_json)['contents']
            except:
                continue

            pairs_to_rerank.append((q_text, doc_text))
            doc_ids.append(hit.docid)

        if not pairs_to_rerank:
            print(f"WARNING: No valid document pairs for query ID {q_id}. Skipping.")
            query_results.append({
                'query_id': q_id,
                'query': q_text,
                'hits': []
            })
            continue

        inputs = tokenizer(
            pairs_to_rerank,
            padding=True,
            truncation='only_second',
            max_length=MAX_LENGTH,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs)

        relevance_scores = outputs.logits[:, 1].cpu().numpy()

        reranked_results = list(zip(doc_ids, relevance_scores))

        reranked_results.sort(key=lambda x: x[1], reverse=True)

        reranked_results = reranked_results[:FINAL_TOP_K]

        query_results.append({
            'query_id': q_id,
            'query': q_text,
            'hits': [{'docid': doc_id, 'score': score} for doc_id, score in reranked_results]
        })

    return query_results

def task1_rerank (query_path: str, bm25_output_file: str, reranked_output_file: str, k: int=BEST_K) -> None:
    queries = get_queries(query_path)

    # bm25_results = bm25_retrieval(queries)

    # save_results_to_trec(bm25_output_file, bm25_results)

    bm25_with_bert_results = bm25_with_bert_retrieval(queries, top_k=k)

    save_results_to_trec(reranked_output_file, bm25_with_bert_results)

if __name__ == "__main__":
    start_time = time.time()
    QUERY_PATH = 'queries.json'
    BM25_OUTPUT_FILE = 'output/task1_bm25_results.txt'
    RERANKED_OUTPUT_FILE = 'output/task1_bm25_bert_reranked_results.txt'

    task1_rerank(
        query_path=QUERY_PATH,
        bm25_output_file=BM25_OUTPUT_FILE,
        reranked_output_file=RERANKED_OUTPUT_FILE,
        k=BEST_K
    )
    end_time = time.time()
    print(f"Total Execution Time: {end_time - start_time} seconds")