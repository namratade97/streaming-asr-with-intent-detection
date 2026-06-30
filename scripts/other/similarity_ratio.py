import json
import argparse
import sys
from collections import defaultdict
import numpy as np

# For cosine similarity with TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# For edit distance similarity (we use nltk.edit_distance)
import nltk
nltk.download('punkt', quiet=True)
from nltk.metrics.distance import edit_distance

# For semantic similarity using sentence-transformers
try:
    from sentence_transformers import SentenceTransformer, util
except ImportError:
    SentenceTransformer = None  # will handle later

def load_jsonl(input_file):
    records = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def write_jsonl(records, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def compute_tfidf_similarities(sentences):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(sentences)
    sim_matrix = cosine_similarity(tfidf)
    return sim_matrix

def compute_semantic_similarities(sentences, model_name='all-MiniLM-L6-v2'):
    if SentenceTransformer is None:
        print("Error: sentence-transformers library is not installed. Install it with: pip install sentence-transformers")
        sys.exit(1)
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences, convert_to_tensor=True)
    sim_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()
    return sim_matrix

def compute_edit_distances(sentences):
    n = len(sentences)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                sim_matrix[i, j] = 1.0
            else:
                ed = edit_distance(sentences[i], sentences[j])
                norm = max(len(sentences[i]), len(sentences[j]))
                sim = 1 - (ed / norm) if norm > 0 else 0
                sim_matrix[i, j] = sim
    return sim_matrix

def compute_intent_similarity_ratios(records, sim_matrix):
    intent_groups = defaultdict(list)
    for idx, record in enumerate(records):
        intent = record.get("intent", "<null>")
        intent_groups[intent].append(idx)

    intent_ratios = {}
    for intent, indices in intent_groups.items():
        if len(indices) < 2:
            intent_ratios[intent] = {"inside": None, "outside": None, "ratio": None}
            continue

        inside_sum = 0.0
        inside_count = 0
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                inside_sum += sim_matrix[indices[i], indices[j]]
                inside_count += 1
        inside_avg = inside_sum / inside_count if inside_count > 0 else 0

        outside_sum = 0.0
        outside_count = 0
        all_indices = set(range(len(records)))
        group_set = set(indices)
        non_group = list(all_indices - group_set)
        for idx in indices:
            for j in non_group:
                outside_sum += sim_matrix[idx, j]
                outside_count += 1
        outside_avg = outside_sum / outside_count if outside_count > 0 else 0

        ratio = inside_avg / outside_avg if outside_avg > 0 else None
        intent_ratios[intent] = {"inside": inside_avg, "outside": outside_avg, "ratio": ratio}

    return intent_ratios

def main(args):
    records = load_jsonl(args.input)
    if not records:
        print("No records found in input file.")
        return

    sentences = [record["sentence"] for record in records]

    if args.metric == "tfidf":
        print("Computing TF-IDF cosine similarities...")
        sim_matrix = compute_tfidf_similarities(sentences)
    elif args.metric == "semantic":
        print("Computing semantic similarities using SentenceTransformer...")
        sim_matrix = compute_semantic_similarities(sentences)
    elif args.metric == "edit":
        print("Computing edit-distance based similarities...")
        sim_matrix = compute_edit_distances(sentences)
    else:
        print("Unknown similarity metric. Choose from: tfidf, semantic, edit")
        return

    for i, record in enumerate(records):
        scores = sim_matrix[i].copy()
        scores[i] = -np.inf if args.metric in ["tfidf", "semantic"] else -1
        top_indices = np.argsort(scores)[-5:][::-1]
        closest = []
        for idx in top_indices:
            similar_entry = {
                "sentence": records[idx]["sentence"],
                "intent": records[idx]["intent"],
                "scenario": records[idx]["scenario"],
                "recordings": records[idx]["recordings"],
                "similarity": float(scores[idx])
            }
            closest.append(similar_entry)
        record["closest"] = closest

    intent_ratios = compute_intent_similarity_ratios(records, sim_matrix)
    sorted_ratios = sorted(
        intent_ratios.items(),
        key=lambda x: float('inf') if x[1]['ratio'] is None else x[1]['ratio']
    )

    intent_freqs = defaultdict(int)
    for record in records:
        intent_freqs[record.get("intent", "<null>")] += 1

    sorted_intents = sorted(intent_ratios.keys(), key=lambda i: intent_freqs[i], reverse=True)

    print("\nIntent similarity ratios (inside average similarity / outside average similarity), sorted by intent frequency:")
    for intent in sorted_intents:
        stats = intent_ratios[intent]
        count = intent_freqs[intent]
        if stats["ratio"] is None:
            print(f"  Intent '{intent}' (n={count}): Not enough data to compute ratio.")
        else:
            print(f"  Intent '{intent}' (n={count}): inside avg = {stats['inside']:.3f}, outside avg = {stats['outside']:.3f}, ratio = {stats['ratio']:.3f}")

    # Print intents with ratio under the given threshold (args.threshold)
    threshold = args.threshold
    total_queries = len(records)
    intents_under_threshold = {intent: stats for intent, stats in intent_ratios.items() if stats["ratio"] is not None and stats["ratio"] < threshold}
    
    print(f"\nIntents with a similarity ratio under {threshold}:")
    total_under_threshold = 0
    for intent, stats in intents_under_threshold.items():
        count = intent_freqs[intent]
        print(f"  Intent '{intent}' (n={count}): ratio = {stats['ratio']:.3f}")
        total_under_threshold += count

    print(f"\nTotal count of intents with a similarity ratio under {threshold}: {total_under_threshold}")
    ratio_under_threshold = (total_under_threshold / total_queries) * 100
    print(f"Percentage of queries with a similarity ratio under {args.threshold}: {ratio_under_threshold:.2f}%")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute sentence similarities and append top 5 closest sentences to JSONL records. Also compute intent-level similarity ratios.")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSONL file.")
    parser.add_argument("--metric", type=str, default="tfidf", choices=["tfidf", "semantic", "edit"],
                        help="Similarity metric to use: 'tfidf', 'semantic', or 'edit'.")
    parser.add_argument("--threshold", type=float, default=3, help="Threshold ratio for identifying intents with low similarity.")
    args = parser.parse_args()
    main(args)
