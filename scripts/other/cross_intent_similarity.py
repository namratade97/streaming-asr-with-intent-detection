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


def compute_tfidf_similarities(sentences):
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(sentences)
    return cosine_similarity(tfidf)


def compute_semantic_similarities(sentences, model_name='all-MiniLM-L6-v2'):
    if SentenceTransformer is None:
        print("Error: install sentence-transformers: pip install sentence-transformers")
        sys.exit(1)
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences, convert_to_tensor=True)
    return util.cos_sim(embeddings, embeddings).cpu().numpy()


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
                sim_matrix[i, j] = 1 - (ed / norm) if norm > 0 else 0
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

        # inside similarity
        inside_sum, inside_count = 0.0, 0
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                inside_sum += sim_matrix[indices[i], indices[j]]
                inside_count += 1
        inside_avg = inside_sum / inside_count if inside_count else 0

        # outside similarity (all others)
        outside_sum, outside_count = 0.0, 0
        all_idx = set(range(len(records)))
        non_group = list(all_idx - set(indices))
        for i in indices:
            for j in non_group:
                outside_sum += sim_matrix[i, j]
                outside_count += 1
        outside_avg = outside_sum / outside_count if outside_count else 0

        ratio = inside_avg / outside_avg if outside_avg > 0 else None
        intent_ratios[intent] = {"inside": inside_avg, "outside": outside_avg, "ratio": ratio}

    return intent_ratios


def main():
    parser = argparse.ArgumentParser(
        description="Compute sentence similarities and intent ratios, with optional cross-intent check."
    )
    parser.add_argument("--input", type=str, required=True, help="Path to input JSONL file.")
    parser.add_argument(
        "--metric", type=str, default="tfidf", choices=["tfidf", "semantic", "edit"],
        help="Similarity metric to use: tfidf, semantic, or edit."
    )
    parser.add_argument(
        "--cross_intents", type=str, default="",
        help="Optional: two intents A,B to compute cross-intent similarity and ratio."
    )
    args = parser.parse_args()

    records = load_jsonl(args.input)
    if not records:
        print("No records found in input file.")
        return

    sentences = [r["sentence"] for r in records]
    if args.metric == "tfidf":
        print("Computing TF-IDF similarities...")
        sim_matrix = compute_tfidf_similarities(sentences)
    elif args.metric == "semantic":
        print("Computing semantic similarities...")
        sim_matrix = compute_semantic_similarities(sentences)
    else:
        print("Computing edit-distance similarities...")
        sim_matrix = compute_edit_distances(sentences)

    # Top-5 closest
    for i, rec in enumerate(records):
        scores = sim_matrix[i].copy()
        scores[i] = -np.inf if args.metric in ["tfidf", "semantic"] else -1
        top5 = np.argsort(scores)[-5:][::-1]
        rec["closest"] = [
            {"sentence": records[j]["sentence"], "intent": records[j]["intent"],
             "similarity": float(scores[j])}
            for j in top5
        ]

    # Compute intent ratios
    intent_ratios = compute_intent_similarity_ratios(records, sim_matrix)
    freqs = defaultdict(int)
    for r in records:
        freqs[r.get("intent", "<null>")] += 1

    # Print sorted by frequency
    sorted_intents = sorted(freqs, key=lambda x: freqs[x], reverse=True)
    print("\nIntent similarity ratios (sorted by freq):")
    for intent in sorted_intents:
        stats = intent_ratios[intent]
        n = freqs[intent]
        if stats["ratio"] is None:
            print(f"  {intent} (n={n}): insufficient data")
        else:
            print(f"  {intent} (n={n}): inside={stats['inside']:.3f},"
                  f" outside={stats['outside']:.3f}, ratio={stats['ratio']:.3f}")

    # Cross-intent if requested
    if args.cross_intents:
        A, B = [x.strip() for x in args.cross_intents.split(',')]
        idxsA = [i for i,r in enumerate(records) if r.get("intent")==A]
        idxsB = [i for i,r in enumerate(records) if r.get("intent")==B]
        if not idxsA or not idxsB:
            print(f"One of the intents '{A}' or '{B}' has no examples.")
        else:
            cross_sum = sum(sim_matrix[i, j] for i in idxsA for j in idxsB)
            cross_avg = cross_sum / (len(idxsA)*len(idxsB))
            inside = intent_ratios[A]["inside"]
            if inside is None:
                print(f"Not enough data for inside-average of {A}.")
            else:
                ratio = inside / cross_avg if cross_avg>0 else None
                print(f"\nCross-avg({A},{B}) = {cross_avg:.3f}")
                print(f"Inside-avg({A}) = {inside:.3f}")
                print(f"Inside/Cross ratio = {ratio:.3f}")

if __name__ == "__main__":
    main()
