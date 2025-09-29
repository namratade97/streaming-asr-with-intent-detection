import json
import argparse
import sys
from collections import defaultdict

# For cosine similarity with TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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
    # Vectorize all sentences
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(sentences)
    # Compute cosine similarity matrix
    sim_matrix = cosine_similarity(tfidf)
    return sim_matrix

def compute_semantic_similarities(sentences, model_name='all-MiniLM-L6-v2'):
    if SentenceTransformer is None:
        print("Error: sentence-transformers library is not installed. Install it with: pip install sentence-transformers")
        sys.exit(1)
    model = SentenceTransformer(model_name)
    embeddings = model.encode(sentences, convert_to_tensor=True)
    # Compute cosine similarity matrix using sentence-transformers util
    sim_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()
    return sim_matrix

def compute_edit_distances(sentences):
    n = len(sentences)
    # We'll compute normalized edit distances: lower value means more similar.
    # Normalize by the max length of the two strings.
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                sim_matrix[i, j] = 1.0  # perfect match
            else:
                ed = edit_distance(sentences[i], sentences[j])
                # Normalize: similarity = 1 - (edit_distance / max(len(s1), len(s2)))
                norm = max(len(sentences[i]), len(sentences[j]))
                sim = 1 - (ed / norm) if norm > 0 else 0
                sim_matrix[i, j] = sim
    return sim_matrix

def main(args):
    records = load_jsonl(args.input)
    if not records:
        print("No records found in input file.")
        return

    # Extract sentences for similarity computation
    sentences = [record["sentence"] for record in records]
    n = len(sentences)

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

    # For each record, get the indices of the top 5 most similar sentences (excluding itself)
    for i, record in enumerate(records):
        # Get similarity scores for record i with every other record
        scores = sim_matrix[i]
        # Exclude self by setting score to -inf (for cosine) or 0 (for our normalized edit similarity)
        if args.metric in ["tfidf", "semantic"]:
            scores[i] = -np.inf
        else:
            scores[i] = -1  # since our similarity is in [0,1]

        # Get indices of top 5 scores (sort descending)
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
        # Add the "closest" field to the record
        record["closest"] = closest

    # Write the new records to the output file
    write_jsonl(records, args.output)
    print(f"Result saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute sentence similarities for JSONL records and append top 5 closest sentences.")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSONL file.")
    parser.add_argument("--output", type=str, required=True, help="Path to output JSONL file.")
    parser.add_argument("--metric", type=str, default="tfidf", choices=["tfidf", "semantic", "edit"],
                        help="Similarity metric to use: 'tfidf', 'semantic', or 'edit'.")
    args = parser.parse_args()
    main(args)
