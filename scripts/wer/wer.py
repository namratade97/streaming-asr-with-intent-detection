import re
import sys
import ast

def parse_input_file(file_path):
    """
    Parses a file with ref=[...] / hyp=[...] lines.
    Safely handles:
        - Single-word lists
        - Apostrophes in words
        - <intent> tokens
        - Lines starting with #
    Returns list of (ref, hyp) tuples.
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.rstrip() for line in f if line.strip() and not line.strip().startswith("#")]

    for i in range(0, len(lines), 2):
        ref_line = lines[i].strip()
        hyp_line = lines[i+1].strip() if i+1 < len(lines) else ""

        # extract everything after 'ref=' and 'hyp='
        ref_match = re.search(r"ref\s*=(.*)", ref_line)
        hyp_match = re.search(r"hyp\s*=(.*)", hyp_line)

        if not ref_match or not hyp_match:
            print(f"[Line {i+1}-{i+2}] Skipping pair (no ref/hyp match)")
            continue

        try:
            ref = ast.literal_eval(ref_match.group(1))
            if isinstance(ref, str):
                ref = [ref]
        except Exception as e:
            print(f"[Line {i+1}] Failed to parse ref: {ref_match.group(1)!r} Error: {e}")
            continue

        try:
            hyp = ast.literal_eval(hyp_match.group(1))
            if isinstance(hyp, str):
                hyp = [hyp]
        except Exception as e:
            print(f"[Line {i+2}] Failed to parse hyp: {hyp_match.group(1)!r} Error: {e}")
            continue

        # remove <intent> tokens or any token like ▁<...>
        ref = [w.strip() for w in ref if not re.match(r"^▁?<.*>$", w)]
        hyp = [w.strip() for w in hyp if not re.match(r"^▁?<.*>$", w)]

        data.append((ref, hyp))

    return data




def calculate_wer(ref, hyp):
    """
    Calculate Word Error Rate (WER) using dynamic programming.
    Returns WER, substitutions, deletions, insertions.
    """
    n = len(ref)
    m = len(hyp)
    
    # initialize DP table
    dp = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(n + 1):
        dp[i][0] = i  # Deletion cost
    for j in range(m + 1):
        dp[0][j] = j  # Insertion cost
    
    # fill DP table
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # No cost if words match
            else:
                dp[i][j] = min(
                    dp[i - 1][j - 1] + 1,  # Substitution
                    dp[i - 1][j] + 1,      # Deletion
                    dp[i][j - 1] + 1       # Insertion
                )
    
    # backtrack to calculate substitutions, deletions, and insertions
    i, j = n, m
    substitutions, deletions, insertions = 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            deletions += 1
            i -= 1
        else:
            insertions += 1
            j -= 1
    
    wer = (substitutions + deletions + insertions) / len(ref)
    return wer, substitutions, deletions, insertions





def process_file_and_calculate_wer(file_path, show_full=False):
    data = parse_input_file(file_path)

    total_ref_words = total_substitutions = total_deletions = total_insertions = 0
    max_wer = -1.0
    max_wer_examples = []
    num_pairs = 0

    for ref, hyp in data:
        num_pairs += 1
        wer, substitutions, deletions, insertions = calculate_wer(ref, hyp)
        total_ref_words += len(ref)
        total_substitutions += substitutions
        total_deletions += deletions
        total_insertions += insertions

        if wer > max_wer:
            max_wer = wer
            max_wer_examples = [(ref, hyp)]
        elif wer == max_wer:
            max_wer_examples.append((ref, hyp))

        print("#########")
        print(f"Ref: {ref}")
        print(f"Hyp: {hyp}")
        print(f"WER: {wer:.2f} (Substitutions: {substitutions}, Deletions: {deletions}, Insertions: {insertions})\n")

    overall_wer = (total_substitutions + total_deletions + total_insertions) / total_ref_words
    print(f"Total pairs processed: {num_pairs}")
    print(f"Overall WER: {overall_wer:.4f}")
    print(f"Highest WER: {max_wer:.2f}")
    print(f"Number of sentences with highest WER: {len(max_wer_examples)}")

    if show_full:
        print("\nAll examples with highest WER:")
        for ref, hyp in max_wer_examples:
            print(f"Ref: {ref}")
            print(f"Hyp: {hyp}\n")
    else:
        # just show one example
        ref, hyp = max_wer_examples[0]
        print("\nExample with highest WER:")
        print(f"Ref: {ref}")
        print(f"Hyp: {hyp}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python wer.py <path_to_file> [-full]")
        sys.exit(1)

    input_file = sys.argv[1]
    show_full = "-full" in sys.argv
    process_file_and_calculate_wer(input_file, show_full)
