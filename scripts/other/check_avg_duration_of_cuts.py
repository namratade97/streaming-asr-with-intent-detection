import gzip
import json
import argparse

def open_jsonl_file(path):
    if path.endswith(".gz"):
        return gzip.open(path, 'rt', encoding='utf-8')
    else:
        return open(path, 'r', encoding='utf-8')


def compute_avg_duration(path):
    total_duration = 0.0
    num_entries = 0

    with open_jsonl_file(path) as f:
        for line in f:
            if line.strip():
                try:
                    entry = json.loads(line)
                    duration = entry.get("duration", 0.0)
                    total_duration += duration
                    num_entries += 1
                except json.JSONDecodeError as e:
                    print(f"Skipping malformed line: {e}")

    if num_entries == 0:
        print("No valid entries found.")
        return

    avg_duration = total_duration / num_entries
    print(f"Total entries: {num_entries}")
    print(f"Average duration: {avg_duration:.3f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute average duration of entries in a cuts jsonl.gz file.")
    parser.add_argument("input_path", type=str, help="Path to cuts_*.jsonl.gz file")
    args = parser.parse_args()

    compute_avg_duration(args.input_path)
