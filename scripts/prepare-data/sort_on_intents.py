import json

input_file = "/disk1/nde/polaris_intent_detection/organized_slurp_data_pods/original_transcripts/val/val.jsonl"
output_file = "rearrange_val.jsonl"

def load_and_simplify_jsonl(path):
    simplified = []
    with open(path, "r") as f:
        for line in f:
            data = json.loads(line)
            simplified.append({
                "sentence": data["sentence"],
                "intent": data["intent"],
                "scenario": data["scenario"],
                "recordings": data["recordings"]
            })
    return simplified

def sort_records(records):
    return sorted(
        records,
        key=lambda x: (
            x["intent"],
            len(x["sentence"].split()),
            x["sentence"].lower()
        )
    )

def write_jsonl(records, path):
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

if __name__ == "__main__":
    records = load_and_simplify_jsonl(input_file)
    sorted_records = sort_records(records)
    write_jsonl(sorted_records, output_file)
    print(f"Sorted file written to: {output_file}")
