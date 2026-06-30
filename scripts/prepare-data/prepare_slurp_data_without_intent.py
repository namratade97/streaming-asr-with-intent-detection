import json
import re
import shutil
from pathlib import Path

base_dir = Path("/mnt/gpu-phx/am-team/workspaces/nde/organized_slurp_data_pods_uppercase_without_intent/data")

INTENT_AT_END = re.compile(r"\s*<[^>\s]+>\s*$")

def drop_trailing_intent(text: str) -> str:
    if not text:
        return text
    return INTENT_AT_END.sub("", text).strip()

def process_cuts_file(infile: Path):
    tmpfile = infile.with_suffix(infile.suffix + ".tmp")
    removed, total = 0, 0
    with open(infile, "r") as fin, open(tmpfile, "w") as fout:
        for line in fin:
            obj = json.loads(line)
            for sup in obj.get("supervisions", []):
                if "text" in sup and isinstance(sup["text"], str):
                    total += 1
                    new_text = drop_trailing_intent(sup["text"])
                    if new_text != sup["text"]:
                        removed += 1
                        sup["text"] = new_text
            fout.write(json.dumps(obj) + "\n")
    # Backup then replace
    shutil.copy2(infile, infile.with_suffix(infile.suffix + ".bak"))
    shutil.move(tmpfile, infile)
    print(f"[cuts] {infile.name}: stripped intent from {removed}/{total} supervisions")

def process_supervisions_file(infile: Path):
    tmpfile = infile.with_suffix(infile.suffix + ".tmp")
    removed, total = 0, 0
    with open(infile, "r") as fin, open(tmpfile, "w") as fout:
        for line in fin:
            obj = json.loads(line)
            if "text" in obj and isinstance(obj["text"], str):
                total += 1
                new_text = drop_trailing_intent(obj["text"])
                if new_text != obj["text"]:
                    removed += 1
                obj["text"] = new_text
            fout.write(json.dumps(obj) + "\n")
    # Backup then replace
    shutil.copy2(infile, infile.with_suffix(infile.suffix + ".bak"))
    shutil.move(tmpfile, infile)
    print(f"[supervisions] {infile.name}: stripped intent from {removed}/{total} rows")


# 1) Cuts (fbank/slurp_cuts_*.jsonl)
cuts_dir = base_dir / "fbank"
for fname in sorted(cuts_dir.glob("slurp_cuts_*.jsonl")):
    print("Processing cuts:", fname)
    process_cuts_file(fname)

# 2) Supervisions (manifests/*_supervisions.jsonl)
manifests_dir = base_dir / "manifests"
for fname in sorted(manifests_dir.glob("*_supervisions.jsonl")):
    print("Processing supervisions:", fname)
    process_supervisions_file(fname)
