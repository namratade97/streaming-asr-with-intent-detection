#!/usr/bin/env python3
import os
import gzip
import json
import glob

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION: point this at your existing manifests dir
# ──────────────────────────────────────────────────────────────────────────────

# MANIFEST_DIR = "/mnt/gpu-phx/am-team/workspaces/nde/LibriSpeech_nde/data/manifests"
MANIFEST_DIR = "/mnt/gpu-phx/am-team/workspaces/nde/LibriSpeech_nde/data/fbank"

# ──────────────────────────────────────────────────────────────────────────────
# PROCESS
# ──────────────────────────────────────────────────────────────────────────────

# for path in glob.glob(os.path.join(MANIFEST_DIR, "librispeech_supervisions_*.jsonl.gz")):
for path in glob.glob(os.path.join(MANIFEST_DIR, "librispeech_cuts_*.jsonl.gz")):
    tmp_path = path + ".tmp"
    print(f"Lower-casing {os.path.basename(path)} → writing to {os.path.basename(tmp_path)}")

    with gzip.open(path, "rt", encoding="utf-8") as fin, \
        gzip.open(tmp_path, "wt", encoding="utf-8") as fout:
        for line in fin:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            # For each supervision in the cut, lowercase its text
            if "supervisions" in obj and isinstance(obj["supervisions"], list):
                for sup in obj["supervisions"]:
                    if "text" in sup and isinstance(sup["text"], str):
                        sup["text"] = sup["text"].lower()

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # with gzip.open(path, "rt", encoding="utf-8") as fin, \
    #      gzip.open(tmp_path, "wt", encoding="utf-8") as fout:
    #     for line in fin:
    #         try:
    #             obj = json.loads(line)
    #         except json.JSONDecodeError:
    #             # skip malformed lines, if any
    #             continue
    #         if "text" in obj and isinstance(obj["text"], str):
    #             obj["text"] = obj["text"].lower()
    #         fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # Replace original with lower-cased version
    os.replace(tmp_path, path)
    print(f"  done.. {os.path.basename(path)} updated")

print("\n All supervision manifests lower-cased in-place.")
