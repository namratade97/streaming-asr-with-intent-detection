#!/usr/bin/env python3
import os
import json
from glob import glob

def lowercase_cut_texts(base_dir: str):
    """
    For each slurp_cuts_*.jsonl under base_dir/data/fbank, lowercase all
    supervision.text fields inside the Cut JSON.
    """
    cuts_dir = os.path.join(base_dir, "data", "fbank")
    cut_files = glob(os.path.join(cuts_dir, "slurp_cuts_*.jsonl"))

    for file_path in cut_files:
        print(f"Processing {file_path}...")
        tmp_path = file_path + ".tmp"
        with open(file_path, "r", encoding="utf-8") as fin, \
             open(tmp_path,  "w", encoding="utf-8") as fout:
            for line in fin:
                obj = json.loads(line)
                for sup in obj.get("supervisions", []):
                    if "text" in sup:
                        sup["text"] = sup["text"].lower()
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        os.replace(tmp_path, file_path)
        print(f"Done: {file_path}\n")

if __name__ == "__main__":
    BASE = "/mnt/gpu-phx/am-team/workspaces/nde/organized_slurp_data_pods_lowercase_without_intent"
    lowercase_cut_texts(BASE)
