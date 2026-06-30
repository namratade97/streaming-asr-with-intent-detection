import os
import json
from glob import glob

def lowercase_supervisions(base_dir):
    manifest_dir = os.path.join(base_dir, "data", "manifests")
    supervision_files = glob(os.path.join(manifest_dir, "*_supervisions.jsonl"))

    for file_path in supervision_files:
        print(f"Processing {file_path}...")
        tmp_path = file_path + ".tmp"

        with open(file_path, "r", encoding="utf-8") as fin, open(tmp_path, "w", encoding="utf-8") as fout:
            for line in fin:
                obj = json.loads(line)
                if "text" in obj:
                    obj["text"] = obj["text"].lower()
                fout.write(json.dumps(obj) + "\n")

        os.replace(tmp_path, file_path)
        print(f"Done: {file_path}")

if __name__ == "__main__":
    base_folder = "/mnt/gpu-phx/am-team/workspaces/nde/organized_slurp_data_pods_lowercase_without_intent"
    lowercase_supervisions(base_folder)
