import json
import os

# Define paths
base_cuts_path = '/disk1/polaris_intent_detection/organized_slurp_data/data/fbank'
transcripts_path = '/disk1/polaris_intent_detection/organized_slurp_data/original_transcripts'

def load_transcripts(split):
    transcript_file = os.path.join(transcripts_path, split, f'{split}.jsonl')
    transcripts = {}
    with open(transcript_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            for rec in data["recordings"]:
                transcripts[rec["file"]] = {
                    "text": data["sentence"],
                    "intent": data["intent"]
                }
    return transcripts

def update_cuts_file(split):
    transcripts = load_transcripts(split)

    cuts_file = os.path.join(base_cuts_path, f'slurp_cuts_{split}.jsonl')
    updated_cuts = []

    with open(cuts_file, 'r') as f:
        for line in f:
            cut = json.loads(line)
            recording_id = cut["recording"]["id"]
            supervision = cut["supervisions"][0]  

            recording_flac = f"{recording_id}.flac"
            if recording_flac in transcripts:
                transcript = transcripts[recording_flac]
                text = transcript["text"].upper()  
                intent = transcript["intent"]
                supervision["text"] = f"{text} <{intent}>"

            updated_cuts.append(cut)

    updated_cuts_file = os.path.join(base_cuts_path, f'slurp_cuts_{split}_updated.jsonl')
    with open(updated_cuts_file, 'w') as f:
        for cut in updated_cuts:
            f.write(json.dumps(cut) + '\n')

    print(f"Updated cuts saved to {updated_cuts_file}")

if __name__ == "__main__":
    for split in ['test', 'train', 'val']:
        print(f"Processing split: {split}")
        update_cuts_file(split)
        print(f"Finished processing {split} split.")
