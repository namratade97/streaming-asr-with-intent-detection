################################# manifests ####################################

# import os
# import json
# from lhotse import Recording, RecordingSet, SupervisionSegment, SupervisionSet

# # Define your dataset paths for train, val, and test
# datasets = {
#     "train": {
#         "audio_dir": "/disk1/polaris_intent_detection/organized_slurp_data/audio/train",
#         "transcript_dir": "/disk1/polaris_intent_detection/organized_slurp_data/original_transcripts/train/train.jsonl",
#         "output_dir": "/disk1/polaris_intent_detection/organized_slurp_data/data/manifests"
#     },
#     "val": {
#         "audio_dir": "/disk1/polaris_intent_detection/organized_slurp_data/audio/val",
#         "transcript_dir": "/disk1/polaris_intent_detection/organized_slurp_data/original_transcripts/val/val.jsonl",
#         "output_dir": "/disk1/polaris_intent_detection/organized_slurp_data/data/manifests"
#     },
#     "test": {
#         "audio_dir": "/disk1/polaris_intent_detection/organized_slurp_data/audio/test",
#         "transcript_dir": "/disk1/polaris_intent_detection/organized_slurp_data/original_transcripts/test/test.jsonl",
#         "output_dir": "/disk1/polaris_intent_detection/organized_slurp_data/data/manifests"
#     }
# }

# # Function to create manifests for all splits
# def create_manifests():
#     for split, paths in datasets.items():
#         print(f"Processing {split} split...")  # Indicate the split being processed
#         audio_dir = paths['audio_dir']
#         transcript_path = paths['transcript_dir']
#         output_dir = paths['output_dir']

#         # Initialize empty lists to hold recordings and supervisions
#         recordings = []
#         supervisions = []

#         # Gather all audio files
#         audio_files = [f for f in os.listdir(audio_dir) if f.endswith(".flac")]
#         total_audio_files = len(audio_files)

#         # Print total audio files for the split
#         print(f"Total audio files found: {total_audio_files}")

#         # Read the transcript file into memory for faster access
#         transcripts = {}
#         if os.path.exists(transcript_path):
#             with open(transcript_path, 'r') as f:
#                 for line in f:
#                     entry = json.loads(line)
#                     for recording_entry in entry.get('recordings', []):
#                         file_id = os.path.splitext(recording_entry['file'])[0]  # Get base file id
#                         transcripts[file_id] = entry.get('sentence', '').strip()

#         # Loop through the audio directory
#         for index, file in enumerate(audio_files):
#             audio_path = os.path.join(audio_dir, file)
#             audio_id = os.path.splitext(file)[0]
#             print(f"Processing audio file: {audio_path}")  # Indicate the audio file being processed

#             # Create a Recording object for each audio file
#             recording = Recording.from_file(audio_path, recording_id=audio_id)
#             recordings.append(recording)

#             # Find the corresponding transcript from the preloaded transcript file
#             full_transcript = transcripts.get(audio_id, None)

#             if full_transcript:
#                 # If the full transcript exists, assign it to the supervision segment
#                 transcript = full_transcript
#             else:
#                 # If no full transcript is found, check for the partial audio
#                 base_audio_id = audio_id.split('-')[0]  # Remove any suffix like '-headset'
#                 transcript = transcripts.get(base_audio_id, None)

#                 if not transcript:
#                     print(f"Warning: No transcript found for {audio_id} or its base audio {base_audio_id}.")
#                     continue  # Skip this file if no transcript is found

#             # Create a SupervisionSegment for each transcript
#             supervision = SupervisionSegment(
#                 id=audio_id,
#                 recording_id=audio_id,
#                 start=0.0,
#                 duration=recording.duration,
#                 text=transcript
#             )
#             supervisions.append(supervision)

#             # Calculate and print progress
#             progress_percentage = (index + 1) / total_audio_files * 100
#             print(f"Progress: {progress_percentage:.2f}% completed")

#         # Check if any recordings or supervisions were created
#         if not recordings:
#             print(f"Warning: No recordings found for '{split}'.")
#         if not supervisions:
#             print(f"Warning: No supervisions created for '{split}'.")

#         # Create RecordingSet and SupervisionSet
#         print(f"Creating RecordingSet and SupervisionSet for '{split}'...")
#         recording_set = RecordingSet.from_recordings(recordings)
#         supervision_set = SupervisionSet.from_segments(supervisions)

#         # Save manifests
#         os.makedirs(output_dir, exist_ok=True)
#         recording_set.to_file(os.path.join(output_dir, f"{split}_recordings.jsonl"))
#         supervision_set.to_file(os.path.join(output_dir, f"{split}_supervisions.jsonl"))

#         print(f"Manifests for '{split}' prepared successfully!")

# # Run the function to create the manifests for all splits
# if __name__ == "__main__":
#     create_manifests()



################################# fbank ####################################

# import os
# from lhotse import CutSet, Fbank, FbankConfig, SupervisionSet, RecordingSet
# from tqdm import tqdm

# # Define paths (update according to your dataset)
# datasets = {
#     "train": {
#         "audio_dir": "/disk1/polaris_intent_detection/organized_slurp_data/audio/train",
#         "transcript_dir": "/disk1/polaris_intent_detection/organized_slurp_data/original_transcripts/train/train.jsonl",
#         "output_dir": "/disk1/polaris_intent_detection/organized_slurp_data/data/fbank/slurp_feats_train",
#         "manifest_dir": "/disk1/polaris_intent_detection/organized_slurp_data/data/manifests"  # Path to manifests folder
#     },
#     "val": {
#         "audio_dir": "/disk1/polaris_intent_detection/organized_slurp_data/audio/val",
#         "transcript_dir": "/disk1/polaris_intent_detection/organized_slurp_data/original_transcripts/val/val.jsonl",
#         "output_dir": "/disk1/polaris_intent_detection/organized_slurp_data/data/fbank/slurp_feats_val",
#         "manifest_dir": "/disk1/polaris_intent_detection/organized_slurp_data/data/manifests"
#     },
#     "test": {
#         "audio_dir": "/disk1/polaris_intent_detection/organized_slurp_data/audio/test",
#         "transcript_dir": "/disk1/polaris_intent_detection/organized_slurp_data/original_transcripts/test/test.jsonl",
#         "output_dir": "/disk1/polaris_intent_detection/organized_slurp_data/data/fbank/slurp_feats_test",
#         "manifest_dir": "/disk1/polaris_intent_detection/organized_slurp_data/data/manifests"
#     }
# }

# # Function to compute fbank features and save manifests
# def compute_fbank():
#     fbank = Fbank(FbankConfig(num_mel_bins=80))  # Configuring FBANK extraction
#     for split, paths in datasets.items():
#         print(f"Processing split: {split}")
#         audio_dir = paths['audio_dir']
#         transcript_path = paths['transcript_dir']
#         output_dir = paths['output_dir']
#         manifest_dir = paths['manifest_dir']  # Correct path to the manifest folder
        
#         os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
        
#         # Load recording set from the correct manifest directory
#         recordings = RecordingSet.from_jsonl(os.path.join(manifest_dir, f'{split}_recordings.jsonl'))
#         supervisions = SupervisionSet.from_jsonl(os.path.join(manifest_dir, f'{split}_supervisions.jsonl'))
        
#         # Create CutSet (combines recordings and transcripts)
#         cuts = CutSet.from_manifests(recordings=recordings, supervisions=supervisions)
        
#         # Extract FBANK features and save to .lca files (default storage behavior)
#         cut_set_with_feats = cuts.compute_and_store_features(
#             extractor=fbank, 
#             storage_path=output_dir,  # Directory to store features
#             num_jobs=4  # Parallel jobs for faster processing
#         )
        
#         # Save the CutSet with the features references
#         cut_set_with_feats.to_file(os.path.join(output_dir, f'slurp_cuts_{split}.jsonl'))
        
#         print(f"FBANK features for {split} split saved successfully!")

# # Run the function to compute fbank
# if __name__ == "__main__":
#     compute_fbank()
##############
# import os
# import torch
# from lhotse import CutSet, Fbank, FbankConfig, SupervisionSet, RecordingSet
# from tqdm import tqdm

# # Set torch threads to 1 to avoid multi-threading issues
# torch.set_num_threads(1)

# # Define paths (update according to your dataset)
# datasets = {
#     "train": {
#         "audio_dir": "/disk1/polaris_intent_detection/organized_slurp_data/audio/train",
#         "transcript_dir": "/disk1/polaris_intent_detection/organized_slurp_data/original_transcripts/train/train.jsonl",
#         "output_dir": "/disk1/polaris_intent_detection/organized_slurp_data/data/fbank/slurp_feats_train",
#         "manifest_dir": "/disk1/polaris_intent_detection/organized_slurp_data/data/manifests"
#     },
#     "val": {
#         "audio_dir": "/disk1/polaris_intent_detection/organized_slurp_data/audio/val",
#         "transcript_dir": "/disk1/polaris_intent_detection/organized_slurp_data/original_transcripts/val/val.jsonl",
#         "output_dir": "/disk1/polaris_intent_detection/organized_slurp_data/data/fbank/slurp_feats_val",
#         "manifest_dir": "/disk1/polaris_intent_detection/organized_slurp_data/data/manifests"
#     },
#     "test": {
#         "audio_dir": "/disk1/polaris_intent_detection/organized_slurp_data/audio/test",
#         "transcript_dir": "/disk1/polaris_intent_detection/organized_slurp_data/original_transcripts/test/test.jsonl",
#         "output_dir": "/disk1/polaris_intent_detection/organized_slurp_data/data/fbank/slurp_feats_test",
#         "manifest_dir": "/disk1/polaris_intent_detection/organized_slurp_data/data/manifests"
#     }
# }

# # Function to compute fbank features and save manifests
# def compute_fbank():
#     fbank = Fbank(FbankConfig(num_mel_bins=80))  # Configure FBANK extraction
#     for split, paths in datasets.items():
#         print(f"Processing split: {split}")
#         audio_dir = paths['audio_dir']
#         output_dir = paths['output_dir']
#         manifest_dir = paths['manifest_dir']
        
#         os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
        
#         # Load recording set and supervisions
#         recordings = RecordingSet.from_jsonl(os.path.join(manifest_dir, f'{split}_recordings.jsonl'))
#         supervisions = SupervisionSet.from_jsonl(os.path.join(manifest_dir, f'{split}_supervisions.jsonl'))
        
#         # Deduplicate recordings based on id (avoiding duplicates)
#         unique_recording_ids = set()
#         deduplicated_recordings = []
        
#         for rec in recordings:
#             if rec.id not in unique_recording_ids:
#                 unique_recording_ids.add(rec.id)
#                 deduplicated_recordings.append(rec)
#             else:
#                 print(f"Duplicate recording found and skipped: {rec.id}")

#         # Create a new RecordingSet with deduplicated entries
#         deduplicated_recording_set = RecordingSet.from_recordings(deduplicated_recordings)
        
#         # Create CutSet (combines recordings and supervisions)
#         cuts = CutSet.from_manifests(recordings=deduplicated_recording_set, supervisions=supervisions)
        
#         # Extract FBANK features and save to .lca files (default storage behavior)
#         with tqdm(total=len(cuts), desc=f"Extracting FBANK for {split}", unit="cut") as pbar:
#             cut_set_with_feats = cuts.compute_and_store_features(
#                 extractor=fbank,
#                 storage_path=output_dir,  # Directory to store features
#                 num_jobs=4  # Parallel jobs for faster processing
#             )
#             pbar.update(1)  # Update progress for each cut processed
        
#         # Save the CutSet with the features references
#         cut_set_with_feats.to_file(os.path.join(output_dir, f'slurp_cuts_{split}.jsonl'))
        
#         print(f"FBANK features for {split} split saved successfully!")

# # Run the function to compute fbank
# if __name__ == "__main__":
#     compute_fbank()
################
# import os
# import torch  # Add this import
# from lhotse import CutSet, Fbank, FbankConfig, SupervisionSet, RecordingSet
# from tqdm import tqdm
# import json

# # Set PyTorch threads to 1 to avoid conflicts with num_jobs
# torch.set_num_threads(1)

# # Define paths (update according to your dataset)
# datasets = {
#     "train": {
#         "audio_dir": "/disk1/polaris_intent_detection/organized_slurp_data/audio/train",
#         "transcript_dir": "/disk1/polaris_intent_detection/organized_slurp_data/original_transcripts/train/train.jsonl",
#         "output_dir": "/disk1/polaris_intent_detection/organized_slurp_data/data/fbank/slurp_feats_train",
#         "manifest_dir": "/disk1/polaris_intent_detection/organized_slurp_data/data/manifests"  # Path to manifests folder
#     },
#     "val": {
#         "audio_dir": "/disk1/polaris_intent_detection/organized_slurp_data/audio/val",
#         "transcript_dir": "/disk1/polaris_intent_detection/organized_slurp_data/original_transcripts/val/val.jsonl",
#         "output_dir": "/disk1/polaris_intent_detection/organized_slurp_data/data/fbank/slurp_feats_val",
#         "manifest_dir": "/disk1/polaris_intent_detection/organized_slurp_data/data/manifests"
#     },
#     "test": {
#         "audio_dir": "/disk1/polaris_intent_detection/organized_slurp_data/audio/test",
#         "transcript_dir": "/disk1/polaris_intent_detection/organized_slurp_data/original_transcripts/test/test.jsonl",
#         "output_dir": "/disk1/polaris_intent_detection/organized_slurp_data/data/fbank/slurp_feats_test",
#         "manifest_dir": "/disk1/polaris_intent_detection/organized_slurp_data/data/manifests"
#     }
# }

# # Function to compute fbank features and save manifests
# def compute_fbank():
#     fbank = Fbank(FbankConfig(num_mel_bins=80))  # Configuring FBANK extraction
#     for split, paths in datasets.items():
#         print(f"Processing split: {split}")
#         audio_dir = paths['audio_dir']
#         transcript_path = paths['transcript_dir']
#         output_dir = paths['output_dir']
#         manifest_dir = paths['manifest_dir']  # Correct path to the manifest folder
        
#         os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists
        
#         # Load recording set from the correct manifest directory
#         recordings = RecordingSet.from_jsonl(os.path.join(manifest_dir, f'{split}_recordings.jsonl'))
#         supervisions = SupervisionSet.from_jsonl(os.path.join(manifest_dir, f'{split}_supervisions.jsonl'))
        
#         # Create CutSet (combines recordings and transcripts)
#         cuts = CutSet.from_manifests(recordings=recordings, supervisions=supervisions)
        
#         # Extract FBANK features and save to .lca files (default storage behavior)
#         cut_set_with_feats = cuts.compute_and_store_features(
#             extractor=fbank, 
#             storage_path=output_dir,  # Directory to store features
#             num_jobs=1  # Use 1 job to avoid multithreading issues
#         )
        
#         # Save the CutSet with the features references
#         cut_set_with_feats.to_file(os.path.join(output_dir, f'1slurp_cuts_{split}.jsonl'))
        
#         print(f"FBANK features for {split} split saved successfully!")

# # Run the function to compute fbank
# if __name__ == "__main__":
#     compute_fbank()

################################ update manifests supervisions with intent ##########################

# import os
# import json

# # Define paths to your data
# dataset_splits = ["train", "val", "test"]
# manifests_dir = "/disk1/polaris_intent_detection/organized_slurp_data/data/manifests"
# transcripts_dir = "/disk1/polaris_intent_detection/organized_slurp_data/original_transcripts"

# def update_manifest_with_intent(manifest_path, transcript_path, output_path):
#     # Load the transcripts
#     transcript_dict = {}
#     with open(transcript_path, 'r') as transcript_file:
#         for line in transcript_file:
#             entry = json.loads(line)
#             for recording in entry["recordings"]:
#                 audio_id = os.path.splitext(recording["file"])[0]
#                 transcript_dict[audio_id] = entry["intent"]

#     # Update the manifest with intent appended to the text
#     updated_entries = []
#     with open(manifest_path, 'r') as manifest_file:
#         for line in manifest_file:
#             entry = json.loads(line)
#             audio_id = entry["recording_id"]
#             intent = transcript_dict.get(audio_id, "")
#             if intent:
#                 entry["text"] = f'{entry["text"]} <{intent}>'
#             updated_entries.append(entry)

#     # Save the updated manifest
#     with open(output_path, 'w') as output_file:
#         for entry in updated_entries:
#             output_file.write(json.dumps(entry) + '\n')

# def process_manifests():
#     for split in dataset_splits:
#         manifest_path = os.path.join(manifests_dir, f"{split}_supervisions.jsonl")
#         transcript_path = os.path.join(transcripts_dir, split, f"{split}.jsonl")
#         output_path = os.path.join(manifests_dir, f"{split}_supervisions_with_intent.jsonl")
        
#         print(f"Processing {split} manifest...")
#         update_manifest_with_intent(manifest_path, transcript_path, output_path)
#         print(f"Updated {split} manifest saved to {output_path}")

# if __name__ == "__main__":
#     process_manifests()

############################## capitalize transcript in manifest/ supervision's text transcript, except the intent(?) ############

# import os
# import json
# import re

# # Define paths to your data
# dataset_splits = ["train", "val", "test"]
# manifests_dir = "/disk1/polaris_intent_detection/organized_slurp_data/data/manifests"

# def capitalize_text_in_manifest(manifest_path, output_path):
#     updated_entries = []
    
#     # Regular expression to find the intent (text within <>)
#     intent_pattern = re.compile(r"<.*?>")
    
#     # Process each entry in the manifest
#     with open(manifest_path, 'r') as manifest_file:
#         for line in manifest_file:
#             entry = json.loads(line)
#             text = entry["text"]

#             # Extract the intent (if any) using regex
#             intent_match = intent_pattern.search(text)
            
#             if intent_match:
#                 # Split the text and intent
#                 intent = intent_match.group(0)
#                 text_without_intent = text.replace(intent, "").strip()
                
#                 # Capitalize the main text and append the intent unchanged
#                 capitalized_text = text_without_intent.upper()
#                 entry["text"] = f"{capitalized_text} {intent}"
#             else:
#                 # If no intent is found, capitalize the entire text
#                 entry["text"] = text.upper()
            
#             updated_entries.append(entry)

#     # Save the updated manifest to a new file
#     with open(output_path, 'w') as output_file:
#         for entry in updated_entries:
#             output_file.write(json.dumps(entry) + '\n')

# def process_manifests():
#     for split in dataset_splits:
#         manifest_path = os.path.join(manifests_dir, f"{split}_supervisions_with_intent.jsonl")
#         output_path = os.path.join(manifests_dir, f"{split}_supervisions_with_intent_capitalized.jsonl")
        
#         print(f"Processing {split} manifest...")
#         capitalize_text_in_manifest(manifest_path, output_path)
#         print(f"Updated {split} manifest with capitalized text saved to {output_path}")

# if __name__ == "__main__":
#     process_manifests()

################################ update fbank cuts text with intent and capitalize it (except intent) ##########################

import json
import os

# Define paths
base_cuts_path = '/disk1/polaris_intent_detection/organized_slurp_data/data/fbank'
transcripts_path = '/disk1/polaris_intent_detection/organized_slurp_data/original_transcripts'

# Load the corresponding transcript for each split
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

# Function to update and save the cuts file with capitalized text and intent
def update_cuts_file(split):
    # Load the transcripts for the current split
    transcripts = load_transcripts(split)

    cuts_file = os.path.join(base_cuts_path, f'slurp_cuts_{split}.jsonl')
    updated_cuts = []

    # Read, modify and update cuts
    with open(cuts_file, 'r') as f:
        for line in f:
            cut = json.loads(line)
            recording_id = cut["recording"]["id"]
            supervision = cut["supervisions"][0]  # Assuming there's always one supervision per cut

            # Lookup corresponding transcript data
            recording_flac = f"{recording_id}.flac"
            if recording_flac in transcripts:
                transcript = transcripts[recording_flac]
                text = transcript["text"].upper()  # Capitalize the text
                intent = transcript["intent"]  # Get the intent
                supervision["text"] = f"{text} <{intent}>"  # Update the supervision text

            updated_cuts.append(cut)

    # Write the updated cuts back to the file
    updated_cuts_file = os.path.join(base_cuts_path, f'slurp_cuts_{split}_updated.jsonl')
    with open(updated_cuts_file, 'w') as f:
        for cut in updated_cuts:
            f.write(json.dumps(cut) + '\n')

    print(f"Updated cuts saved to {updated_cuts_file}")

# Run the script for all splits: test, train, val
if __name__ == "__main__":
    for split in ['test', 'train', 'val']:
        print(f"Processing split: {split}")
        update_cuts_file(split)
        print(f"Finished processing {split} split.")
