# import json
# import os

# # Paths to your data
# input_jsonl_path = '/disk1/polaris_intent_detection/slurp/dataset/slurp/train.jsonl'
# output_dir = '/disk1/polaris_intent_detection/Prepare_SLURP/data/manifests'
# supervision_file_path = os.path.join(output_dir, 'supervisions_alternative.jsonl')

# os.makedirs(output_dir, exist_ok=True)

# def create_supervision_manifest(input_jsonl_path, supervision_file_path):
#     with open(input_jsonl_path, 'r') as infile, open(supervision_file_path, 'w') as outfile:
#         for line in infile:
#             entry = json.loads(line)
#             recording_id = entry.get('slurp_id', '')
#             transcript = entry.get('sentence', '')
#             intent = entry.get('intent', '')
            
#             if not recording_id or not transcript or not intent:
#                 continue
            
#             # Process each recording
#             recordings = entry.get('recordings', [])
#             for recording in recordings:
#                 file_name = recording.get('file', '')
#                 if not file_name:
#                     continue
                
#                 # Include the intent in the text field
#                 text_with_intent = f"{transcript} <{intent}>"
                
#                 # Create a supervision entry with intent in the text
#                 supervision_entry = {
#                     'id': f'{recording_id}_{file_name}',
#                     'recording_id': recording_id,
#                     'text': text_with_intent,
#                     'audio_file': file_name
#                 }
#                 outfile.write(json.dumps(supervision_entry) + '\n')

# create_supervision_manifest(input_jsonl_path, supervision_file_path)

import json
import os

# Paths to your data
input_jsonl_path = '/disk1/polaris_intent_detection/slurp/dataset/slurp/train.jsonl'
output_dir = '/disk1/polaris_intent_detection/Prepare_SLURP/data/manifests'
supervision_file_path = os.path.join(output_dir, 'slurp_supervisions_all.jsonl')

os.makedirs(output_dir, exist_ok=True)

def create_supervision_manifest(input_jsonl_path, supervision_file_path):
    with open(input_jsonl_path, 'r') as infile, open(supervision_file_path, 'w') as outfile:
        for line in infile:
            entry = json.loads(line)
            recording_id = str(entry.get('slurp_id', ''))
            transcript = entry.get('sentence', '')
            intent = entry.get('intent', '')
            
            if not recording_id or not transcript or not intent:
                continue
            
            # Process each recording
            recordings = entry.get('recordings', [])
            for recording in recordings:
                file_name = recording.get('file', '')
                if not file_name:
                    continue
                
                # Dummy values for `start` and `duration` (adjust as needed)
                start = 0.0  # Replace with actual start time if available
                duration = 5.0  # Replace with actual duration if available
                
                # Include the intent in the text field
                text_with_intent = f"{transcript} <{intent}>"
                
                # Create a supervision entry with intent in the text
                supervision_entry = {
                    'id': f'{recording_id}_{file_name}',
                    'recording_id': recording_id,
                    'start': start,
                    'duration': duration,
                    'text': text_with_intent
                }
                outfile.write(json.dumps(supervision_entry) + '\n')

create_supervision_manifest(input_jsonl_path, supervision_file_path)
