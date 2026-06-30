import os
import json

def check_audio_files(jsonl_file, audio_folder):
    missing_files = []
    
    with open(jsonl_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            
            recordings = data.get('recordings', [])
            for recording in recordings:
                file_name = recording['file']
                file_path = os.path.join(audio_folder, file_name)
                
                if not os.path.exists(file_path):
                    missing_files.append(file_path)
    
    if missing_files:
        print(f"Missing files: {len(missing_files)}")
        for file in missing_files:
            print(file)
    else:
        print("All files exist!")

check_audio_files('organized_slurp_data/original_transcripts/train/train.jsonl', 'organized_slurp_data/audio/train')
check_audio_files('organized_slurp_data/original_transcripts/test/test.jsonl', 'organized_slurp_data/audio/test')
check_audio_files('organized_slurp_data/original_transcripts/val/val.jsonl', 'organized_slurp_data/audio/val')
