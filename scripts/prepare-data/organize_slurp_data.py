import os
import json
import shutil
import random

SLURP_AUDIO_REAL = 'slurp/audio/slurp_real'
SLURP_AUDIO_SYNTH = 'slurp/audio/slurp_synth'
SLURP_DATASET = 'slurp/dataset/slurp'
OUTPUT_DIR = 'organize_slurp_data'

AUDIO_DIR = os.path.join(OUTPUT_DIR, 'audio')
TRANSCRIPTS_DIR = os.path.join(OUTPUT_DIR, 'original_transcripts')

def create_directory_structure():
    for folder in ['train', 'val', 'test']:
        os.makedirs(os.path.join(AUDIO_DIR, folder), exist_ok=True)
        os.makedirs(os.path.join(TRANSCRIPTS_DIR, folder), exist_ok=True)

def split_train_val(train_jsonl_path, val_ratio=0.1):
    with open(train_jsonl_path, 'r') as f:
        data = [json.loads(line) for line in f]

    random.shuffle(data)
    val_size = int(len(data) * val_ratio)
    train_data = data[val_size:]
    val_data = data[:val_size]

    return train_data, val_data

def save_jsonl(data, output_path):
    with open(output_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def copy_audio_files(data, dest_dir, audio_dirs):
    for item in data:
        for recording in item['recordings']:
            audio_file = recording['file']
            for audio_dir in audio_dirs:
                src_file = os.path.join(audio_dir, audio_file)
                if os.path.exists(src_file):
                    shutil.copy(src_file, os.path.join(dest_dir, audio_file))
                    break

def process_dataset():
    create_directory_structure()

    # Split the train data into train and val
    train_data, val_data = split_train_val(os.path.join(SLURP_DATASET, 'train.jsonl'))

    # Copy audio files and transcripts for train
    save_jsonl(train_data, os.path.join(TRANSCRIPTS_DIR, 'train', 'train.jsonl'))
    copy_audio_files(train_data, os.path.join(AUDIO_DIR, 'train'), [SLURP_AUDIO_REAL, SLURP_AUDIO_SYNTH])

    # Copy audio files and transcripts for val
    save_jsonl(val_data, os.path.join(TRANSCRIPTS_DIR, 'val', 'val.jsonl'))
    copy_audio_files(val_data, os.path.join(AUDIO_DIR, 'val'), [SLURP_AUDIO_REAL, SLURP_AUDIO_SYNTH])

    # Copy test set (audio and transcripts already exist)
    shutil.copy(os.path.join(SLURP_DATASET, 'test.jsonl'), os.path.join(TRANSCRIPTS_DIR, 'test', 'test.jsonl'))
    test_data = [json.loads(line) for line in open(os.path.join(SLURP_DATASET, 'test.jsonl'))]
    copy_audio_files(test_data, os.path.join(AUDIO_DIR, 'test'), [SLURP_AUDIO_REAL, SLURP_AUDIO_SYNTH])

if __name__ == "__main__":
    process_dataset()
