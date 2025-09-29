# if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
#   log "Stage 1: Prepare LibriSpeech manifest"
#   # We assume that you have downloaded the LibriSpeech corpus
#   # to $dl_dir/LibriSpeech
#   mkdir -p data/manifests
#   if [ ! -e data/manifests/.librispeech.done ]; then
#     lhotse prepare librispeech -j $nj $dl_dir/LibriSpeech data/manifests
#     touch data/manifests/.librispeech.done
#   fi
# fi


import os
from lhotse import AudioSource, Recording, RecordingSet, SupervisionSegment, SupervisionSet

# Define your paths for train and test
datasets = {
    "train": {
        "audio_dir": "/disk1/polaris_intent_detection/slurp/audio/slurp_real",
        "transcript_dir": "/disk1/polaris_intent_detection/TOKENIZER/train_transcripts",
        "output_dir": "/disk1/polaris_intent_detection/Prepare_SLURP/data/manifests"
    }
    # },
    # "test": {
    #     "audio_dir": "../slurp/audio/slurp_real/test",
    #     "transcript_dir": "../slurp/transcripts/test",
    #     "output_dir": "data/manifests/test"
    # }
}

for split, paths in datasets.items():
    audio_dir = paths['audio_dir']
    transcript_dir = paths['transcript_dir']
    output_dir = paths['output_dir']

    # Initialize empty lists to hold recordings and supervisions
    recordings = []
    supervisions = []

    # Loop through the audio directory
    for root, _, files in os.walk(audio_dir):
        for file in files:
            if file.endswith(".flac"):
                audio_path = os.path.join(root, file)
                audio_id = os.path.splitext(file)[0]

                # Create a Recording object for each audio file
                recording = Recording.from_file(audio_path, recording_id=audio_id)
                recordings.append(recording)

                # Find the corresponding transcript
                transcript_path = os.path.join(transcript_dir, f"{audio_id}.txt")  # Adjust extension if needed
                if os.path.exists(transcript_path):
                    with open(transcript_path, 'r') as f:
                        transcript = f.read().strip()

                    # Create a SupervisionSegment for each transcript
                    supervision = SupervisionSegment(
                        id=audio_id,
                        recording_id=audio_id,
                        start=0.0,
                        duration=recording.duration,
                        text=transcript
                    )
                    supervisions.append(supervision)

    # Create RecordingSet and SupervisionSet
    recording_set = RecordingSet.from_recordings(recordings)
    supervision_set = SupervisionSet.from_segments(supervisions)

    # Save manifests
    os.makedirs(output_dir, exist_ok=True)
    recording_set.to_file(os.path.join(output_dir, "recordings.jsonl"))
    supervision_set.to_file(os.path.join(output_dir, "supervisions.jsonl"))

    print(f"Manifests for {split} prepared successfully!")
