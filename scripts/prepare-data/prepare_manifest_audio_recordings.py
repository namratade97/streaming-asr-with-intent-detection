import os
from lhotse import AudioSource, Recording, RecordingSet, SupervisionSegment, SupervisionSet

datasets = {
    "train": {
        "audio_dir": "/disk1/polaris_intent_detection/slurp/audio/slurp_real",
        "transcript_dir": "/disk1/polaris_intent_detection/TOKENIZER/train_transcripts",
        "output_dir": "/disk1/polaris_intent_detection/Prepare_SLURP/data/manifests"
    }
}

for split, paths in datasets.items():
    audio_dir = paths['audio_dir']
    transcript_dir = paths['transcript_dir']
    output_dir = paths['output_dir']

    recordings = []
    supervisions = []

    for root, _, files in os.walk(audio_dir):
        for file in files:
            if file.endswith(".flac"):
                audio_path = os.path.join(root, file)
                audio_id = os.path.splitext(file)[0]

                recording = Recording.from_file(audio_path, recording_id=audio_id)
                recordings.append(recording)

                transcript_path = os.path.join(transcript_dir, f"{audio_id}.txt")  # Adjust extension if needed
                if os.path.exists(transcript_path):
                    with open(transcript_path, 'r') as f:
                        transcript = f.read().strip()

                    supervision = SupervisionSegment(
                        id=audio_id,
                        recording_id=audio_id,
                        start=0.0,
                        duration=recording.duration,
                        text=transcript
                    )
                    supervisions.append(supervision)

    recording_set = RecordingSet.from_recordings(recordings)
    supervision_set = SupervisionSet.from_segments(supervisions)

    os.makedirs(output_dir, exist_ok=True)
    recording_set.to_file(os.path.join(output_dir, "recordings.jsonl"))
    supervision_set.to_file(os.path.join(output_dir, "supervisions.jsonl"))

    print(f"Manifests for {split} prepared successfully!")
