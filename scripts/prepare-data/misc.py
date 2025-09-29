# from lhotse import SupervisionSet

# # Example usage to verify if SupervisionSet works as expected
# supervision_set = SupervisionSet.from_dicts([

# {"id": "9024_audio-1501754435.flac", "recording_id": "9024", "start": 0.0, "duration": 5.0, "text": "event <calendar_set>"},
# {"id": "9024_audio-1501407267-headset.flac", "recording_id": "9024", "start": 0.0, "duration": 5.0, "text": "event <calendar_set>"},
# {"id": "9024_audio-1501407267.flac", "recording_id": "9024", "start": 0.0, "duration": 5.0, "text": "event <calendar_set>"},
# {"id": "9024_audio-1501771798-headset.flac", "recording_id": "9024", "start": 0.0, "duration": 5.0, "text": "event <calendar_set>"},
# {"id": "9024_audio-1501771798.flac", "recording_id": "9024", "start": 0.0, "duration": 5.0, "text": "event <calendar_set>"},
# {"id": "9024_audio-1490705711-headset.flac", "recording_id": "9024", "start": 0.0, "duration": 5.0, "text": "event <calendar_set>"},
# {"id": "9024_audio-1490705711.flac", "recording_id": "9024", "start": 0.0, "duration": 5.0, "text": "event <calendar_set>"},
# {"id": "9024_audio-1494416970-headset.flac", "recording_id": "9024", "start": 0.0, "duration": 5.0, "text": "event <calendar_set>"}


# ])
# print(supervision_set)


# from lhotse.supervision import SupervisionSegment

# # Print the expected attributes of SupervisionSegment
# help(SupervisionSegment)

import json

input_file = 'data/manifests/slurp_supervisions_all.jsonl'
output_file = 'data/manifests/slurp_supervisions_all_upper.jsonl'

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        data = json.loads(line.strip())  # Parse the JSON data from each line
        data['text'] = data['text'].upper()  # Convert the 'text' field to uppercase
        outfile.write(json.dumps(data) + '\n')  # Write the modified data back as JSONL
