import json

input_file = 'data/manifests/slurp_supervisions_all.jsonl'
output_file = 'data/manifests/slurp_supervisions_all_upper.jsonl'

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        data = json.loads(line.strip())  
        data['text'] = data['text'].upper()
        outfile.write(json.dumps(data) + '\n')
