import json

# Path to SLURP dataset
input_file = "/disk1/polaris_intent_detection/slurp/dataset/slurp/train.jsonl"
output_file = "/disk1/polaris_intent_detection/Tokenizer_SLURP/train_transcript_words_custom_slurp.txt"

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        data = json.loads(line)
        sentence = data["sentence"]
        intent = data["intent"]
        
        # Combine sentence and intent
        formatted_sentence = f"{sentence.upper()} <{intent}>"
        
        outfile.write(formatted_sentence + '\n')

print(f"Transcripts have been saved to {output_file}")
