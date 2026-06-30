import re

input_file = '/disk1/polaris_intent_detection/Tokenizer_SLURP/transcript_words_custom_slurp.txt'
output_file = '/disk1/polaris_intent_detection/Tokenizer_SLURP/intent_tokens.txt'

unique_intents = set()

with open(input_file, 'r') as infile:
    for line in infile:
        intents = re.findall(r'<[^>]+>', line)
        unique_intents.update(intents)

with open(output_file, 'w') as outfile:
    for intent in sorted(unique_intents):
        outfile.write(f"{intent}\n")

print(f"Extracted {len(unique_intents)} unique intents to {output_file}")
