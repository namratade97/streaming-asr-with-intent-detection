import shutil

vocab_file = '/disk1/polaris_intent_detection/Tokenizer_SLURP/models_og/slurp_bpe.vocab'
intent_file = '/disk1/polaris_intent_detection/Tokenizer_SLURP/intent_tokens.txt'
combined_vocab_file = '/disk1/polaris_intent_detection/Tokenizer_SLURP/model_vocab_updated/combined_slurp.vocab'

with open(vocab_file, 'r') as f:
    vocab_lines = f.readlines()

with open(intent_file, 'r') as f:
    intent_lines = f.readlines()

with open(combined_vocab_file, 'w') as f:
    for line in vocab_lines:
        f.write(line)
    for line in intent_lines:
        f.write(line)

print(f"Combined vocab file created at {combined_vocab_file}")
