import sentencepiece as spm

input_file = '/disk1/polaris_intent_detection/Tokenizer_SLURP/train_transcript_words_custom_slurp.txt'
output_directory = '/disk1/polaris_intent_detection/NEWBATCH_Tokenizer'
model_prefix = f'{output_directory}/cslurp_with_predefined_intents'
user_defined_symbols_file = '/disk1/polaris_intent_detection/Tokenizer_SLURP/intent_tokens.txt'
vocab_size = 592 #2000  librispeech , 2000 + 92 on slurp, test
character_coverage = 0.9995
model_type = 'bpe'

with open(user_defined_symbols_file, 'r') as file:
    user_defined_symbols = file.read().strip().replace('\n', ',')
    user_defined_symbols = ','.join(symbol for symbol in user_defined_symbols.split(',') if symbol not in ['<unk>', '<s>', '</s>', '<pad>'])

# Training the SentencePiece model
spm.SentencePieceTrainer.train(
    input=input_file,
    model_prefix=model_prefix,
    vocab_size=vocab_size,
    character_coverage=character_coverage,
    model_type=model_type,
    user_defined_symbols=user_defined_symbols
)

print(f"SentencePiece model trained and saved as {model_prefix}.model and {model_prefix}.vocab")
