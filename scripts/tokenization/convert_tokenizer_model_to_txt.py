"""Converts a BPE model in the binary format to a tokens text file.

Usage:
    python convert_bpe_model_to_tokens.py <bpe_model_path> <tokens_file_path>
"""

bpe_model = "/disk1/nde/polaris_intent_detection/bpe_500.model" 
tokens_file = "/disk1/nde/polaris_intent_detection/bpe_500.txt"

print(f"Loading BPE model from {bpe_model}.")
print(f"Tokens file written to {tokens_file}.")

sp = spm.SentencePieceProcessor(bpe_model)
# sp = spm.SentencePieceProcessor()
# sp.load(args.bpe_model)
with open(tokens_file, "w") as f:
    for i in range(sp.vocab_size()):
        f.write(f"{sp.id_to_piece(i)} {i}\n")

print("Done.")