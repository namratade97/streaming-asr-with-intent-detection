"""Converts a BPE model in the binary format to a tokens text file.

Usage:
    python convert_bpe_model_to_tokens.py <bpe_model_path> <tokens_file_path>
"""

# import argparse
import sentencepiece as spm
# from sentencepiece import sentencepiece_model_pb2 as model

# Directly provide the paths here
# bpe_model = "Tokenizer_SLURP/models_LIBRI_2000_INTENT/libri_bpe_2000_intent.model"  # Update with your BPE model path
# tokens_file = "Tokenizer_SLURP/models_LIBRI_2000_INTENT/libri_bpe_2000_intent.vocab"  # Update with your desired tokens file path

# bpe_model = "Multisource_Tokenizer/polaris_bpe592_lower.model"  # Update with your BPE model path
# tokens_file = "Multisource_Tokenizer/polaris_bpe592_lower.vocab"  # Update with your desired tokens file path

bpe_model = "/disk1/nde/polaris_intent_detection/bpe_500.model"  # Update with your BPE model path
tokens_file = "/disk1/nde/polaris_intent_detection/bpe_500.txt"  # Update with your desired tokens file path


# parser = argparse.ArgumentParser()
# parser.add_argument("bpe_model", type=str, help="Path to BPE model.")
# parser.add_argument("tokens_file", type=str, help="Path to output tokens file.")

# args = parser.parse_args()

print(f"Loading BPE model from {bpe_model}.")
print(f"Tokens file written to {tokens_file}.")

sp = spm.SentencePieceProcessor(bpe_model)
# sp = spm.SentencePieceProcessor()
# sp.load(args.bpe_model)
with open(tokens_file, "w") as f:
    for i in range(sp.vocab_size()):
        f.write(f"{sp.id_to_piece(i)} {i}\n")

print("Done.")