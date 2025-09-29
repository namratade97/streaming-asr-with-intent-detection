import re
from pathlib import Path
from sentencepiece import sentencepiece_model_pb2 as model
# import sentencepiece_model_pb2 as model  # Import the protobuf model for SentencePiece

def update_tokenizer(
    bpe_path: Path,
    new_bpe_path: Path,
    text_path: Path,
):
    # Load the existing SentencePiece model
    m = model.ModelProto()
    with bpe_path.open("rb") as f:
        m.ParseFromString(f.read())

    # Read the text file and extract tokens enclosed in angle brackets
    with text_path.open() as f:
        # tags = set()  # Use a set to avoid duplicates
        # for line in f:
        #     matches = re.findall(r'<.*?>', line)  # Extract tokens like <add>, <modify>
        #     tags.update(matches)
        tags = []
        seen = set()
        for line in f:
            matches = re.findall(r'<.*?>', line)
            for tag in matches:
                if tag not in seen:
                    tags.append(tag)
                    seen.add(tag)

    print(f"Tokens to be added: {tags}")

    # Append new tokens to the SentencePiece model
    # for tag in tags:
    #     new_token = model.ModelProto().SentencePiece()
    #     new_token.piece = f"▁{tag}"  # SentencePiece tokens usually start with ▁
    #     new_token.score = 0  # You can set a score for each new token if desired
    #     # m.pieces.append(new_token)
    #     # m.pieces.insert(0, new_token)
    #     m.pieces.insert(3, new_token)

    # Get set of existing pieces for faster lookup
    existing_pieces = set(p.piece for p in m.pieces)

    insertion_index = 3  # After <unk>, <s>, </s>
    for tag in tags:
        token_piece = f"▁{tag}"
        if token_piece in existing_pieces:
            print(f"Token {token_piece} already in model, skipping.")
            continue

        new_token = model.ModelProto().SentencePiece()
        new_token.piece = token_piece
        new_token.score = 0
        m.pieces.insert(insertion_index, new_token)
        insertion_index += 1  # So that order is preserved


    # Write the new model to the specified path
    with new_bpe_path.open("wb") as f:
        f.write(m.SerializeToString())

    print(f"Wrote new BPE model to {new_bpe_path}")

# Define paths
# bpe_path = Path("/disk1/polaris_intent_detection/lang-bpe-og/bpe.model")  # Path to your existing SentencePiece model
bpe_path = Path("/disk1/nde/polaris_intent_detection/Multisource_Tokenizer/polaris_bpe500_lower.model")
# bpe_path = Path("/disk1/nde/polaris_intent_detection/LIBRI_TRANSCRIPT_lower.model")


# new_bpe_path = Path("/disk1/polaris_intent_detection/organized_slurp_data/data/lang_bpe_og_plus_intent/libri_intent_592.model")  # Path to save the new model with added tokens
new_bpe_path = Path("/disk1/nde/polaris_intent_detection/Multisource_NEW_Tokenizer/new_polaris_bpe592_lower.model")


text_path = Path("/disk1/nde/polaris_intent_detection/intent_sorted.txt")  # Text file containing new tokens to add

# Call the function to update the tokenizer
update_tokenizer(bpe_path, new_bpe_path, text_path)
