import sentencepiece as spm
import time

def compare_sentencepiece_models(spm_path_1, spm_path_2, sentence):
    sp1 = spm.SentencePieceProcessor()
    sp2 = spm.SentencePieceProcessor()

    sp1.load(spm_path_1)
    sp2.load(spm_path_2)

    print(f"Vocab size tokenizer 1: {sp1.get_piece_size()}")
    print(f"Vocab size tokenizer 2: {sp2.get_piece_size()}\n")

    tokens_1 = sp1.encode(sentence, out_type=str)
    tokens_2 = sp2.encode(sentence, out_type=str)

    ids_1 = sp1.encode(sentence, out_type=int)
    ids_2 = sp2.encode(sentence, out_type=int)

    print(f"Tokenizer 1 tokens ({len(tokens_1)}): {tokens_1}")
    print(f"Tokenizer 2 tokens ({len(tokens_2)}): {tokens_2}\n")

    print(f"Tokenizer 1 token IDs: {ids_1}")
    print(f"Tokenizer 2 token IDs: {ids_2}\n")

    print("Tokens side by side:")
    max_len = max(len(tokens_1), len(tokens_2))
    for i in range(max_len):
        t1 = tokens_1[i] if i < len(tokens_1) else ""
        t2 = tokens_2[i] if i < len(tokens_2) else ""
        print(f"{t1:<20} | {t2:<20}")

    # Optional: timing
    runs = 1000

    start = time.time()
    for _ in range(runs):
        sp1.encode(sentence, out_type=str)
    t1_time = time.time() - start

    start = time.time()
    for _ in range(runs):
        sp2.encode(sentence, out_type=str)
    t2_time = time.time() - start

    print(f"\nTokenizer 1 time for {runs} runs: {t1_time:.4f} sec")
    print(f"Tokenizer 2 time for {runs} runs: {t2_time:.4f} sec")



if __name__ == "__main__":
    tokenizer_path_1 = "LIBRI_TRANSCRIPT_lower_592.model" #bad one
    tokenizer_path_2 = "LIBRI_TRANSCRIPT_lower_592_retrained.model" #good one
    sentence = "play a taylor swift song <play_music>"

    compare_sentencepiece_models(tokenizer_path_1, tokenizer_path_2, sentence)

    print("----------------------------------")
    tokenizer_path_3 = "libri_intent_592.model" #good one uppercase
    sentence_uppercase = "PLAY A TAYLOR SWIFT SONG <play_music>"
    sp3 = spm.SentencePieceProcessor()
    sp3.load(tokenizer_path_3)
    print(f"Vocab size tokenizer 3: {sp3.get_piece_size()}\n")
    tokens_3 = sp3.encode(sentence_uppercase, out_type=str)
    ids_3 = sp3.encode(sentence_uppercase, out_type=int)

    print(f"Tokenizer 3 tokens ({len(tokens_3)}): {tokens_3}")
    print(f"Tokenizer 3 token IDs: {ids_3}\n")

    print("----------------------------------")
    tokenizer_path_4 = "Multisource_NEW_Tokenizer/new_polaris_bpe592_lower_retrained.model" #good one uppercase
    sentence_4 = "play a taylor swift song <play_music>"
    sp4 = spm.SentencePieceProcessor()
    sp4.load(tokenizer_path_4)
    print(f"Vocab size tokenizer 4 (Multisource): {sp4.get_piece_size()}\n")
    tokens_4 = sp4.encode(sentence_4, out_type=str)
    ids_4 = sp4.encode(sentence_4, out_type=int)

    print(f"Tokenizer 4 (Multisource) tokens ({len(tokens_4)}): {tokens_4}")
    print(f"Tokenizer 4 (Multisource) token IDs: {ids_4}\n")


    
