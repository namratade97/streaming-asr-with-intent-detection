import sentencepiece as spm
sp = spm.SentencePieceProcessor()
sp.load("LIBRI_TRANSCRIPT_lower_592.model")

print("Vocab size:", sp.get_piece_size())
print("<email_query>  →", sp.piece_to_id("<email_query>"))
print("▁<email_query> →", sp.piece_to_id("▁<email_query>"))
print("Encode '<email_query>':", sp.encode("<email_query>", out_type=int))
print("Encode ' <email_query>':", sp.encode(" <email_query>", out_type=int))
