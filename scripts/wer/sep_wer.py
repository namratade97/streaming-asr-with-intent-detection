import sys
import re

def process_file(filename):
    ref_lines = []
    hyp_lines = []
    
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            match = re.match(r"(audio-\S+):\s*(ref|hyp)=\[(.*)\]", line)
            if match:
                key, label, content = match.groups()
                words = content.split(', ')
                
                words = [w.strip("'") for w in words if not re.match(r'<.*?>', w)]
                sentence = ' '.join(words)
                
                sentence = re.sub(r' <.*?>$', '', sentence)
                
                if label == 'ref':
                    ref_lines.append(sentence)
                elif label == 'hyp':
                    hyp_lines.append(sentence)
    
    ref_output = filename + "_ref.txt"
    hyp_output = filename + "_hyp.txt"
    
    with open(ref_output, 'w', encoding='utf-8') as ref_file:
        ref_file.write('\n'.join(ref_lines) + '\n')
    
    with open(hyp_output, 'w', encoding='utf-8') as hyp_file:
        hyp_file.write('\n'.join(hyp_lines) + '\n')
    
    print(f"Created {ref_output} and {hyp_output}")

def calculate_wer(ref_file, hyp_file):
    with open(ref_file, 'r', encoding='utf-8') as ref_f, open(hyp_file, 'r', encoding='utf-8') as hyp_f:
        ref_sentences = [line.strip().split() for line in ref_f]
        hyp_sentences = [line.strip().split() for line in hyp_f]
    
    total_words = 0
    total_errors = 0
    
    for ref, hyp in zip(ref_sentences, hyp_sentences):
        total_words += len(ref)
        total_errors += levenshtein_distance(ref, hyp)
    
    wer = total_errors / total_words if total_words > 0 else 0
    print(f"Word Error Rate (WER): {wer:.2%}")

def levenshtein_distance(ref, hyp):
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        for j in range(len(hyp) + 1):
            if i == 0:
                d[i][j] = j
            elif j == 0:
                d[i][j] = i
            else:
                cost = 0 if ref[i - 1] == hyp[j - 1] else 1
                d[i][j] = min(d[i - 1][j] + 1,  
                              d[i][j - 1] + 1,  
                              d[i - 1][j - 1] + cost)
    return d[-1][-1]

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_file>")
        sys.exit(1)
    
    process_file(sys.argv[1])
    ref_output = sys.argv[1] + "_ref.txt"
    hyp_output = sys.argv[1] + "_hyp.txt"
    calculate_wer(ref_output, hyp_output)
