import sys
import re

def parse_wer_report(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    entries = []
    current_entry = []
    current_wer = None
    
    for line in lines:
        line = line.strip()
        if line.startswith("WER: "):
            match = re.search(r"WER: ([\d\.]+)%", line)
            if match:
                current_wer = float(match.group(1))
            current_entry.append(line)
            entries.append((current_wer, current_entry))
            current_entry = []
            current_wer = None
        elif line:
            current_entry.append(line)
    
    return sorted(entries, key=lambda x: x[0], reverse=True)

def write_sorted_wer_report(sorted_entries, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for _, entry in sorted_entries:
            f.write("\n".join(entry) + "\n\n")

def main():
    if len(sys.argv) != 3:
        print("Usage: python sort_descending_wer.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    sorted_entries = parse_wer_report(input_file)
    write_sorted_wer_report(sorted_entries, output_file)
    print(f"Sorted WER report written to {output_file}")

if __name__ == "__main__":
    main()
