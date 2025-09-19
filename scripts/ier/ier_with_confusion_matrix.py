import re
import sys
import argparse
from collections import defaultdict, Counter
from sklearn.metrics import precision_recall_fscore_support, classification_report

def print_overall_metrics(samples):
    y_true = [sample['ref'] for sample in samples.values()]
    y_pred = [sample['hyp'] for sample in samples.values()]

    print("\nOverall Performance:")
    print(classification_report(y_true, y_pred, digits=3))



def build_confusion_matrix(samples, selected_intents):
    from collections import defaultdict
    matrix = defaultdict(lambda: defaultdict(int))

    for sample in samples.values():
        ref = sample.get('ref')
        hyp = sample.get('hyp')
        if ref in selected_intents or hyp in selected_intents:
            matrix[ref][hyp] += 1

    return matrix


def print_binary_confusion_matrix(samples, intent):
    """
    Prints a binary confusion matrix for 'intent' vs all other intents.
    Format:

                      predicted intent    predicted NOT intent
    true intent         TP                 FN
    true NOT intent     FP                 TN
    """
    TP = 0
    FN = 0
    FP = 0
    TN = 0

    for sample in samples.values():
        ref = sample.get('ref')
        hyp = sample.get('hyp')
        if ref == intent:
            if hyp == intent:
                TP += 1
            else:
                FN += 1
        else:
            if hyp == intent:
                FP += 1
            else:
                TN += 1

    print(f"\nBinary confusion matrix for intent: {color_text(intent, 'yellow')}")
    print(f"{'':>20} {'Predicted ' + intent:>20} {'Predicted NOT ' + intent:>25}")
    print(f"{'True ' + intent:>20} {TP:>20} {FN:>25}")
    print(f"{'True NOT ' + intent:>20} {FP:>20} {TN:>25}")
    print(f"Precision: {TP/(TP+FP) if (TP+FP)>0 else 0:.3f}")
    print(f"Recall:    {TP/(TP+FN) if (TP+FN)>0 else 0:.3f}")
    print(f"F1 Score:  {(2*TP)/(2*TP + FP + FN) if (2*TP + FP + FN)>0 else 0:.3f}")



# ANSI escape sequences for coloring text
def color_text(text, color):
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'magenta': '\033[95m',
        'cyan': '\033[96m',
        'bold': '\033[1m',
        'underline': '\033[4m',
        'reset': '\033[0m',
    }
    return colors.get(color, '') + text + colors['reset']

def calculate_ier(file_path, exclude_intents=None):
    """
    Calculate Intent Error Rate (IER), optionally excluding any references
    whose intent is in `exclude_intents`.
    """
    exclude_intents = exclude_intents or set()
    samples = defaultdict(dict)
    incomplete_samples = []

    id_pattern     = re.compile(r'^(.*?):')
    intent_pattern = re.compile(r'<([^>]+)>')
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            

            # skip comment / XAI lines
            if not line or line.startswith('#'):
                continue

            m = id_pattern.match(line)
            if not m:
                continue
            sid = m.group(1)

            if 'ref=' in line:
                content = line.split('ref=')[1].strip()
                tokens  = content.split()
                im      = intent_pattern.search(content)
                ref_int = im.group(1) if im else "<null>"
                samples[sid]['ref']        = ref_int
                samples[sid]['ref_tokens'] = tokens

            elif 'hyp=' in line:
                content = line.split('hyp=')[1].strip()
                tokens  = content.split()
                im      = intent_pattern.search(content)
                hyp_int = im.group(1) if im else "<null>"
                samples[sid]['hyp']        = hyp_int
                samples[sid]['hyp_tokens'] = tokens

    total_samples = 0
    error_samples = 0

    for sid, sample in samples.items():
        # skip any sample whose reference intent is in the exclusion set
        if 'ref' in sample and sample['ref'] in exclude_intents:
            continue

        if 'ref' not in sample:
            incomplete_samples.append(sid)
            continue

        # ensure we have a hypothesis
        if 'hyp' not in sample:
            sample['hyp']        = "<null>"
            sample['hyp_tokens'] = ["<null>"]

        total_samples += 1
        if sample['ref'] != sample['hyp']:
            error_samples += 1

    ier = (error_samples / total_samples) * 100 if total_samples else 0.0
    return ier, total_samples, error_samples, samples, incomplete_samples

def main():
    p = argparse.ArgumentParser(description="Compute Intent Error Rate (IER).")
    p.add_argument("file_path", help="Path to the input .txt file.")
    p.add_argument(
        "--exclude_intents",
        type=str,
        default="",
        help="Comma‑separated list of ref‑intents to exclude from IER calculation."
    )
    args = p.parse_args()

    # build exclusion set
    exclude_set = {
        intent.strip()
        for intent in args.exclude_intents.split(',')
        if intent.strip()
    }

    ier, total, errors, samples, incomplete = calculate_ier(
        args.file_path,
        exclude_intents=exclude_set
    )

    print(color_text(f"File: {args.file_path}", "bold"))
    print(color_text(f"Excluded intents: {sorted(exclude_set)}", "magenta"))
    print(color_text(f"Total samples evaluated: {total}", "cyan"))
    print(color_text(f"Number of samples with intent errors: {errors}", "red"))
    print(color_text(f"Intent Error Rate (IER): {ier:.2f}%", "yellow"))
    print()

    if incomplete:
        print(color_text(f"Samples missing reference (skipped): {len(incomplete)}", "red"))
        for sid in incomplete:
            print(f"  {sid}")
        print()

    # build per‑intent counters, skipping excluded intents
    error_counter   = Counter()
    correct_counter = Counter()
    wrong_examples  = defaultdict(dict)

    for sid, sample in samples.items():
        ref = sample.get('ref')
        hyp = sample.get('hyp')
        if ref is None or ref in exclude_set:
            continue

        if ref != hyp:
            error_counter[ref] += 1
            if hyp not in wrong_examples[ref]:
                wrong_examples[ref][hyp] = sid
        else:
            correct_counter[ref] += 1

    print(color_text("Top intents wrongly predicted:", "red"))
    for intent, cnt in error_counter.most_common():
        print(f"  Intent {color_text(intent,'yellow')}: {color_text(str(cnt),'red')} errors")

    print()
    print(color_text("Top intents correctly predicted:", "green"))
    for intent, cnt in correct_counter.most_common():
        print(f"  Intent {color_text(intent,'yellow')}: {color_text(str(cnt),'green')} correct")

    print()
    print(color_text("Breakdown of wrong predictions for top 10 most erroneous intents:", "bold"))

    # Confusion Matrix for Top Intents

    print("Sklearn matrix")
    print_overall_metrics(samples)




    top_wrong_intents = [intent for intent, _ in error_counter.most_common(2)]
    top_correct_intents = [intent for intent, _ in correct_counter.most_common(2)]

    selected_intents = set(top_wrong_intents + top_correct_intents)
    print(color_text("\nBinary confusion matrices for selected intents:", "bold"))
    for intent in selected_intents:
        print_binary_confusion_matrix(samples, intent)

    confusion = build_confusion_matrix(samples, selected_intents)

    print(color_text("\nConfusion Matrix (selected intents):", "bold"))
    sorted_intents = sorted(selected_intents)

    # header row
    header = ["{:>20}".format(" ")] + ["{:>20}".format(hyp) for hyp in sorted_intents]
    print("".join(header))

    for ref in sorted_intents:
        row = ["{:>20}".format(ref)]
        for hyp in sorted_intents:
            count = confusion.get(ref, {}).get(hyp, 0)
            cell = color_text("{:>20}".format(count), "red") if ref != hyp else color_text("{:>20}".format(count), "green")
            row.append(cell)
        print("".join(row))




    for ref_intent, total_err in error_counter.most_common(10):
        print(color_text(f"\nFor reference intent '{ref_intent}' ({total_err} errors):", "cyan"))
        wrong_counter = Counter(
            sample['hyp']
            for sample in samples.values()
            if sample.get('ref') == ref_intent and sample.get('hyp') != ref_intent
        )
        for hyp_intent, cnt in wrong_counter.most_common():
            pct = (cnt / total_err) * 100
            example_id = wrong_examples[ref_intent].get(hyp_intent, "N/A")
            example_str = "No example available."
            if example_id in samples:
                ex = samples[example_id]
                ref_tok_str = str(ex['ref_tokens'])
                hyp_tok_str = str(ex['hyp_tokens'])
                example_str = (
                    f"{color_text(example_id, 'blue')}:  ref={color_text(ref_tok_str, 'yellow')}\n"
                    f"{color_text(example_id, 'blue')}:  hyp={color_text(hyp_tok_str, 'red')}"
                )
            print(f"   Predicted as '{color_text(hyp_intent,'red')}': {cnt} errors ({pct:.2f}%)")
            print(f"      Example:\n      {example_str}")

if __name__ == "__main__":
    main()

