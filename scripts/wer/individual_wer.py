import argparse
import logging
from collections import defaultdict
from typing import Dict, List, TextIO, Tuple

import kaldialign


def calculate_wer(ref: List[str], hyp: List[str]) -> Tuple[float, int, int, int, int]:
    """Calculates WER for a single utterance.

    Returns:
        Tuple: (wer_rate, insertions, deletions, substitutions, reference_length)
    """
    subs: Dict[Tuple[str, str], int] = defaultdict(int)
    ins: Dict[str, int] = defaultdict(int)
    dels: Dict[str, int] = defaultdict(int)
    num_corr = 0
    ERR = "*"

    ali = kaldialign.align(ref, hyp, ERR)
    for ref_word, hyp_word in ali:
        if ref_word == ERR:
            ins[hyp_word] += 1
        elif hyp_word == ERR:
            dels[ref_word] += 1
        elif hyp_word != ref_word:
            subs[(ref_word, hyp_word)] += 1
        else:
            num_corr += 1

    ref_len = len(ref)
    sub_errs = sum(subs.values())
    ins_errs = sum(ins.values())
    del_errs = sum(dels.values())
    tot_errs = sub_errs + ins_errs + del_errs

    if ref_len == 0:  # Avoid division by zero if reference is empty
        wer_rate = 0.0
    else:
        wer_rate = 100.0 * tot_errs / ref_len

    return wer_rate, ins_errs, del_errs, sub_errs, ref_len


def read_recogs_file(recogs_file: str) -> List[Tuple[str, List[str], List[str]]]:
    """Reads a recogs file and returns a list of tuples."""
    results: List[Tuple[str, List[str], List[str]]] = []
    with open(recogs_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for i in range(0, len(lines), 2):
            ref_line = lines[i].strip()
            hyp_line = lines[i + 1].strip()
            ref_split = ref_line.split(":")
            hyp_split = hyp_line.split(":")
            cut_id = ref_split[0].strip()
            ref_transcript = ref_split[1].replace("ref=[", "").replace("]", "").replace("'", "").strip()
            hyp_transcript = hyp_split[1].replace("hyp=[", "").replace("]", "").replace("'", "").strip()
            ref_words = ref_transcript.split(", ")
            hyp_words = hyp_transcript.split(", ")
            results.append((cut_id, ref_words, hyp_words))
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Calculate individual WER for each audio in a recogs file."
    )
    parser.add_argument(
        "recogs_file",
        type=str,
        help="Path to the recogs file.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="individual_wer_report.txt",
        help="Path to the output file for individual WER statistics.",
    )
    parser.add_argument(
        "--enable_log",
        action="store_true",
        help="Enable logging to the console.",
    )

    args = parser.parse_args()

    if args.enable_log:
        logging.basicConfig(
            format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
            level=logging.INFO,
        )

    logging.info(f"Reading recogs file: {args.recogs_file}")
    results = read_recogs_file(args.recogs_file)

    logging.info(f"Writing individual WER statistics to: {args.output_file}")
    with open(args.output_file, "w", encoding="utf-8") as f:
        for cut_id, ref, hyp in results:
            wer_rate, ins, dels, subs, ref_len = calculate_wer(ref, hyp)

            f.write(f"{cut_id}:\tref={ref}\n")
            f.write(f"{cut_id}:\thyp={hyp}\n")
            f.write(
                f"WER: {wer_rate:.2f}%, "
                f"Insertions: {ins}, Deletions: {dels}, Substitutions: {subs}, "
                f"Reference Length: {ref_len}\n\n"
            )

            if args.enable_log:
                logging.info(
                    f"{cut_id}: WER: {wer_rate:.2f}%, "
                    f"Insertions: {ins}, Deletions: {dels}, Substitutions: {subs}, "
                    f"Reference Length: {ref_len}"
                )

    logging.info("Done!")


if __name__ == "__main__":
    main()