# Note: following icefall's fbank creation script, but modified to work with SLURP dataset

"""
This file computes fbank features for the SLURP dataset.
It looks for manifests in the directory data/manifests.

The generated fbank features are saved in data/fbank.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import sentencepiece as spm
import torch
# from filter_cuts import filter_cuts
from lhotse import CutSet, Fbank, FbankConfig, LilcomChunkyWriter
from lhotse.recipes.utils import read_manifests_if_cached

from datetime import datetime
import subprocess
import torch.distributed as dist
# from typing import PathLike
# from lhotse import Cut
from lhotse.cut import Cut

from pathlib import Path
from typing import Optional
import os
from contextlib import contextmanager



torch.set_num_threads(1)
torch.set_num_interop_threads(1)

@contextmanager
def get_executor():
    try:
        
        name = subprocess.check_output("hostname -f", shell=True, text=True)
        if name.strip().endswith(".clsp.jhu.edu"):
            import plz
            from distributed import Client

            with plz.setup_cluster() as cluster:
                cluster.scale(80)
                yield Client(cluster)
            return
    except Exception:
        pass
    
    yield None


def str2bool(v):
    """Used in argparse.ArgumentParser.add_argument to indicate
    that a type is a bool type and user can enter

        - yes, true, t, y, 1, to represent True
        - no, false, f, n, 0, to represent False

    See https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse  # noqa
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def setup_logger(
    # log_filename: Pathlike,
    log_filename: Path,
    log_level: str = "info",
    use_console: bool = True,
) -> None:
    """Setup log level.

    Args:
      log_filename:
        The filename to save the log.
      log_level:
        The log level to use, e.g., "debug", "info", "warning", "error",
        "critical"
      use_console:
        True to also print logs to console.
    """
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        formatter = f"%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] ({rank}/{world_size}) %(message)s"  # noqa
        log_filename = f"{log_filename}-{date_time}-{rank}"
    else:
        formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
        log_filename = f"{log_filename}-{date_time}"

    os.makedirs(os.path.dirname(log_filename), exist_ok=True)

    level = logging.ERROR
    if log_level == "debug":
        level = logging.DEBUG
    elif log_level == "info":
        level = logging.INFO
    elif log_level == "warning":
        level = logging.WARNING
    elif log_level == "critical":
        level = logging.CRITICAL

    logging.basicConfig(
        filename=log_filename,
        format=formatter,
        level=level,
        filemode="w",
        force=True,
    )
    if use_console:
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(logging.Formatter(formatter))
        logging.getLogger("").addHandler(console)


class AttributeDict(dict):
    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(f"No such attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        if key in self:
            del self[key]
            return
        raise AttributeError(f"No such attribute '{key}'")



def filter_cuts(cut_set: CutSet, sp: spm.SentencePieceProcessor):
    total = 0  # number of total utterances before removal
    removed = 0  # number of removed utterances

    def remove_short_and_long_utterances(c: Cut):
        """Return False to exclude the input cut"""
        nonlocal removed, total
        # Keep only utterances with duration between 1 second and 20 seconds
        #
        # Caution: There is a reason to select 20.0 here. Please see
        # ./display_manifest_statistics.py
        #
        # You should use ./display_manifest_statistics.py to get
        # an utterance duration distribution for your dataset to select
        # the threshold
        total += 1
        if c.duration < 1.0 or c.duration > 20.0:
            logging.warning(
                f"Exclude cut with ID {c.id} from training. Duration: {c.duration}"
            )
            removed += 1
            return False

        # In pruned RNN-T, we require that T >= S
        # where T is the number of feature frames after subsampling
        # and S is the number of tokens in the utterance

        # In ./pruned_transducer_stateless2/conformer.py, the
        # conv module uses the following expression
        # for subsampling
        if c.num_frames is None:
            num_frames = c.duration * 100  # approximate
        else:
            num_frames = c.num_frames

        T = ((num_frames - 1) // 2 - 1) // 2
        # Note: for ./lstm_transducer_stateless/lstm.py, the formula is
        #  T = ((num_frames - 3) // 2 - 1) // 2

        # Note: for ./pruned_transducer_stateless7/zipformer.py, the formula is
        # T = ((num_frames - 7) // 2 + 1) // 2

        tokens = sp.encode(c.supervisions[0].text, out_type=str)

        if T < len(tokens):
            logging.warning(
                f"Exclude cut with ID {c.id} from training. "
                f"Number of frames (before subsampling): {c.num_frames}. "
                f"Number of frames (after subsampling): {T}. "
                f"Text: {c.supervisions[0].text}. "
                f"Tokens: {tokens}. "
                f"Number of tokens: {len(tokens)}"
            )
            removed += 1
            return False

        return True

    # We use to_eager() here so that we can print out the value of total
    # and removed below.
    ans = cut_set.filter(remove_short_and_long_utterances).to_eager()
    ratio = removed / total * 100
    logging.info(
        f"Removed {removed} cuts from {total} cuts. {ratio:.3f}% data is removed."
    )
    return ans



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bpe-model",
        type=str,
        help="""Path to the bpe.model. If not None, we will remove short and
        long utterances before extracting features""",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="""Dataset parts to compute fbank. If None, we will use all""",
    )

    parser.add_argument(
        "--perturb-speed",
        type=str2bool,
        default=True,
        help="""Perturb speed with factor 0.9 and 1.1 on train subset.""",
    )

    return parser.parse_args()


def compute_fbank_slurp(
    bpe_model: Optional[str] = None,
    dataset: Optional[str] = None,
    perturb_speed: Optional[bool] = True,

    # bpe_model: Optional[str] = "/disk1/polaris_intent_detection/Tokenizer_SLURP/models_LIBRI_2000_INTENT/libri_bpe_2000_intent.model",
    # dataset: Optional[str] = None,
    # perturb_speed: Optional[bool] = True,
):
    src_dir = Path("/disk1/polaris_intent_detection/Prepare_SLURP/data/manifests")
    output_dir = Path("/disk1/polaris_intent_detection/Prepare_SLURP/data/fbank")
    num_jobs = min(15, os.cpu_count())
    num_mel_bins = 80

    if bpe_model:
        logging.info(f"Loading {bpe_model}")
        sp = spm.SentencePieceProcessor()
        sp.load(bpe_model)

    if dataset is None:
        dataset_parts = (
            "all"
        )
    else:
        dataset_parts = dataset.split(" ", -1)

    prefix = "slurp" 
    suffix = "jsonl" 
    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=src_dir,
        prefix=prefix,
        suffix=suffix,
    )

    print ("MANIFESTS", manifests)

    assert manifests is not None

    print("len(manifests) - ", len(manifests))
    print("len(dataset_parts) - ", len(dataset_parts))

    assert len(manifests) == len(dataset_parts), (
        len(manifests),
        len(dataset_parts),
        list(manifests.keys()),
        dataset_parts,
    )

    extractor = Fbank(FbankConfig(num_mel_bins=num_mel_bins))

    with get_executor() as ex:  # Initialize the executor only once.
        for partition, m in manifests.items():
            cuts_filename = f"{prefix}_cuts_{partition}.{suffix}"
            if (output_dir / cuts_filename).is_file():
                logging.info(f"{partition} already exists - skipping.")
                continue
            logging.info(f"Processing {partition}")
            cut_set = CutSet.from_manifests(
                recordings=m["recordings"],
                supervisions=m["supervisions"],
            )

            if "train" in partition:
                if bpe_model:
                    cut_set = filter_cuts(cut_set, sp)
                if perturb_speed:
                    logging.info(f"Doing speed perturb")
                    cut_set = (
                        cut_set
                        + cut_set.perturb_speed(0.9)
                        + cut_set.perturb_speed(1.1)
                    )
            cut_set = cut_set.compute_and_store_features(
                extractor=extractor,
                storage_path=f"{output_dir}/{prefix}_feats_{partition}",
                # when an executor is specified, make more partitions
                num_jobs=num_jobs if ex is None else 80,
                executor=ex,
                storage_type=LilcomChunkyWriter,
            )
            cut_set.to_file(output_dir / cuts_filename)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    args = get_args()
    logging.info(vars(args))
    compute_fbank_slurp(
        bpe_model=args.bpe_model,
        dataset=args.dataset,
        perturb_speed=args.perturb_speed,
    )
