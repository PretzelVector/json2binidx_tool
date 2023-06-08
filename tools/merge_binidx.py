# Takes a list of binidx datasets and merges them into a single one.

import argparse

from indexed_dataset import IndexedDataset, make_builder
from tokenizer import build_tokenizer
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title="input/output")
    group.add_argument(
        "--input",
        nargs="+",
        help="List of binidx files to merge",
        required=True,
    )
    group.add_argument(
        "--output",
        type=str,
        help="Output binidx file",
        required=True,
    )

    group = parser.add_argument_group(title="tokenizer")
    group.add_argument(
        "--tokenizer-type",
        type=str,
        required=True,
        choices=[
            "HFGPT2Tokenizer",
            "HFTokenizer",
            "GPT2BPETokenizer",
            "CharLevelTokenizer",
            "TiktokenTokenizer",
        ],
        help="What type of tokenizer to use.",
    )
    group.add_argument(
        "--vocab-file", type=str, default=None, help="Path to the vocab file"
    )
    group.add_argument(
        "--merge-file",
        type=str,
        default=None,
        help="Path to the BPE merge file (if necessary).",
    )

    args = parser.parse_args()
    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.model_parallel_size = 1

    return args


def main():
    args = get_args()
    tokenizer = build_tokenizer(args)

    # Create the output file that we're going to merge into
    builder = make_builder(
        f"{args.output}.bin", impl="mmap", vocab_size=tokenizer.vocab_size
    )
    for input_file in tqdm(
        args.input, desc="Merging", unit="dataset", total=len(args.input)
    ):
        if not IndexedDataset.exists(input_file):
            print(f"[!] {input_file} does not exist. Skipping.")
            continue

        builder.merge_file_(input_file)
    builder.finalize(f"{args.output}.idx")


if __name__ == "__main__":
    main()
