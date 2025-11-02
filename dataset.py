#!/usr/bin/env python3
"""
dataset.py

Download common NLP datasets (via the Hugging Face `datasets` library)
and save each split into a local `dataset/` folder as JSONL files.

This script intentionally uses the public `datasets` API so it downloads
data from the internet and stores it locally. It's configurable via
command-line flags to select datasets, limit rows for quick testing,
and choose the output directory/format.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Iterable

try:
    from datasets import load_dataset, DatasetDict
except Exception as e:
    print("Required package 'datasets' is missing. Please install requirements.txt and try again.")
    raise


DEFAULT_DATASETS: Dict[str, str] = {
    # key: friendly name, value: HF dataset id
    "go_emotions": "go_emotions",
    "emotion": "emotion",  # Alternative emotion dataset
    "daily_dialog": "daily_dialog",  # Conversational dataset
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_jsonl(dataset, outpath: str) -> None:
    """Save a Hugging Face `Dataset` to a JSONL file one record per line."""
    ensure_dir(os.path.dirname(outpath) or ".")
    with open(outpath, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def download_and_save(dataset_id: str, outdir: str, splits: Iterable[str] | None = None, limit: int | None = None) -> None:
    """Load `dataset_id` using `datasets.load_dataset` and save splits to `outdir/dataset_id/`.

    Each split will be written to `<outdir>/<dataset_id>/<split>.jsonl`.
    If `limit` is provided, only the first `limit` rows of each split will be written.
    """
    print(f"Loading dataset '{dataset_id}' from Hugging Face...")
    try:
        ds = load_dataset(dataset_id)
    except UnicodeDecodeError as e:
        print(f"  Unicode error with '{dataset_id}': {e}")
        print("  Trying to load with streaming=True...")
        try:
            ds = load_dataset(dataset_id, streaming=True)
            # Convert streaming dataset to regular dataset for processing
            if isinstance(ds, dict):
                ds = {split: list(split_ds) for split, split_ds in ds.items()}
            else:
                ds = list(ds)
        except Exception as e2:
            print(f"  Failed to load '{dataset_id}' even with streaming: {e2}")
            return

    if isinstance(ds, DatasetDict):
        for split_name, split_ds in ds.items():
            if splits and split_name not in splits:
                continue
            print(f"  - saving split '{split_name}' (rows={len(split_ds)})...")
            subset = split_ds
            if limit is not None:
                subset = split_ds.select(range(min(limit, len(split_ds))))
            outpath = os.path.join(outdir, dataset_id, f"{split_name}.jsonl")
            save_jsonl(subset, outpath)
            print(f"    saved -> {outpath}")
    else:
        # single split dataset
        print(f"  - saving dataset (rows={len(ds)})...")
        subset = ds
        if limit is not None:
            subset = ds.select(range(min(limit, len(ds))))
        outpath = os.path.join(outdir, dataset_id + ".jsonl")
        save_jsonl(subset, outpath)
        print(f"    saved -> {outpath}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download datasets and save them locally as JSONL files.")
    parser.add_argument("--datasets", "-d", type=str, default="",
                        help="Comma-separated list of dataset ids or keys from the default list. If empty, nothing is downloaded unless --all is set.")
    parser.add_argument("--all", action="store_true", help="Download all default datasets")
    parser.add_argument("--outdir", "-o", type=str, default="dataset", help="Output directory to save datasets")
    parser.add_argument("--limit", "-n", type=int, default=None, help="Limit number of rows per split (useful for testing)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    to_download = []
    if args.all:
        to_download = list(DEFAULT_DATASETS.values())
    elif args.datasets:
        # Accept either friendly keys (e.g. 'go_emotions') or raw HF ids
        parts = [p.strip() for p in args.datasets.split(",") if p.strip()]
        for p in parts:
            if p in DEFAULT_DATASETS:
                to_download.append(DEFAULT_DATASETS[p])
            else:
                to_download.append(p)
    else:
        print("No datasets selected. Use --all or --datasets to choose datasets to download.")
        return 0

    ensure_dir(args.outdir)

    for ds in to_download:
        try:
            download_and_save(ds, args.outdir, limit=args.limit)
        except Exception as e:
            print(f"Error downloading '{ds}': {e}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
