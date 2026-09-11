#!/usr/bin/env python3
"""Publish the dataset or browser model artifacts to Hugging Face Hub."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from datasets import load_from_disk
from huggingface_hub import HfApi


DATASET_REPO_ID = "langningchen/luogu-captcha-dataset"
MODEL_REPO_ID = "langningchen/luogu-captcha-model"


def get_api() -> HfApi:
    # HfApi falls back to the locally stored token when HF_TOKEN is unset.
    return HfApi(token=os.environ.get("HF_TOKEN"))


def ensure_repo(api: HfApi, repo_id: str, repo_type: str) -> None:
    api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)


def upload_dataset(local_path: Path) -> None:
    if not local_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {local_path}")

    dataset = load_from_disk(str(local_path))
    if not hasattr(dataset, "push_to_hub"):
        raise TypeError("Expected a Dataset or DatasetDict loaded from disk")

    api = get_api()
    ensure_repo(api, DATASET_REPO_ID, "dataset")
    dataset.push_to_hub(DATASET_REPO_ID)
    print(f"Published dataset: https://huggingface.co/datasets/{DATASET_REPO_ID}")


def upload_model(local_path: Path) -> None:
    if not local_path.exists():
        raise FileNotFoundError(f"Model path not found: {local_path}")

    api = get_api()
    ensure_repo(api, MODEL_REPO_ID, "model")
    if local_path.is_dir():
        api.upload_folder(
            folder_path=str(local_path),
            repo_id=MODEL_REPO_ID,
            repo_type="model",
            commit_message="Publish validated browser model artifacts",
        )
    else:
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=f"keras/{local_path.name}",
            repo_id=MODEL_REPO_ID,
            repo_type="model",
            commit_message="Publish Keras model artifact",
        )
    print(f"Published model: https://huggingface.co/{MODEL_REPO_ID}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    dataset_parser = subparsers.add_parser("upload_dataset")
    dataset_parser.add_argument("path", type=Path)

    model_parser = subparsers.add_parser("upload_model")
    model_parser.add_argument(
        "path",
        type=Path,
        help="A model artifact directory or a single Keras model file",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.command == "upload_dataset":
            upload_dataset(args.path)
        else:
            upload_model(args.path)
    except Exception as error:
        print(f"Hugging Face upload failed: {error}", file=sys.stderr)
        print(
            "Set HF_TOKEN to a write token or run `hf auth login` before uploading.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
