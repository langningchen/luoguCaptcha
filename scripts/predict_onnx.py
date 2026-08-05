#!/usr/bin/env python3
"""Run Luogu captcha inference with an exported ONNX model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


ALPHABET = "abcdefghijklmnopqrstuvwxyz123456789"
IMAGE_HEIGHT = 35
IMAGE_WIDTH = 90


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/browser/onnx/luogu-captcha-fp32.onnx"),
    )
    return parser.parse_args()


def preprocess(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        return np.asarray(image, dtype=np.float32)[None, ...] / 255.0


def main() -> None:
    args = parse_args()
    session = ort.InferenceSession(
        str(args.model), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    logits = session.run(
        [output_name], {input_name: preprocess(args.image)}
    )[0]
    indices = np.argmax(logits, axis=-1)[0]
    print("".join(ALPHABET[index] for index in indices))


if __name__ == "__main__":
    main()
