# Copyright (C) 2025 zzsqjdhqgb
#
# This file is part of luoguCaptcha.
#
# luoguCaptcha is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# luoguCaptcha is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with luoguCaptcha.  If not, see <https://www.gnu.org/licenses/>.

"""
从 Hugging Face Hub 拉取数据集，然后转换为 TFRecord 格式。

Usage:
  python download_and_convert.py [--repo-id REPO_ID] [--tfrecord-dir DIR] [--samples-per-file N]

Examples:
  python download_and_convert.py
  python download_and_convert.py --repo-id langningchen/luogu-captcha-dataset --tfrecord-dir data/luogu_captcha_tfrecord --samples-per-file 5000
"""

import argparse
import math
import os
from io import BytesIO


import numpy as np
import tensorflow as tf
from datasets import load_dataset, DatasetDict
from PIL import Image
from tqdm import tqdm

# ── 常量（与 generate.py 保持一致） ──────────────────────────────
DATASET_REPO_ID = "langningchen/luogu-captcha-dataset"
TFRECORD_DIR = "data/luogu_captcha_tfrecord"
CHAR_SIZE = 256
CHARS_PER_LABEL = 4
SAMPLES_PER_TFRECORD = 5000
IMG_HEIGHT, IMG_WIDTH = 35, 90


# ── TFRecord 序列化（照抄 generate.py） ─────────────────────────
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def serialize_example(image, label):
    """
    Creates a tf.train.Example message ready to be written to a file.
    image: NumPy array (H, W, 1) or (H, W), float32 in [0, 1]
    label: list/array of CHARS_PER_LABEL ints (ASCII codes)
    """
    image_np = np.asarray(image, dtype=np.float32)

    # 统一为 (H, W)
    if image_np.ndim == 3 and image_np.shape[-1] == 1:
        image_np = image_np.squeeze(-1)
    if image_np.shape != (IMG_HEIGHT, IMG_WIDTH):
        raise ValueError(
            f"Unexpected image shape: {image_np.shape}, expected ({IMG_HEIGHT}, {IMG_WIDTH})"
        )

    # 转 uint8 → PNG bytes
    image_uint8 = (image_np * 255).astype(np.uint8)
    image_pil = Image.fromarray(image_uint8, mode="L")
    buffer = BytesIO()
    image_pil.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    # 标签
    label_list = list(label) if not isinstance(label, list) else label
    if len(label_list) != CHARS_PER_LABEL:
        raise ValueError(
            f"Unexpected label length: {len(label_list)}, expected {CHARS_PER_LABEL}"
        )

    feature = {
        "image": _bytes_feature(image_bytes),
        "label": _int64_feature(label_list),
    }
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature)
    )
    return example_proto.SerializeToString()


# ── 写 TFRecord（照抄 generate.py） ─────────────────────────────
def write_tfrecords(dataset_dict: DatasetDict, tfrecord_dir: str, samples_per_file: int):
    """
    Writes Hugging Face DatasetDict (train/test) to TFRecord files.
    Each file contains at most `samples_per_file` samples.
    """
    os.makedirs(tfrecord_dir, exist_ok=True)

    for split_name in dataset_dict.keys():
        dataset = dataset_dict[split_name]
        num_samples = len(dataset)
        num_files = math.ceil(num_samples / samples_per_file)

        print(
            f"\nWriting '{split_name}' split: {num_samples} samples → {num_files} TFRecord file(s)"
        )

        with tqdm(total=num_samples, desc=f"TFRecord ({split_name})") as pbar:
            for file_idx in range(num_files):
                start_idx = file_idx * samples_per_file
                end_idx = min(start_idx + samples_per_file, num_samples)
                filename = os.path.join(
                    tfrecord_dir, f"{split_name}_part_{file_idx:04d}.tfrecord"
                )

                with tf.io.TFRecordWriter(filename) as writer:
                    for i in range(start_idx, end_idx):
                        try:
                            sample = dataset[i]
                            image = np.asarray(sample["image"], dtype=np.float32)
                            label = sample["label"]

                            # Hub 上的数据可能已经是 [0,1] float，也可能是 [0,255] uint8
                            # 如果最大值 > 1，认为还没归一化
                            if image.max() > 1.0:
                                image = image / 255.0

                            # 确保有 channel 维度以便 squeeze
                            if image.ndim == 3 and image.shape[-1] == 1:
                                pass  # serialize_example 内部会 squeeze
                            elif image.ndim == 2:
                                pass  # 已经是 (H, W)

                            label_list = (
                                label.tolist()
                                if isinstance(label, np.ndarray)
                                else list(label)
                            )

                            example = serialize_example(image, label_list)
                            writer.write(example)
                        except Exception as e:
                            print(f"  ⚠ Skipped sample {i} in '{split_name}': {e}")
                            continue
                        pbar.update(1)


# ── 主流程 ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Download dataset from Hugging Face Hub and convert to TFRecord."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DATASET_REPO_ID,
        help=f"Hugging Face dataset repo ID (default: {DATASET_REPO_ID})",
    )
    parser.add_argument(
        "--tfrecord-dir",
        type=str,
        default=TFRECORD_DIR,
        help=f"Output directory for TFRecord files (default: {TFRECORD_DIR})",
    )
    parser.add_argument(
        "--samples-per-file",
        type=int,
        default=SAMPLES_PER_TFRECORD,
        help=f"Max samples per TFRecord file (default: {SAMPLES_PER_TFRECORD})",
    )
    args = parser.parse_args()

    # 1. 从 Hugging Face Hub 拉取数据集
    print(f"Downloading dataset from Hugging Face Hub: {args.repo_id} ...")
    try:
        dataset_dict = load_dataset(args.repo_id)
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Please check the repo ID and your network/authentication.")
        return

    # 打印基本信息
    for split_name, ds in dataset_dict.items():
        print(f"  Split '{split_name}': {len(ds)} samples")

    # 2. 转换为 TFRecord
    print(f"\nConverting to TFRecord (output: {args.tfrecord_dir}) ...")
    write_tfrecords(dataset_dict, args.tfrecord_dir, args.samples_per_file)

    print("\n✅ Done! TFRecord files saved to:", args.tfrecord_dir)
    print("You can now run train.py or train.v2.py to train the model.")


if __name__ == "__main__":
    main()