# scripts/generate.py
# Copyright (C) 2025 Langning Chen
#
# This file is part of luoguCaptcha.
#
# This script generates captcha images and saves them as a Hugging Face Dataset
# and also exports to TFRecord format (5000 samples per file).

import subprocess
import threading
from io import BytesIO
from os import path, makedirs
from sys import argv
from PIL import Image
from datasets import Dataset, DatasetDict
from tqdm import tqdm
from queue import Queue, Empty
import numpy as np
import tensorflow as tf
import math

DATA_DIR = "data/luogu_captcha_dataset"
TFRECORD_DIR = "data/luogu_captcha_tfrecord"
CHAR_SIZE = 256
CHARS_PER_LABEL = 4
SAMPLES_PER_TFRECORD = 5000


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def serialize_example(image, label):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    try:
        # Convert image to NumPy array if it's a list or other type
        image_np = np.asarray(image, dtype=np.float32)

        # Verify image shape (should be 35, 90, 1 or 35, 90)
        if len(image_np.shape) == 3 and image_np.shape[-1] == 1:
            image_np = image_np.squeeze(-1)  # Remove channel dimension for PIL
        elif image_np.shape != (35, 90):
            raise ValueError(
                f"Unexpected image shape: {image_np.shape}, expected (35, 90, 1) or (35, 90)"
            )

        # Convert to uint8 for PIL
        image_uint8 = (image_np * 255).astype(np.uint8)
        image_pil = Image.fromarray(image_uint8, mode="L")

        # Save to PNG bytes
        buffer = BytesIO()
        image_pil.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        # Ensure label is a list of integers
        if not isinstance(label, (list, np.ndarray)) or len(label) != CHARS_PER_LABEL:
            raise ValueError(
                f"Unexpected label format: {label}, expected list or array of length {CHARS_PER_LABEL}"
            )
        label_list = label if isinstance(label, list) else label.tolist()

        # Validate that all label elements are integers
        if not all(isinstance(x, (int, np.integer)) for x in label_list):
            raise ValueError(f"Label contains non-integer values: {label_list}")

        feature = {
            "image": _bytes_feature(image_bytes),
            "label": _int64_feature(label_list),
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()

    except Exception as e:
        print(f"Error in serialize_example: {e}")
        raise


def run_subprocess(generate_number, worker_id, result_list, progress_queue):
    """Runs the PHP script to generate captcha images."""
    command = f"php generate.php {generate_number} {worker_id}"
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        shell=True,
    )

    images = []  # 存储 NumPy 数组 (35, 90, 1)
    labels = []  # 存储 NumPy 数组 [int, int, int, int]
    for i in range(generate_number):
        try:
            image_size_bytes = process.stdout.read(2)
            if len(image_size_bytes) < 2:
                print(f"Worker {worker_id}: Incomplete image size read. Ending.")
                break

            label_bytes = process.stdout.read(4)
            if len(label_bytes) < 4:
                print(f"Worker {worker_id}: Incomplete label read. Ending.")
                break

            label_string = label_bytes.decode("utf-8")

            real_size = image_size_bytes[0] * 256 + image_size_bytes[1]
            image_data = process.stdout.read(real_size)
            if len(image_data) < real_size:
                print(f"Worker {worker_id}: Incomplete image data read. Ending.")
                break

            # 核心优化：在生成机上完成所有 CPU 密集型预处理
            image_pil = Image.open(BytesIO(image_data))

            # 1. 图像处理：灰度 -> NumPy -> 归一化 -> 添加通道维度
            image_np = np.array(image_pil.convert("L"), dtype=np.float32) / 255.0
            image_np = np.expand_dims(image_np, axis=-1)
            images.append(image_np)

            # 2. 标签处理：字符串 -> 整数 ASCII 值 NumPy 数组 (Sparse Label)
            label_int_array = np.array([ord(c) for c in label_string], dtype=np.int32)
            labels.append(label_int_array)

            progress_queue.put(1)  # Report progress for each image

        except Exception as e:
            print(f"Worker {worker_id}: Error during generation: {e}")
            break

    result_list.append({"image": images, "label": labels})
    process.stdout.close()
    process.wait()


def write_tfrecords(dataset_dict, tfrecord_dir, samples_per_file=5000):
    """
    Writes Hugging Face DatasetDict (train/test) to TFRecord files.
    Each file contains `samples_per_file` samples.
    """
    if not path.exists(tfrecord_dir):
        makedirs(tfrecord_dir)

    for split_name in ["train", "test"]:
        dataset = dataset_dict[split_name]
        num_samples = len(dataset)
        num_files = math.ceil(num_samples / samples_per_file)

        print(
            f"Writing {split_name} split to TFRecord: {num_samples} samples -> {num_files} files"
        )

        with tqdm(total=num_samples, desc=f"Writing TFRecords ({split_name})") as pbar:
            for file_idx in range(num_files):
                start_idx = file_idx * samples_per_file
                end_idx = min(start_idx + samples_per_file, num_samples)
                filename = path.join(
                    tfrecord_dir, f"{split_name}_part_{file_idx:04d}.tfrecord"
                )

                with tf.io.TFRecordWriter(filename) as writer:
                    for i in range(start_idx, end_idx):
                        try:
                            image = dataset[i]["image"]  # Should be NumPy array or list
                            label = dataset[i]["label"]  # Should be list or NumPy array
                            example = serialize_example(image, label)
                            writer.write(example)
                            pbar.update(1)
                        except Exception as e:
                            print(f"Error writing sample {i} in {split_name}: {e}")
                            continue


if __name__ == "__main__":
    if len(argv) != 3:
        print("Usage: python scripts/generate.py <TotalImages> <WorkersCount>")
        exit(1)

    total_images = int(argv[1])
    workers_count = int(argv[2])

    if not path.exists("data"):
        makedirs("data")

    images_per_worker = total_images // workers_count
    worker_results = []
    worker_threads = []
    progress_queue = Queue()

    print(f"Starting {workers_count} workers to generate {total_images} images...")

    for i in range(workers_count):
        num_to_generate = images_per_worker
        if i == workers_count - 1:
            # Assign the remainder to the last worker
            num_to_generate = total_images - (images_per_worker * (workers_count - 1))

        thread = threading.Thread(
            target=run_subprocess,
            args=(num_to_generate, i, worker_results, progress_queue),
        )
        worker_threads.append(thread)
        thread.start()

    # Progress bar logic
    with tqdm(total=total_images, desc="Generating Captchas") as pbar:
        completed_count = 0
        while completed_count < total_images:
            try:
                # Update bar by 1 for each item from the queue
                pbar.update(progress_queue.get(timeout=1))
                completed_count += 1
            except Empty:
                # If the queue is empty, check if threads are still running
                if not any(t.is_alive() for t in worker_threads):
                    # All threads are done, but maybe not all images were generated
                    pbar.n = completed_count  # Adjust pbar to the actual count
                    pbar.refresh()
                    break  # Exit the loop

    for thread in worker_threads:
        thread.join()

    print("\nAll workers finished. Aggregating results...")

    # Combine results from all workers
    final_images = []
    final_labels = []
    for result in worker_results:
        final_images.extend(result["image"])
        final_labels.extend(result["label"])

    if not final_images:
        print("No images were generated. Exiting.")
        exit(1)

    # Create a Hugging Face Dataset
    full_dataset_dict = {"image": final_images, "label": final_labels}
    full_dataset = Dataset.from_dict(full_dataset_dict)

    # 核心修改：在生成端进行训练/测试分割
    split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)

    # 将其包装成一个 DatasetDict
    dataset_dict = DatasetDict(
        {"train": split_dataset["train"], "test": split_dataset["test"]}
    )

    print(f"Successfully generated {len(full_dataset)} images.")
    print(f"Saving Hugging Face dataset to '{DATA_DIR}'...")

    # Save the DatasetDict to disk
    dataset_dict.save_to_disk(DATA_DIR)
    print("Hugging Face Dataset saved successfully.")

    # === 新增：导出为 TFRecord 格式 ===
    print(f"Saving dataset as TFRecord to '{TFRECORD_DIR}' (5000 samples per file)...")
    try:
        write_tfrecords(
            dataset_dict, TFRECORD_DIR, samples_per_file=SAMPLES_PER_TFRECORD
        )
        print("TFRecord export completed.")
    except Exception as e:
        print(f"Failed to export TFRecords: {e}")

    print(
        f"Run `python scripts/huggingface.py upload_dataset {DATA_DIR}` to upload Hugging Face dataset."
    )
    print(f"TFRecord files are saved in '{TFRECORD_DIR}'.")
