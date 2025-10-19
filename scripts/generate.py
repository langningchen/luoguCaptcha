# Copyright (C) 2025 Langning Chen
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

DATA_DIR = "data/luogu_captcha_dataset"
CHAR_SIZE = 256
CHARS_PER_LABEL = 4


def run_subprocess(generate_number, worker_id, result_list, progress_queue):
    """Runs the PHP script to generate captcha images."""
    command = f"php generate.php {generate_number}"
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
    print(f"Saving dataset to '{DATA_DIR}'...")

    # Save the DatasetDict to disk
    dataset_dict.save_to_disk(DATA_DIR)

    print("Dataset saved successfully.")
    print(f"Run `python scripts/huggingface.py upload_dataset {DATA_DIR}` to upload.")
