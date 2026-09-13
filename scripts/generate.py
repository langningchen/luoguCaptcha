import os
import subprocess
import threading
from os import makedirs, path
from sys import argv
import tensorflow as tf
from tqdm import tqdm

TFRECORD_DIR = "data"
SAMPLES_PER_TFRECORD = 5000


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def fast_serialize(jpeg_bytes, label_bytes):
    label_list = [ord(c) for c in label_bytes.decode("utf-8")]
    feature = {
        "image": _bytes_feature(jpeg_bytes),
        "label": _int64_feature(label_list),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()


class GlobalFileIndex:
    """原子计数器：为各个 Worker 分发全局唯一的分片文件编号"""
    def __init__(self):
        self._count = 0
        self._lock = threading.Lock()

    def get_next_id(self):
        with self._lock:
            idx = self._count
            self._count += 1
            return idx


def direct_worker(num_to_gen, worker_id, file_index_mgr, pbar):
    """
    直通 Worker：
    PHP 进程 -> 管道读取 -> 当前线程并行序列化 -> 当前线程独立落盘
    彻底消除全局 Queue 锁竞争！
    """
    # 启用 PHP 8 JIT 参数（如果是纯数值密集运算，JIT 提速显著）
    command = (
        f"php -d opcache.enable_cli=1 -d opcache.jit=tracing -d opcache.jit_buffer_size=64M "
        f"generate.php {num_to_gen} {worker_id}"
    )

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        shell=True,
        bufsize=131072  # 128KB 管道缓冲
    )

    stdout = process.stdout
    writer = None
    samples_in_file = 0
    batch_update = 0

    try:
        for _ in range(num_to_gen):
            header = stdout.read(6)
            if len(header) < 6:
                break

            real_size = (header[0] << 8) | header[1]
            label_bytes = header[2:6]

            jpeg_data = stdout.read(real_size)
            if len(jpeg_data) < real_size:
                break

            # 轮转当前 Worker 负责的分片
            if writer is None or samples_in_file >= SAMPLES_PER_TFRECORD:
                if writer:
                    writer.close()
                file_idx = file_index_mgr.get_next_id()
                filepath = path.join(TFRECORD_DIR, f"part_{file_idx:05d}.tfrecord")
                writer = tf.io.TFRecordWriter(filepath)
                samples_in_file = 0

            # 独立在当前核并行执行 Proto 序列化和磁盘写入
            example_str = fast_serialize(jpeg_data, label_bytes)
            writer.write(example_str)
            samples_in_file += 1

            # 批量刷新 tqdm 进度条，避免频繁加锁
            batch_update += 1
            if batch_update >= 50:
                pbar.update(batch_update)
                batch_update = 0

        if batch_update > 0:
            pbar.update(batch_update)

    finally:
        if writer:
            writer.close()
        process.stdout.close()
        process.wait()


def main():
    total_images = int(argv[1]) if len(argv) > 1 else 1000000
    nproc = os.cpu_count() or 4
    if len(argv) > 2:
        nproc = int(argv[2])

    if not path.exists(TFRECORD_DIR):
        makedirs(TFRECORD_DIR)

    print(f"Total images: {total_images} | Workers (Parallel Pipelines): {nproc}")
    print(f"Architecture: Direct Sharded Pipelines (Zero Lock / JIT Enabled)\n")

    file_index_mgr = GlobalFileIndex()
    pbar = tqdm(total=total_images, desc="Generating Dataset")

    # 全核全部投入直通生成与写入
    threads = []
    base_chunk = total_images // nproc
    remainder = total_images % nproc

    for i in range(nproc):
        count = base_chunk + (1 if i < remainder else 0)
        if count <= 0:
            continue
        t = threading.Thread(
            target=direct_worker,
            args=(count, i, file_index_mgr, pbar),
            daemon=True
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    pbar.close()
    print(f"\n[Done] All {total_images} samples saved to '{TFRECORD_DIR}'.")


if __name__ == "__main__":
    main()
