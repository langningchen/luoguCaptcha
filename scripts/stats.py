import os
import sys
from collections import Counter
import numpy as np
from datasets import load_dataset, load_from_disk

# 假设你的数据集 ID 和形状
DATASET_PATH_HUB = "langningchen/luogu-captcha-dataset"
DATASET_PATH_LOCAL = "data/luogu_captcha_dataset"
CHARS_PER_LABEL = 4  # 验证码字符数


def get_decoded_label_batch(labels_batch):
    """
    向量化函数：将一批稀疏标签 (ASCII codes 的列表或数组) 转换为可读字符列表。
    确保返回的列表长度与输入的批次大小完全一致。
    """
    decoded_labels = []

    # labels_batch 是一个包含多个样本标签的列表。
    # 它的长度就是当前的批次大小 (例如 1000)。
    for label_data in labels_batch["label"]:  # 注意：这里要从字典中取出 'label' 列表
        try:
            # 确保是 NumPy 数组
            label_np = np.array(label_data)

            # 确保 label 是一个一维数组 (移除任何多余维度)
            if label_np.ndim > 1:
                label_np = np.squeeze(label_np)

            # 转换为 Python 列表，然后使用 map(chr, ...) 解码
            decoded_label = "".join(map(chr, label_np.tolist()))

            if len(decoded_label) == CHARS_PER_LABEL:
                decoded_labels.append(decoded_label)
            else:
                # 即使长度不符合预期，也要占位 None
                decoded_labels.append(None)

        except Exception:
            # 捕获任何错误，并占位 None
            decoded_labels.append(None)

    return {"decoded_label": decoded_labels}


def count_labels(source_path, from_hub=True, split="train", top_n=10, batch_size=1000):
    """
    加载数据集并统计验证码答案的出现频率 (使用向量化优化)。

    Args:
        source_path (str): 数据集路径 (Hub ID 或本地路径)。
        from_hub (bool): 是否从 Hugging Face Hub 加载。
        split (str): 要统计的数据集分割。
        top_n (int): 显示最常见的 N 个标签。
        batch_size (int): 向量化处理的批次大小。
    """
    print("=" * 70)
    print(
        f"Starting label count for split: '{split}' (Optimized with map/batched=True)"
    )

    # 1. 加载数据集
    try:
        if from_hub:
            print(f"Loading dataset from Hugging Face Hub: {source_path}")
            dataset_dict = load_dataset(source_path)
        else:
            print(f"Loading dataset from local disk: {source_path}")
            dataset_dict = load_from_disk(source_path)

        if split not in dataset_dict:
            print(
                f"Error: Split '{split}' not found in dataset. Available splits: {list(dataset_dict.keys())}"
            )
            return

        dataset = dataset_dict[split]

    except Exception as e:
        print(f"Error loading dataset from {source_path}: {e}")
        print("Please check the path/ID and ensure the dataset exists.")
        return

    print(f"Dataset loaded. Total samples in '{split}' split: {len(dataset)}")

    # 2. 向量化解码标签并统计
    print(f"Vectorized decoding and counting labels (batch_size={batch_size})...")

    # 使用 map 批量处理
    # remove_columns=['image', 'label'] 可以节省内存，因为我们只需要新的 'decoded_label' 列
    # num_proc 可以使用多核并行加速，但对于 I/O 密集型操作（如从 disk/hub 读取），效果可能不明显
    mapped_dataset = dataset.map(
        get_decoded_label_batch,
        batched=True,
        batch_size=batch_size,
        remove_columns=[col for col in dataset.column_names if col != "label"],
        load_from_cache_file=False,  # 确保重新计算
    )

    # 3. 统计结果

    # 从新的 'decoded_label' 列中提取所有有效的标签
    all_labels = [
        label for label in mapped_dataset["decoded_label"] if label is not None
    ]

    label_counts = Counter(all_labels)

    total_unique_labels = len(label_counts)
    total_counted_samples = sum(label_counts.values())

    print("\n" + "=" * 70)
    print("Label Statistics Summary")
    print(f"Total samples processed: {len(dataset)}")
    print(f"Total valid samples counted: {total_counted_samples}")
    print(f"Total unique captcha labels: {total_unique_labels}")
    print("-" * 70)

    print(f"Top {min(top_n, total_unique_labels)} Most Common Captcha Labels:")

    # 获取最常见的 N 个标签
    most_common = label_counts.most_common(top_n)

    # 格式化输出
    max_label_len = max([len(label) for label, count in most_common] or [5])

    for label, count in most_common:
        # 使用 f-string 的对齐功能
        print(
            f"  Captcha: {label:<{max_label_len}} | Count: {count:>6} | Frequency: {count/total_counted_samples*100:.4f}%"
        )

    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python scripts/stats_optimized.py <local|hub> [split] [top_n] [batch_size]"
        )
        print(
            "\nExample 1 (Local, test split, top 5, batch 5000): python scripts/stats_optimized.py local test 5 5000"
        )
        print(
            "Example 2 (Hub, train split, top 10, batch 1000): python scripts/stats_optimized.py hub train 10 1000"
        )
        sys.exit(1)

    source_type = sys.argv[1].lower()
    split = sys.argv[2] if len(sys.argv) > 2 else "train"
    top_n = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    batch_size = int(sys.argv[4]) if len(sys.argv) > 4 else 1000

    if source_type == "local":
        count_labels(
            DATASET_PATH_LOCAL,
            from_hub=False,
            split=split,
            top_n=top_n,
            batch_size=batch_size,
        )
    elif source_type == "hub":
        count_labels(
            DATASET_PATH_HUB,
            from_hub=True,
            split=split,
            top_n=top_n,
            batch_size=batch_size,
        )
    else:
        print("Invalid source type. Use 'local' or 'hub'.")
