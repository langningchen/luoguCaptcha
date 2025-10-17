# scripts/preview.py
import os
import sys
import numpy as np
from datasets import load_dataset, load_from_disk
from matplotlib import pyplot as plt

# 假设你的数据集 ID 和形状
DATASET_PATH_HUB = "langningchen/luogu-captcha-dataset" 
DATASET_PATH_LOCAL = "data/luogu_captcha_dataset"
IMG_HEIGHT, IMG_WIDTH = 35, 90
CHARS_PER_LABEL = 4

def preview_dataset(source_path, from_hub=True, num_samples=5):
    """
    加载数据集并预览前 N 个样本的图像和标签。
    
    Args:
        source_path (str): 数据集路径 (Hub ID 或本地路径)。
        from_hub (bool): 是否从 Hugging Face Hub 加载。
        num_samples (int): 预览的样本数量。
    """
    try:
        if from_hub:
            print(f"Loading dataset from Hugging Face Hub: {source_path}")
            # 加载数据集（假设为 DatasetDict）
            dataset_dict = load_dataset(source_path)
            # 默认预览 'train' 分割
            dataset = dataset_dict["train"]
        else:
            print(f"Loading dataset from local disk: {source_path}")
            # 加载本地数据集（假设为 DatasetDict）
            dataset_dict = load_from_disk(source_path)
            # 默认预览 'train' 分割
            dataset = dataset_dict["train"]
            
    except Exception as e:
        print(f"Error loading dataset from {source_path}: {e}")
        print("Please check the path/ID and ensure the dataset exists.")
        return

    print(f"Dataset loaded. Total samples in 'train' split: {len(dataset)}")
    print(f"Displaying first {num_samples} samples (Image is pre-processed NumPy Array).")
    
    # 确保样本数量不超过数据集大小
    num_samples = min(num_samples, len(dataset))

    # 创建一个图表来显示图像
    fig, axes = plt.subplots(num_samples, 1, figsize=(6, 1.5 * num_samples))
    if num_samples == 1: # 确保 axes 始终是数组
        axes = [axes]

    for i in range(num_samples):
        # 获取样本
        sample = dataset[i]
        
        # 核心修改 1: 确保 image 和 label 是 NumPy 数组
        image_np = np.array(sample["image"])
        label_np = np.array(sample["label"])
        
        # 1. 验证图像形状和类型
        # 注意：如果 image_np 是 (35, 90, 1) 的数组，且被包裹在列表中，
        # 转换后可能形状会增加一个维度，例如 (1, 35, 90, 1)。
        # 这里应该只检查最后一个维度是否是 1
        
        # 移除任何多余的 Batch/List 维度，只保留 (H, W, C)
        while image_np.ndim > 3:
             image_np = np.squeeze(image_np, axis=0)
             
        if image_np.shape != (IMG_HEIGHT, IMG_WIDTH, 1):
             print(f"\nWarning: Sample {i} image shape is {image_np.shape}, expected ({IMG_HEIGHT}, {IMG_WIDTH}, 1). Adjusting...")
             
        # 确保 label 是一维数组 (4,)
        if label_np.ndim > 1:
            label_np = np.squeeze(label_np) # 移除任何额外的维度
             
        if label_np.shape != (CHARS_PER_LABEL,):
             print(f"\nWarning: Sample {i} label shape is {label_np.shape}, expected ({CHARS_PER_LABEL},). Adjusting...")
        
        # 2. 将 Sparse Label (ASCII codes) 转换为可读字符
        try:
            label_chars = "".join(map(chr, label_np.tolist()))
        except ValueError:
            label_chars = "Decoding Error (Not valid ASCII)"

        # 3. 显示图像
        ax = axes[i]
        # 移除单通道维度以供 matplotlib 显示 (35, 90, 1) -> (35, 90)
        display_img = np.squeeze(image_np) 
        
        ax.imshow(display_img, cmap='gray')
        ax.set_title(f"Sample {i} | Label (Sparse): {label_np} | Decoded: '{label_chars}'", fontsize=10)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/preview.py <local|hub> [num_samples]")
        print("\nExample 1 (Local): python scripts/preview.py local 5")
        print("Example 2 (Hub):   python scripts/preview.py hub 3")
        sys.exit(1)

    source_type = sys.argv[1].lower()
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    if source_type == 'local':
        preview_dataset(DATASET_PATH_LOCAL, from_hub=False, num_samples=num_samples)
    elif source_type == 'hub':
        preview_dataset(DATASET_PATH_HUB, from_hub=True, num_samples=num_samples)
    else:
        print("Invalid source type. Use 'local' or 'hub'.")
