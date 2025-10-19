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

import os
import sys
from huggingface_hub import HfApi, login, create_repo
from datasets import load_from_disk
from pathlib import Path

# 请将 YOUR_REPO_ID 替换为你希望在 Hugging Face Hub 上使用的仓库 ID
DATASET_REPO_ID = "langningchen/luogu-captcha-dataset"
MODEL_REPO_ID = "langningchen/luogu-captcha-model"


def upload_dataset(local_path):
    """
    将本地数据集上传到 Hugging Face Hub。
    假设 local_path 包含一个 DatasetDict (train/test)。
    """
    if not Path(local_path).exists():
        print(f"Error: Dataset path not found at {local_path}")
        sys.exit(1)

    try:
        # 1. 加载本地数据集
        dataset_dict = load_from_disk(local_path)

        # 2. 检查是否为 DatasetDict，并推动到 Hub
        if hasattr(dataset_dict, "push_to_hub"):
            print(f"Uploading dataset to {DATASET_REPO_ID}...")
            # 创建仓库 (如果不存在)
            api = HfApi()
            create_repo(repo_id=DATASET_REPO_ID, repo_type="dataset", exist_ok=True)

            # 推送到 Hub
            dataset_dict.push_to_hub(DATASET_REPO_ID)
            print(
                f"Dataset successfully uploaded to: https://huggingface.co/datasets/{DATASET_REPO_ID}"
            )
        else:
            print("Error: Loaded object is not a DatasetDict. Check generate.py.")

    except Exception as e:
        print(f"An error occurred during dataset upload: {e}")
        print("Please ensure you are logged in using `huggingface-cli login`")


def upload_model(local_model_path):
    """
    将 Keras 模型文件上传到 Hugging Face Hub。
    """
    if not os.path.exists(local_model_path):
        print(f"Error: Model file not found at {local_model_path}")
        sys.exit(1)

    try:
        api = HfApi()
        # 创建仓库 (如果不存在)
        create_repo(repo_id=MODEL_REPO_ID, repo_type="model", exist_ok=True)

        # 上传文件
        api.upload_file(
            path_or_fileobj=local_model_path,
            path_in_repo=os.path.basename(
                local_model_path
            ),  # 文件名: luoguCaptcha.keras
            repo_id=MODEL_REPO_ID,
        )
        print(
            f"Model successfully uploaded to: https://huggingface.co/{MODEL_REPO_ID}/blob/main/{os.path.basename(local_model_path)}"
        )

    except Exception as e:
        print(f"An error occurred during model upload: {e}")
        print("Please ensure you are logged in using `huggingface-cli login`")


def main():
    if len(sys.argv) < 3:
        print(
            "Usage: python scripts/huggingface.py <upload_dataset|upload_model> <path>"
        )
        sys.exit(1)

    command = sys.argv[1]
    path_arg = sys.argv[2]

    # 确保用户已登录，或至少尝试连接
    try:
        login(token=os.environ.get("HF_TOKEN"), add_to_git_credential=False)
    except Exception:
        pass  # 如果未登录，上传时会抛出异常，这是预期行为

    if command == "upload_dataset":
        upload_dataset(path_arg)
    elif command == "upload_model":
        upload_model(path_arg)
    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
