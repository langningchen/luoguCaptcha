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

import numpy as np
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import sys
import os

# 自动选择设备
physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("Using GPU:", physical_devices[0])
else:
    print("Using CPU")

CharSize = 256
CharsPerLabel = 4
IMG_HEIGHT, IMG_WIDTH = 35, 90

model_path = os.path.join("models", "luoguCaptcha.keras")

if len(sys.argv) == 2:
    img_path = sys.argv[1]

    # 1. 加载模型
    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Please ensure the model is saved at {model_path} or upload it first.")
        sys.exit(1)

    # 2. 预处理输入图像 (与 generate.py 保持一致)
    try:
        img = Image.open(img_path)
    except FileNotFoundError:
        print(f"Error: Image file not found at {img_path}")
        sys.exit(1)

    # 灰度转换 -> NumPy -> 归一化 -> 添加通道 -> 添加 Batch 维度
    # 目标形状: (1, 35, 90, 1)
    image_np = np.array(img.convert("L"), dtype=np.float32) / 255.0
    image_np = np.expand_dims(image_np, axis=-1)
    image_np = np.expand_dims(image_np, axis=0)  # 添加 Batch 维度

    # 3. 预测
    pred_probabilities = model.predict(image_np)

    # 4. 解码：找到每个字符位置的最高概率索引 (ASCII 值)
    # pred_probabilities 形状: (1, 4, 256)
    predicted_ascii_codes = tf.math.argmax(pred_probabilities, axis=-1).numpy()[0]

    # 5. 转换为字符并打印
    predicted_captcha = "".join(map(chr, predicted_ascii_codes))
    print(f"Predicted CAPTCHA: {predicted_captcha}")

else:
    print("Usage: python predict.py <image_path>")
