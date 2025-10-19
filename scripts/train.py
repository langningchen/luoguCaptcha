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
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datasets import load_dataset

# 自动选择设备
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Using GPU: {gpus}")
    except Exception as e:
        print(f"GPU setup error: {e}")
else:
    print("Using CPU")

# 参数
CHAR_SIZE = 256
CHARS_PER_LABEL = 4
IMG_HEIGHT, IMG_WIDTH = 35, 90
EPOCHS = 15
BATCH_SIZE = 256
DATASET_PATH = "langningchen/luogu-captcha-dataset"


# 数据加载与预处理
def load_and_preprocess_data(dataset_path):
    """Loads pre-processed dataset from Hugging Face Hub (zero CPU pre-processing on trainer)."""

    # 1. 在线加载 DatasetDict (直接获取预处理好的 train/test 分割)
    dataset_dict = load_dataset(dataset_path)

    train_ds_hf = dataset_dict["train"]
    val_ds_hf = dataset_dict["test"]

    # 2. 转换为 tf.data.Dataset (无需任何 map/shuffle/num_proc)
    train_ds = train_ds_hf.to_tf_dataset(
        columns="image",
        label_cols="label",  # 标签是 (4) 形状的整数
        batch_size=BATCH_SIZE,
        shuffle=True,  # 使用 to_tf_dataset 内置的 shuffle
    )
    val_ds = val_ds_hf.to_tf_dataset(
        columns="image", label_cols="label", batch_size=BATCH_SIZE
    )

    # 3. 使用 prefetch 隐藏 I/O 延迟
    return train_ds.prefetch(tf.data.AUTOTUNE), val_ds.prefetch(tf.data.AUTOTUNE)


train_dataset, val_dataset = load_and_preprocess_data(DATASET_PATH)

# 模型架构（数学优化，适合验证码识别）
inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name="input")
x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Flatten()(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(512, activation="relu")(x)
x = layers.Dense(CHARS_PER_LABEL * CHAR_SIZE, activation="softmax")(x)
outputs = layers.Reshape((CHARS_PER_LABEL, CHAR_SIZE))(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="LuoguCaptcha")
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    # 必须使用 SparseCategoricalCrossentropy 匹配 (4) 形状的整数标签
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# 训练
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=3, min_lr=1e-5
        ),
    ],
)

# 保存模型 (本地)
os.makedirs("models", exist_ok=True)
final_model_path = "models/luoguCaptcha.keras"
model.save(final_model_path)
print(f"Model saved to {final_model_path}")

# 提示上传
print(f"Run `python scripts/huggingface.py upload_model {final_model_path}` to upload.")
