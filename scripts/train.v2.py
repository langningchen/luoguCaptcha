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

import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm

# 固定随机种子
RANDOM_SEED = 484858
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
print(f"Random seed set to {RANDOM_SEED}")

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
EPOCHS_STAGE1 = 6  # ResNet+Dense 阶段
EPOCHS_STAGE2 = 10  # 冻结ResNet+Attention+BiLSTM 阶段
EPOCHS_STAGE3 = 150  # 解冻全部模型阶段
BATCH_SIZE = 256
TFRECORD_DIR = "data/luogu_captcha_tfrecord"


def parse_tfrecord(example_proto):
    """Parses a single TFRecord example into image and label."""
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([CHARS_PER_LABEL], tf.int64),
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    image = tf.io.decode_png(example["image"], channels=1)
    image = tf.image.convert_image_dtype(image, tf.float32)  # Normalize to [0, 1]
    label = example["label"]  # Shape: (4,)
    return image, label


def load_and_preprocess_data(tfrecord_dir):
    """Loads and preprocesses data from TFRecord files."""
    # Get train and test TFRecord files
    train_files = sorted(glob.glob(os.path.join(tfrecord_dir, "train_part_*.tfrecord")))
    test_files = sorted(glob.glob(os.path.join(tfrecord_dir, "test_part_*.tfrecord")))

    if not train_files or not test_files:
        raise ValueError(f"No TFRecord files found in {tfrecord_dir}")

    print(f"Found {len(train_files)} train files and {len(test_files)} test files")

    # Create tf.data datasets
    train_ds = tf.data.TFRecordDataset(train_files, num_parallel_reads=tf.data.AUTOTUNE)
    test_ds = tf.data.TFRecordDataset(test_files, num_parallel_reads=tf.data.AUTOTUNE)

    # Parse TFRecords
    train_ds = train_ds.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.map(parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)

    # Shuffle and batch train dataset
    train_ds = (
        train_ds.shuffle(buffer_size=10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    )
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_ds, test_ds


def residual_block(x, filters, kernel_size=3, downsample=False):
    """ResNet residual block with optional downsampling."""
    shortcut = x
    strides = 2 if downsample else 1
    
    # First conv layer
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    # Second conv layer
    x = layers.Conv2D(filters, kernel_size, padding="same")(x)
    x = layers.BatchNormalization()(x)
    
    # Adjust shortcut if dimensions changed
    if downsample or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=strides, padding="same")(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    
    # Add residual connection
    x = layers.Add()([x, shortcut])
    x = layers.Activation("relu")(x)
    
    return x


# Load data
try:
    train_dataset, val_dataset = load_and_preprocess_data(TFRECORD_DIR)
    print("Data loaded successfully")
except Exception as e:
    print(f"Error loading TFRecord data: {e}")
    exit(1)

# ========== 阶段1: ResNet+Dense 训练20个epoch ==========
print("\n" + "="*50)
print("Stage 1: Training ResNet+Dense for 20 epochs")
print("="*50 + "\n")

inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name="input")

# Initial conv layer
x = layers.Conv2D(64, 3, padding="same")(inputs)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)

# ResNet blocks
x = residual_block(x, 64, kernel_size=3)
x = residual_block(x, 64, kernel_size=3)
x = residual_block(x, 128, kernel_size=3, downsample=True)
x = residual_block(x, 128, kernel_size=3)
x = residual_block(x, 256, kernel_size=3, downsample=True)
x = residual_block(x, 256, kernel_size=3)

# Dense layers
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(512, activation="relu")(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(CHARS_PER_LABEL * CHAR_SIZE, activation="softmax")(x)
outputs = layers.Reshape((CHARS_PER_LABEL, CHAR_SIZE))(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="LuoguCaptcha_Stage1")
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.002),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

history1 = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS_STAGE1,
    callbacks=[
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=3, min_lr=1e-5
        ),
    ],
)

# ========== 阶段2: 冻结ResNet, 添加Attention+BiLSTM, 训练20个epoch ==========
print("\n" + "="*50)
print("Stage 2: Freezing ResNet, adding Attention+BiLSTM, training for 20 epochs")
print("="*50 + "\n")

# 1. 保存stage1模型的权重
print("Saving Stage 1 weights...")
os.makedirs("models", exist_ok=True)
temp_model_path = "models/temp_stage1_resnet.weights.h5"
model.save_weights(temp_model_path)

# 2. 创建新模型，复用ResNet部分
inputs_lstm = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name="input")

# Initial conv layer
x = layers.Conv2D(64, 3, padding="same")(inputs_lstm)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)

# ResNet blocks (same as stage 1)
x = residual_block(x, 64, kernel_size=3)
x = residual_block(x, 64, kernel_size=3)
x = residual_block(x, 128, kernel_size=3, downsample=True)
x = residual_block(x, 128, kernel_size=3)
x = residual_block(x, 256, kernel_size=3, downsample=True)
x = residual_block(x, 256, kernel_size=3)

# 保存ResNet输出的shape用于reshape
resnet_output_shape = x.shape

# 将CNN输出reshape为序列形式 (batch, time_steps, features)
x = layers.Reshape((resnet_output_shape[1] * resnet_output_shape[2], resnet_output_shape[3]))(x)

# Self-Attention mechanism
attention_output = layers.MultiHeadAttention(
    num_heads=4, 
    key_dim=64,
    dropout=0.1
)(x, x)
x = layers.LayerNormalization()(x + attention_output)

# 双向LSTM层
x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
x = layers.Dropout(0.2)(x)
x = layers.Bidirectional(layers.LSTM(128, return_sequences=False))(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(CHARS_PER_LABEL * CHAR_SIZE, activation="softmax")(x)
outputs_lstm = layers.Reshape((CHARS_PER_LABEL, CHAR_SIZE))(x)

model_lstm = keras.Model(inputs=inputs_lstm, outputs=outputs_lstm, name="LuoguCaptcha_Stage2")

print("Loading ResNet weights into Stage 2...")
model_lstm.load_weights(temp_model_path, skip_mismatch=True)  # by_name=True

# 4. 冻结ResNet部分
print("Freezing ResNet layers...")
frozen_count = 0
for layer in model_lstm.layers:
    if isinstance(layer, layers.Reshape):
        break  # ResNet部分结束
    if isinstance(layer, (layers.Conv2D, layers.BatchNormalization, layers.Add)):
        layer.trainable = False
        frozen_count += 1

print(f"Frozen {frozen_count} ResNet layers.")

# 5. 清理临时文件
if os.path.exists(temp_model_path):
    os.remove(temp_model_path)
    print(f"Removed temporary file: {temp_model_path}")

model_lstm.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.002),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model_lstm.summary()

history2 = model_lstm.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS_STAGE2,
    callbacks=[
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=3, min_lr=1e-5
        ),
    ],
)

# ========== 阶段3: 解冻全部模型, 训练150个epoch ==========
print("\n" + "="*50)
print("Stage 3: Unfreezing all layers, training for 150 epochs")
print("="*50 + "\n")

# 解冻所有层
for layer in model_lstm.layers:
    layer.trainable = True

# 使用较小的学习率
model_lstm.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history3 = model_lstm.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS_STAGE3,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=5, min_lr=1e-6
        ),
    ],
)

# 保存模型 (本地)
final_model_path = "models/luoguCaptcha.ResNet+SelfAttention-BiLSTM.keras"
model_lstm.save(final_model_path)
print(f"Model saved to {final_model_path}")

# 提示上传
print(f"Run `python scripts/huggingface.py upload_model {final_model_path}` to upload.")