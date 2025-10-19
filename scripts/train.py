# scripts/train.py
# Copyright (C) 2025 Langning Chen
# Math-optimized captcha training script

import os
import glob
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm

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


# Load data
try:
    train_dataset, val_dataset = load_and_preprocess_data(TFRECORD_DIR)
    print("Data loaded successfully")
except Exception as e:
    print(f"Error loading TFRecord data: {e}")
    exit(1)

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
