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
PATCH_SIZE = 5  # patch大小，要求能整除IMG_HEIGHT和IMG_WIDTH
D_MODEL = 128  # Transformer嵌入维度 (256 -> 128)
NUM_HEADS = 4  # 多头注意力头数 (8 -> 4)
NUM_LAYERS = 4  # Transformer encoder层数 (6 -> 4)
DFF = 256  # 前馈网络中间层维度 (512 -> 256)
DROPOUT_RATE = 0.1
EPOCHS = 150
BATCH_SIZE = 128  # (256 -> 128)
TFRECORD_DIR = "data/luogu_captcha_tfrecord"

# 计算patch数量
NUM_PATCHES_H = IMG_HEIGHT // PATCH_SIZE  # 35 / 5 = 7
NUM_PATCHES_W = IMG_WIDTH // PATCH_SIZE   # 90 / 5 = 18
NUM_PATCHES = NUM_PATCHES_H * NUM_PATCHES_W  # 7 * 18 = 126


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


class PatchEmbedding(layers.Layer):
    """将图像分割为patch并进行线性嵌入。"""

    def __init__(self, patch_size, d_model, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.d_model = d_model
        # 用卷积实现patch提取+线性投影
        self.projection = layers.Conv2D(
            d_model,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
        )

    def call(self, x):
        # x: (batch, H, W, 1)
        x = self.projection(x)  # (batch, num_patches_h, num_patches_w, d_model)
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, (batch_size, -1, self.d_model))  # (batch, num_patches, d_model)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size, "d_model": self.d_model})
        return config


class LearnedPositionalEncoding(layers.Layer):
    """可学习的位置编码。"""

    def __init__(self, num_positions, d_model, **kwargs):
        super().__init__(**kwargs)
        self.num_positions = num_positions
        self.d_model = d_model

    def build(self, input_shape):
        self.pos_embedding = self.add_weight(
            name="pos_embedding",
            shape=(1, self.num_positions, self.d_model),
            initializer="truncated_normal",
            trainable=True,
        )

    def call(self, x):
        return x + self.pos_embedding

    def get_config(self):
        config = super().get_config()
        config.update({"num_positions": self.num_positions, "d_model": self.d_model})
        return config


class TransformerEncoderBlock(layers.Layer):
    """单个Transformer Encoder块：Multi-Head Attention + Feed Forward + LayerNorm + Dropout。"""

    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate

        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // num_heads, dropout=dropout_rate
        )
        self.ffn = keras.Sequential([
            layers.Dense(dff, activation="gelu"),
            layers.Dropout(dropout_rate),
            layers.Dense(d_model),
            layers.Dropout(dropout_rate),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)

    def call(self, x, training=None):
        # Pre-norm架构
        x_norm = self.layernorm1(x)
        attn_output = self.mha(x_norm, x_norm, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        x = x + attn_output

        x_norm = self.layernorm2(x)
        ffn_output = self.ffn(x_norm, training=training)
        x = x + ffn_output

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "dropout_rate": self.dropout_rate,
        })
        return config


class CLSTokens(layers.Layer):
    """可学习的CLS tokens，拼接到序列前面。"""

    def __init__(self, num_tokens, d_model, **kwargs):
        super().__init__(**kwargs)
        self.num_tokens = num_tokens
        self.d_model = d_model

    def build(self, input_shape):
        self.cls_tokens = self.add_weight(
            name="cls_tokens",
            shape=(1, self.num_tokens, self.d_model),
            initializer="truncated_normal",
            trainable=True,
        )

    def call(self, x):
        batch_size = keras.ops.shape(x)[0]
        cls_broadcast = keras.ops.broadcast_to(
            self.cls_tokens, (batch_size, self.num_tokens, self.d_model)
        )
        return keras.ops.concatenate([cls_broadcast, x], axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({"num_tokens": self.num_tokens, "d_model": self.d_model})
        return config


def build_vit_captcha_model():
    """构建基于Vision Transformer (Encoder-Only) 的验证码识别模型。"""

    inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1), name="input")

    # Patch Embedding
    x = PatchEmbedding(patch_size=PATCH_SIZE, d_model=D_MODEL, name="patch_embedding")(inputs)
    # x: (batch, NUM_PATCHES, D_MODEL)

    # 添加4个可学习的 CLS tokens 并拼接到序列前面
    x = CLSTokens(num_tokens=CHARS_PER_LABEL, d_model=D_MODEL, name="cls_tokens")(x)
    # x: (batch, 4 + NUM_PATCHES, D_MODEL)

    # 位置编码 (覆盖 CLS tokens + patch tokens)
    x = LearnedPositionalEncoding(
        num_positions=CHARS_PER_LABEL + NUM_PATCHES,
        d_model=D_MODEL,
        name="positional_encoding",
    )(x)
    x = layers.Dropout(DROPOUT_RATE)(x)

    # Transformer Encoder 层
    for i in range(NUM_LAYERS):
        x = TransformerEncoderBlock(
            d_model=D_MODEL,
            num_heads=NUM_HEADS,
            dff=DFF,
            dropout_rate=DROPOUT_RATE,
            name=f"transformer_block_{i}",
        )(x)

    # 最终LayerNorm
    x = layers.LayerNormalization(epsilon=1e-6, name="final_norm")(x)

    # 取前4个token (CLS tokens) 作为4个字符的表示
    x = layers.Lambda(lambda t: t[:, :CHARS_PER_LABEL, :], name="extract_cls")(x)
    # x: (batch, 4, D_MODEL)

    # 每个字符分类头
    x = layers.Dense(DFF, activation="gelu", name="cls_head_dense")(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    outputs = layers.Dense(CHAR_SIZE, activation="softmax", name="cls_head_output")(x)
    # outputs: (batch, 4, CHAR_SIZE)

    model = keras.Model(inputs=inputs, outputs=outputs, name="LuoguCaptcha_ViT")
    return model

# Load data
try:
    train_dataset, val_dataset = load_and_preprocess_data(TFRECORD_DIR)
    print("Data loaded successfully")
except Exception as e:
    print(f"Error loading TFRecord data: {e}")
    exit(1)

# ========== 构建并训练 Vision Transformer 模型 ==========
print("\n" + "=" * 50)
print("Building Vision Transformer (Encoder-Only) Model")
print(f"  Patch size: {PATCH_SIZE}x{PATCH_SIZE}")
print(f"  Num patches: {NUM_PATCHES} ({NUM_PATCHES_H}x{NUM_PATCHES_W})")
print(f"  D_model: {D_MODEL}, Heads: {NUM_HEADS}, Layers: {NUM_LAYERS}, DFF: {DFF}")
print("=" * 50 + "\n")

model = build_vit_captcha_model()

# Warmup + Cosine Decay 学习率调度
class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000, max_lr=1e-3):
        super().__init__()
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.max_lr = max_lr

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps = tf.cast(self.warmup_steps, tf.float32)
        # Linear warmup
        warmup_lr = self.max_lr * (step / warmup_steps)
        # Cosine decay after warmup
        decay_lr = self.max_lr * 0.5 * (
            1.0 + tf.cos(np.pi * (step - warmup_steps) / (50000.0 - warmup_steps))
        )
        return tf.where(step < warmup_steps, warmup_lr, decay_lr)

    def get_config(self):
        return {
            "d_model": self.d_model,
            "warmup_steps": self.warmup_steps,
            "max_lr": self.max_lr,
        }


lr_schedule = WarmupCosineDecay(d_model=D_MODEL, warmup_steps=2000, max_lr=5e-4)

model.compile(
    optimizer=keras.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=1e-4,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

os.makedirs("models", exist_ok=True)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=7, min_lr=1e-6
        ),
    ],
)

# 保存模型
final_model_path = "models/luoguCaptcha.ViT-EncoderOnly.keras"
model.save(final_model_path)
print(f"Model saved to {final_model_path}")

# 提示上传
print(f"Run `python scripts/huggingface.py upload_model {final_model_path}` to upload.")