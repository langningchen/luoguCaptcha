#!/usr/bin/env python3
"""Train and export the browser-sized Luogu captcha model.

The default configuration is intended for a Google Colab T4 runtime. It reads
the colored Hugging Face dataset directly and exports both a Keras model and a
validated quantized TensorFlow.js Layers model.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf
from datasets import Image as DatasetImage
from datasets import load_dataset
from tensorflow import keras
from tensorflow.keras import layers


DATASET_REPO_ID = "langningchen/luogu-captcha-dataset-colored"
# Keep the source encoding stable so an existing decoded-image cache remains
# reusable. The model target is case-insensitive because Luogu's verifier is.
ALPHABET = "abcdefghijklmnpqrstuvwxyz123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
MODEL_ALPHABET = "abcdefghijklmnopqrstuvwxyz123456789"
SOURCE_TO_MODEL_CLASS = tf.constant(
    [MODEL_ALPHABET.index(char.lower()) for char in ALPHABET], dtype=tf.int32
)
IMAGE_HEIGHT = 35
IMAGE_WIDTH = 90
IMAGE_CHANNELS = 3
CAPTCHA_LENGTH = 4
SEED = 484858


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET_REPO_ID)
    parser.add_argument("--output-dir", default="models/browser")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--validation-size", type=int, default=10_000)
    parser.add_argument(
        "--cache-dir",
        default="data/luogu_tf_cache",
        help="Local decoded-image cache. Colab should place this on /content.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional subset size for smoke tests; the default uses all samples.",
    )
    parser.add_argument(
        "--skip-tfjs",
        action="store_true",
        help="Save Keras artifacts without invoking tensorflowjs_converter.",
    )
    parser.add_argument(
        "--compress-only",
        action="store_true",
        help="Re-export and validate an existing Keras model without training.",
    )
    parser.add_argument(
        "--skip-onnx",
        action="store_true",
        help="Do not export or validate ONNX artifacts.",
    )
    return parser.parse_args()


def configure_runtime() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    if gpus:
        keras.mixed_precision.set_global_policy("mixed_float16")
        print(f"Using GPU with mixed precision: {gpus[0].name}")
    else:
        print("No GPU detected; training will be slow.")


def encode_labels(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
    char_to_index = {char: index for index, char in enumerate(ALPHABET)}
    encoded: list[list[int]] = []
    for label in batch["label"]:
        if len(label) != CAPTCHA_LENGTH:
            raise ValueError(f"Expected a 4-character label, got {label!r}")
        try:
            encoded.append([char_to_index[char] for char in label])
        except KeyError as error:
            raise ValueError(
                f"Unsupported character {error.args[0]!r} in {label!r}"
            ) from error
    return {"target": encoded}


def load_splits(dataset_id: str, validation_size: int, max_train_samples: int | None):
    dataset = load_dataset(dataset_id, split="train")
    # The Hub stores this column with decode=False; TensorFlow needs dense RGB arrays.
    dataset = dataset.cast_column("image", DatasetImage(decode=True))
    if validation_size <= 0 or validation_size >= len(dataset):
        raise ValueError("validation-size must be between 1 and dataset_size - 1")

    split = dataset.train_test_split(test_size=validation_size, seed=SEED, shuffle=True)
    train_data = split["train"]
    validation_data = split["test"]
    if max_train_samples is not None:
        sample_count = min(max_train_samples, len(train_data))
        train_data = train_data.select(range(sample_count))

    train_data = train_data.map(encode_labels, batched=True, desc="Encoding labels")
    validation_data = validation_data.map(
        encode_labels, batched=True, desc="Encoding validation labels"
    )
    return train_data, validation_data


def normalize_batch(images: tf.Tensor, labels: tf.Tensor):
    images = tf.cast(images, tf.float32) / 255.0
    images = tf.ensure_shape(
        images, [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS]
    )
    labels = tf.cast(labels, tf.int32)
    labels = tf.gather(SOURCE_TO_MODEL_CLASS, labels)
    labels = tf.ensure_shape(labels, [None, CAPTCHA_LENGTH])
    return images, labels


def compact_for_cache(images: tf.Tensor, labels: tf.Tensor):
    # Hugging Face promotes integer image arrays to int64 in to_tf_dataset.
    # Values are still 0..255, so preserve them as uint8 before disk caching.
    return tf.cast(images, tf.uint8), tf.cast(labels, tf.int32)


def to_tf_dataset(
    dataset, batch_size: int, training: bool, cache_dir: Path
) -> tf.data.Dataset:
    split_name = "train" if training else "validation"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{split_name}-{dataset._fingerprint}-b{batch_size}"
    tf_dataset = dataset.to_tf_dataset(
        columns="image",
        label_cols="target",
        shuffle=False,
        batch_size=batch_size,
        drop_remainder=training,
        num_workers=min(4, max(1, os.cpu_count() or 1)),
    )
    tf_dataset = tf_dataset.map(
        compact_for_cache, num_parallel_calls=tf.data.AUTOTUNE
    )
    # Cache uint8 images before float normalization. This uses about 4.7 GB for
    # the full training split and avoids decoding all JPEGs again every epoch.
    tf_dataset = tf_dataset.cache(str(cache_path))
    if training:
        tf_dataset = tf_dataset.shuffle(
            buffer_size=128, seed=SEED, reshuffle_each_iteration=True
        )
    tf_dataset = tf_dataset.map(normalize_batch, num_parallel_calls=tf.data.AUTOTUNE)
    print(f"{split_name.capitalize()} cache: {cache_path}")
    return tf_dataset.prefetch(tf.data.AUTOTUNE)


def conv_layer(x: tf.Tensor, filters: int) -> tf.Tensor:
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    return layers.ReLU()(x)


def conv_block(x: tf.Tensor, filters: int) -> tf.Tensor:
    x = conv_layer(x, filters)
    x = conv_layer(x, filters)
    return layers.MaxPooling2D(pool_size=2)(x)


def build_model() -> keras.Model:
    inputs = keras.Input(
        shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), name="captcha"
    )
    x = conv_block(inputs, 32)
    x = conv_block(x, 64)
    x = conv_block(x, 128)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(512, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(0.25)(x)

    # float32 logits keep loss/argmax numerically stable under mixed precision.
    x = layers.Dense(
        CAPTCHA_LENGTH * len(MODEL_ALPHABET), dtype="float32", name="logits"
    )(x)
    outputs = layers.Reshape(
        (CAPTCHA_LENGTH, len(MODEL_ALPHABET)), name="characters"
    )(x)
    return keras.Model(inputs, outputs, name="luogu_captcha_browser")


@keras.utils.register_keras_serializable(package="luoguCaptcha")
def captcha_accuracy(labels: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
    predicted = tf.argmax(logits, axis=-1, output_type=tf.int32)
    labels = tf.cast(labels, tf.int32)
    exact_matches = tf.reduce_all(tf.equal(labels, predicted), axis=-1)
    return tf.cast(exact_matches, tf.float32)


def compile_model(model: keras.Model) -> None:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(name="char_accuracy"),
            captcha_accuracy,
        ],
    )


def run_tfjs_converter(
    keras_path: Path,
    tfjs_dir: Path,
    profile_name: str,
    quantization_map: dict[str, list[str] | None] | None,
) -> None:
    if tfjs_dir.exists():
        shutil.rmtree(tfjs_dir)
    tfjs_dir.mkdir(parents=True)
    command = [
        "tensorflowjs_converter",
        "--input_format=keras",
        "--output_format=tfjs_layers_model",
    ]
    if quantization_map is not None:
        for dtype, patterns in quantization_map.items():
            if patterns is None:
                command.append(f"--quantize_{dtype}")
            else:
                command.append(f"--quantize_{dtype}={','.join(patterns)}")
    command.extend([str(keras_path), str(tfjs_dir)])
    print(f"Exporting {profile_name} TensorFlow.js candidate...", flush=True)
    converter_env = os.environ.copy()
    converter_env["CUDA_VISIBLE_DEVICES"] = "-1"
    subprocess.run(command, check=True, env=converter_env)
    normalize_tfjs_model_json(tfjs_dir)


def normalize_tfjs_model_json(tfjs_dir: Path) -> None:
    """Make Keras 3 topology fields consumable by TensorFlow.js Layers."""
    model_json_path = tfjs_dir / "model.json"
    model_json = json.loads(model_json_path.read_text(encoding="utf-8"))
    topology = model_json.get("modelTopology", {})
    model_config = topology.get("model_config")
    if not isinstance(model_config, dict):
        return

    if model_config.get("class_name") == "Functional":
        model_config["class_name"] = "Model"
    config = model_config.get("config", {})
    for key in ("input_layers", "output_layers"):
        value = config.get(key)
        if value and not isinstance(value[0], list):
            config[key] = [value]

    for layer in config.get("layers", []):
        layer_config = layer.get("config", {})
        if layer.get("class_name") == "InputLayer":
            if "batch_shape" in layer_config:
                layer_config["batchInputShape"] = layer_config.pop("batch_shape")
            layer_config.pop("optional", None)

        normalized_nodes = []
        for node in layer.get("inbound_nodes", []):
            connections = []
            if isinstance(node, list):
                normalized_nodes.append(node)
                continue
            for argument in node.get("args", []):
                history = argument.get("config", {}).get("keras_history")
                if not history or len(history) != 3:
                    raise ValueError(
                        f"Cannot normalize inbound node for layer {layer.get('name')}"
                    )
                connections.append([*history, {}])
            normalized_nodes.append(connections)
        layer["inbound_nodes"] = normalized_nodes

    model_json_path.write_text(
        json.dumps(model_json, ensure_ascii=True, separators=(",", ":")),
        encoding="utf-8",
    )


def evaluate_tfjs_model(
    model: keras.Model,
    validation_dataset: tf.data.Dataset,
    tfjs_dir: Path,
) -> dict[str, float]:
    from tensorflowjs.read_weights import read_weights

    model_json = json.loads((tfjs_dir / "model.json").read_text(encoding="utf-8"))
    entries = read_weights(
        model_json["weightsManifest"], str(tfjs_dir), flatten=True
    )
    weights_by_name = {entry["name"]: entry["data"] for entry in entries}
    candidate_weights = []
    missing_names = []
    for variable in model.weights:
        variable_path = getattr(variable, "path", variable.name).removesuffix(":0")
        if variable_path in weights_by_name:
            candidate_weights.append(weights_by_name[variable_path])
            continue

        suffix_matches = [
            name
            for name in weights_by_name
            if variable_path.endswith(f"/{name}") or name.endswith(f"/{variable_path}")
        ]
        if len(suffix_matches) == 1:
            candidate_weights.append(weights_by_name[suffix_matches[0]])
        else:
            missing_names.append(variable_path)

    if missing_names:
        raise ValueError(
            "TensorFlow.js manifest is missing Keras weights: "
            f"{missing_names}; available={list(weights_by_name)}"
        )
    reference_weights = model.get_weights()
    reference_shapes = [weight.shape for weight in reference_weights]
    candidate_shapes = [weight.shape for weight in candidate_weights]
    if candidate_shapes != reference_shapes:
        raise ValueError(
            "Mapped TensorFlow.js weight shapes do not match Keras. "
            f"Expected {reference_shapes}, got {candidate_shapes}"
        )

    try:
        model.set_weights(candidate_weights)
        return {
            key: float(value)
            for key, value in model.evaluate(
                validation_dataset, return_dict=True, verbose=2
            ).items()
        }
    finally:
        model.set_weights(reference_weights)


def metric_counts(metrics: dict[str, float], sample_count: int) -> dict[str, int]:
    return {
        "captcha_correct": round(metrics["captcha_accuracy"] * sample_count),
        "characters_correct": round(
            metrics["char_accuracy"] * sample_count * CAPTCHA_LENGTH
        ),
    }


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def weight_specs(tfjs_dir: Path) -> list[dict]:
    model_json = json.loads((tfjs_dir / "model.json").read_text(encoding="utf-8"))
    return [
        weight
        for group in model_json["weightsManifest"]
        for weight in group["weights"]
    ]


def mixed_quantization_profiles(specs: list[dict]):
    small_names = [spec["name"] for spec in specs if len(spec["shape"]) <= 1]
    conv_names = [spec["name"] for spec in specs if len(spec["shape"]) == 4]
    matrix_specs = sorted(
        (spec for spec in specs if len(spec["shape"]) == 2),
        key=lambda spec: int(np.prod(spec["shape"])),
    )
    if len(matrix_specs) != 2:
        return []

    output_name = matrix_specs[0]["name"]
    hidden_name = matrix_specs[1]["name"]

    def mixed(names: list[str]):
        return {"uint8": None, "uint16": sorted(set(names))}

    return [
        ("uint8-small16", mixed(small_names)),
        ("uint8-small-output16", mixed(small_names + [output_name])),
        (
            "uint8-small-output-conv16",
            mixed(small_names + [output_name] + conv_names),
        ),
        ("uint8-small-hidden16", mixed(small_names + [hidden_name])),
    ]


def export_smallest_accurate_tfjs(
    model: keras.Model,
    keras_path: Path,
    output_dir: Path,
    validation_dataset: tf.data.Dataset,
    reference_metrics: dict[str, float],
    validation_size: int,
) -> dict:
    reference_counts = metric_counts(reference_metrics, validation_size)
    candidates: dict[str, dict] = {}
    selected_dir: Path | None = None
    selected_quantization: str | None = None

    profiles: list[tuple[str, dict[str, list[str] | None] | None]] = [
        ("uint8", {"uint8": ["*"]})
    ]
    profile_index = 0
    while profile_index < len(profiles):
        profile_name, quantization_map = profiles[profile_index]
        profile_index += 1
        candidate_dir = output_dir / f".tfjs-{profile_name}-candidate"
        run_tfjs_converter(
            keras_path,
            candidate_dir,
            profile_name,
            quantization_map,
        )
        candidate_metrics = evaluate_tfjs_model(
            model, validation_dataset, candidate_dir
        )
        candidate_counts = metric_counts(candidate_metrics, validation_size)
        passed = all(
            candidate_counts[key] >= value
            for key, value in reference_counts.items()
        )
        candidates[profile_name] = {
            "size_bytes": directory_size(candidate_dir),
            "validation": candidate_metrics,
            "correct_counts": candidate_counts,
            "no_measured_accuracy_loss": passed,
        }
        print(json.dumps({profile_name: candidates[profile_name]}, indent=2))
        if passed:
            selected_dir = candidate_dir
            selected_quantization = profile_name
            break

        if profile_name == "uint8":
            profiles.extend(mixed_quantization_profiles(weight_specs(candidate_dir)))
            profiles.append(("uint16", {"uint16": ["*"]}))
        shutil.rmtree(candidate_dir)

    if selected_dir is None:
        selected_quantization = "float32"
        selected_dir = output_dir / ".tfjs-float32-candidate"
        run_tfjs_converter(
            keras_path,
            selected_dir,
            profile_name="float32",
            quantization_map=None,
        )
        candidates["float32"] = {
            "size_bytes": directory_size(selected_dir),
            "validation": reference_metrics,
            "correct_counts": reference_counts,
            "no_measured_accuracy_loss": True,
        }

    tfjs_dir = output_dir / "tfjs"
    if tfjs_dir.exists():
        shutil.rmtree(tfjs_dir)
    selected_dir.rename(tfjs_dir)
    return {
        "selected": selected_quantization,
        "size_bytes": directory_size(tfjs_dir),
        "reference_correct_counts": reference_counts,
        "candidates": candidates,
    }


def validate_existing_tfjs(
    model: keras.Model,
    output_dir: Path,
    validation_dataset: tf.data.Dataset,
    reference_metrics: dict[str, float],
    validation_size: int,
) -> dict:
    tfjs_dir = output_dir / "tfjs"
    if not (tfjs_dir / "model.json").exists():
        raise FileNotFoundError(f"Existing TensorFlow.js model not found: {tfjs_dir}")
    normalize_tfjs_model_json(tfjs_dir)
    metrics = evaluate_tfjs_model(model, validation_dataset, tfjs_dir)
    reference_counts = metric_counts(reference_metrics, validation_size)
    candidate_counts = metric_counts(metrics, validation_size)
    passed = all(
        candidate_counts[key] >= value for key, value in reference_counts.items()
    )
    if not passed:
        raise RuntimeError(
            "Existing TensorFlow.js model reduced validation accuracy: "
            f"{candidate_counts} < {reference_counts}"
        )
    quantized_dtypes = sorted(
        {
            spec.get("quantization", {}).get("dtype", spec["dtype"])
            for spec in weight_specs(tfjs_dir)
        }
    )
    return {
        "selected": "existing-validated",
        "quantized_dtypes": quantized_dtypes,
        "size_bytes": directory_size(tfjs_dir),
        "reference_correct_counts": reference_counts,
        "candidates": {
            "existing": {
                "size_bytes": directory_size(tfjs_dir),
                "validation": metrics,
                "correct_counts": candidate_counts,
                "no_measured_accuracy_loss": True,
            }
        },
    }


def evaluate_onnx_model(
    validation_dataset: tf.data.Dataset, onnx_path: Path
) -> tuple[dict[str, float], dict[str, str]]:
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    captcha_correct = 0
    characters_correct = 0
    loss_sum = 0.0
    sample_count = 0
    for images, labels in validation_dataset:
        logits = session.run([output_name], {input_name: images.numpy()})[0]
        logits = tf.convert_to_tensor(logits)
        predicted = tf.argmax(logits, axis=-1, output_type=tf.int32)
        labels = tf.cast(labels, tf.int32)
        characters_correct += int(
            tf.reduce_sum(tf.cast(tf.equal(labels, predicted), tf.int32))
        )
        captcha_correct += int(
            tf.reduce_sum(
                tf.cast(
                    tf.reduce_all(tf.equal(labels, predicted), axis=-1),
                    tf.int32,
                )
            )
        )
        loss_sum += float(
            tf.reduce_sum(
                keras.losses.sparse_categorical_crossentropy(
                    labels, logits, from_logits=True
                )
            )
        )
        sample_count += int(tf.shape(labels)[0])

    return (
        {
            "captcha_accuracy": captcha_correct / sample_count,
            "char_accuracy": characters_correct
            / (sample_count * CAPTCHA_LENGTH),
            "loss": loss_sum / (sample_count * CAPTCHA_LENGTH),
        },
        {"input_name": input_name, "output_name": output_name},
    )


def export_onnx(
    model: keras.Model,
    output_dir: Path,
    validation_dataset: tf.data.Dataset,
    reference_metrics: dict[str, float],
    validation_size: int,
) -> dict:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    onnx_dir = output_dir / "onnx"
    if onnx_dir.exists():
        shutil.rmtree(onnx_dir)
    onnx_dir.mkdir(parents=True)
    fp32_path = onnx_dir / "luogu-captcha-fp32.onnx"
    int8_path = onnx_dir / "luogu-captcha-int8.onnx"
    input_signature = [
        tf.TensorSpec(
            shape=(None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS),
            dtype=tf.float32,
            name="captcha",
        )
    ]

    previous_policy = keras.mixed_precision.global_policy().name
    keras.mixed_precision.set_global_policy("float32")
    try:
        onnx_model = build_model()
    finally:
        keras.mixed_precision.set_global_policy(previous_policy)
    onnx_model.set_weights(model.get_weights())
    compile_model(onnx_model)
    float32_keras_metrics = {
        key: float(value)
        for key, value in onnx_model.evaluate(
            validation_dataset, return_dict=True, verbose=2
        ).items()
    }

    print("Exporting pure-FP32 ONNX model...", flush=True)
    onnx_model.export(
        fp32_path,
        format="onnx",
        input_signature=input_signature,
        opset_version=18,
        verbose=False,
    )
    fp32_metrics, io_names = evaluate_onnx_model(validation_dataset, fp32_path)
    reference_counts = metric_counts(reference_metrics, validation_size)
    fp32_counts = metric_counts(fp32_metrics, validation_size)
    fp32_passed = all(
        fp32_counts[key] >= value for key, value in reference_counts.items()
    )

    print("Exporting dynamic INT8 ONNX candidate...", flush=True)
    int8_error = None
    try:
        quantize_dynamic(
            str(fp32_path),
            str(int8_path),
            weight_type=QuantType.QUInt8,
        )
        int8_metrics, int8_io_names = evaluate_onnx_model(
            validation_dataset, int8_path
        )
        int8_counts = metric_counts(int8_metrics, validation_size)
        int8_passed = all(
            int8_counts[key] >= value for key, value in reference_counts.items()
        )
    except Exception as error:  # Preserve the validated FP32 fallback.
        int8_metrics = None
        int8_io_names = io_names
        int8_counts = None
        int8_passed = False
        int8_error = f"{type(error).__name__}: {error}"
        int8_path.unlink(missing_ok=True)
    selected = "int8" if int8_passed else "fp32"
    selected_path = int8_path if int8_passed else fp32_path
    result = {
        "selected": selected,
        "selected_file": selected_path.name,
        "input_shape": [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS],
        "input_name": (int8_io_names if int8_passed else io_names)["input_name"],
        "output_name": (int8_io_names if int8_passed else io_names)["output_name"],
        "reference_correct_counts": reference_counts,
        "float32_keras_validation": float32_keras_metrics,
        "has_no_loss_candidate": fp32_passed or int8_passed,
        "candidates": {
            "fp32": {
                "file": fp32_path.name,
                "size_bytes": fp32_path.stat().st_size,
                "validation": fp32_metrics,
                "correct_counts": fp32_counts,
                "no_measured_accuracy_loss": fp32_passed,
            },
            "int8": {
                "file": int8_path.name,
                "size_bytes": int8_path.stat().st_size if int8_path.exists() else None,
                "validation": int8_metrics,
                "correct_counts": int8_counts,
                "no_measured_accuracy_loss": int8_passed,
                "error": int8_error,
            },
        },
    }
    print(json.dumps({"onnx_export": result}, indent=2))
    return result


def write_metadata(
    output_dir: Path,
    model: keras.Model,
    validation_metrics: dict,
    tfjs_export: dict | None,
    onnx_export: dict | None,
) -> None:
    tfjs_dir = output_dir / "tfjs"
    metadata = {
        "format_version": 1,
        "alphabet": MODEL_ALPHABET,
        "case_sensitive": False,
        "captcha_length": CAPTCHA_LENGTH,
        "input": {
            "height": IMAGE_HEIGHT,
            "width": IMAGE_WIDTH,
            "channels": IMAGE_CHANNELS,
            "normalization": "divide_by_255",
        },
        "parameter_count": model.count_params(),
        "weight_quantization": tfjs_export["selected"] if tfjs_export else None,
        "validation": {key: float(value) for key, value in validation_metrics.items()},
        "tfjs_export": tfjs_export,
        "onnx_export": onnx_export,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if tfjs_dir.exists():
        shutil.copy2(output_dir / "metadata.json", tfjs_dir / "metadata.json")


def main() -> None:
    args = parse_args()
    configure_runtime()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)

    train_data, validation_data = load_splits(
        args.dataset, args.validation_size, args.max_train_samples
    )
    validation_dataset = to_tf_dataset(
        validation_data, args.batch_size, training=False, cache_dir=cache_dir
    )

    keras_path = output_dir / "luogu-captcha-mobile.h5"
    if args.compress_only:
        if not keras_path.exists():
            raise FileNotFoundError(f"Existing Keras model not found: {keras_path}")
        model = keras.models.load_model(keras_path, compile=False)
        compile_model(model)
        metrics = {
            key: float(value)
            for key, value in model.evaluate(
                validation_dataset, return_dict=True, verbose=2
            ).items()
        }
        if args.skip_tfjs:
            tfjs_export = validate_existing_tfjs(
                model,
                output_dir,
                validation_dataset,
                metrics,
                len(validation_data),
            )
        else:
            tfjs_export = export_smallest_accurate_tfjs(
                model,
                keras_path,
                output_dir,
                validation_dataset,
                metrics,
                len(validation_data),
            )
        write_metadata(output_dir, model, metrics, tfjs_export, None)
        onnx_export = None
        if not args.skip_onnx:
            onnx_export = export_onnx(
                model,
                output_dir,
                validation_dataset,
                metrics,
                len(validation_data),
            )
        write_metadata(output_dir, model, metrics, tfjs_export, onnx_export)
        print(json.dumps(tfjs_export, indent=2))
        print(f"Artifacts written to {output_dir.resolve()}")
        return

    train_dataset = to_tf_dataset(
        train_data, args.batch_size, training=True, cache_dir=cache_dir
    )

    model = build_model()
    compile_model(model)
    model.summary()
    print(
        f"Parameters: {model.count_params():,} "
        f"({model.count_params() * 4 / 1_000_000:.2f} MB fp32)"
    )

    checkpoint_path = output_dir / "best.weights.h5"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_captcha_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_captcha_accuracy",
            mode="max",
            patience=6,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_captcha_accuracy",
            mode="max",
            factor=0.3,
            patience=2,
            min_lr=1e-5,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(output_dir / "training.csv"),
        keras.callbacks.TerminateOnNaN(),
    ]
    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    model.load_weights(checkpoint_path)
    metrics = model.evaluate(validation_dataset, return_dict=True, verbose=2)
    model.save(keras_path, include_optimizer=False)
    checkpoint_path.unlink(missing_ok=True)
    tfjs_export = None
    if not args.skip_tfjs:
        tfjs_export = export_smallest_accurate_tfjs(
            model,
            keras_path,
            output_dir,
            validation_dataset,
            metrics,
            len(validation_data),
        )
    onnx_export = None
    if not args.skip_onnx:
        onnx_export = export_onnx(
            model,
            output_dir,
            validation_dataset,
            metrics,
            len(validation_data),
        )
    write_metadata(output_dir, model, metrics, tfjs_export, onnx_export)
    print(json.dumps(metrics, indent=2))
    print(f"Artifacts written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
