# Luogu Captcha Predict

[![Open training notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/langningchen/luoguCaptcha/blob/main/notebooks/train_colab.ipynb)

Recognize Luogu's four-character captchas locally in the browser. The userscript uses TensorFlow.js and does not send captcha images to a prediction server.

## Browser Usage

1. Install [Tampermonkey](https://www.tampermonkey.net/) or another userscript manager.
2. Install [`predict.user.js`](https://github.com/langningchen/luoguCaptcha/raw/refs/heads/main/predict.user.js).
3. Open a Luogu page containing a captcha. The model is embedded in the userscript and is available without a model-file download.

The canonical browser artifacts are published in [`langningchen/luogu-captcha-model`](https://huggingface.co/langningchen/luogu-captcha-model/tree/main/tfjs). The userscript embeds the validated TensorFlow.js topology and shard directly, so it does not request model files from Hugging Face at runtime. TensorFlow.js itself is loaded through the userscript manager's `@require` dependency.

## Browser Model

The browser model is designed around the real captcha generator and the fixed `90x35` input:

- RGB input, matching the colored training data
- 35 case-insensitive output classes instead of a 256-class ASCII head
- standard convolution blocks and a 512-unit dense head, prioritizing accuracy
- no recurrent, attention, or custom layers, keeping TensorFlow.js compatibility simple
- exact four-character accuracy as the checkpoint metric
- validated post-training quantization for browser delivery

The network contains 3,245,804 parameters: about 13.0 MB in FP32, 6.5 MB with uint16 weights, or 3.25 MB with uint8 weights, before the small TensorFlow.js model manifest. The published mixed-precision browser artifact is 3,279,923 bytes including its manifest and metadata. The generated userscript is about 4.37 MB because Base64 adds transport overhead, and decodes the 3,249,784-byte shard in memory. The exporter tries uint8 first and keeps it only when both character and full-captcha correct counts do not decrease on the 10,000-image validation split; otherwise it selectively restores sensitive tensors to uint16 before falling back to full uint16 or FP32. Uppercase and lowercase samples share one target class because Luogu verifies captchas case-insensitively; the browser outputs lowercase letters.

To regenerate the embedded block after publishing a replacement model, run:

```bash
python scripts/embed_tfjs_model.py /path/to/tfjs \
  --revision <hugging-face-commit>
```

The same checkpoint is also exported to ONNX with a dynamic batch dimension and NHWC float32 input shaped `[N, 35, 90, 3]`. Both FP32 and dynamic INT8 files are validated with ONNX Runtime, and `metadata.json` identifies the recommended artifact.

## Colab Training

The dataset is [`langningchen/luogu-captcha-dataset-colored`](https://huggingface.co/datasets/langningchen/luogu-captcha-dataset-colored): 500,000 colored captcha images with string labels.

Open [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb), select a T4 GPU runtime, and run all cells. The notebook:

1. downloads and splits the Hugging Face dataset;
2. decodes the JPEGs once into a local uint8 TensorFlow cache, then trains with mixed precision and adaptive learning-rate reduction;
3. keeps the checkpoint with the best full-captcha validation accuracy;
4. exports the smallest TensorFlow.js artifact that shows no validation-accuracy loss;
5. exports and validates FP32 and dynamic INT8 ONNX artifacts;
6. optionally publishes the artifacts when an `HF_TOKEN` Colab secret is present.

The same pipeline can be run from a GPU machine:

```bash
python scripts/train.py \
  --epochs 30 \
  --batch-size 512 \
  --validation-size 10000 \
  --output-dir models/browser
```

For a short integration test, add `--max-train-samples 4096 --epochs 1`.

## Generated Artifacts

```text
models/browser/
|-- luogu-captcha-mobile.h5
|-- metadata.json
|-- training.csv
|-- onnx/
|   |-- luogu-captcha-fp32.onnx
|   `-- luogu-captcha-int8.onnx
`-- tfjs/
    |-- model.json
    |-- group1-shard*.bin
    `-- metadata.json
```

The userscript and `metadata.json` must use the same alphabet and input preprocessing. The exported metadata records the training constants, and the same values are mirrored in [`predict.user.js`](predict.user.js).

Run the ONNX model from Python with:

```bash
python scripts/predict_onnx.py captcha.jpg \
  --model models/browser/onnx/luogu-captcha-fp32.onnx
```

Use `luogu-captcha-int8.onnx` instead when `metadata.json` marks INT8 as the selected ONNX artifact.

## Legacy Tools

The scripts under [`scripts/`](scripts/) for generating TFRecords, server-side Keras prediction, and manually testing fetched captchas remain available for the older models. The browser model uses the colored Hugging Face dataset directly and does not require those TFRecords.

## License

This project is licensed under the GNU General Public License v3.0. The colored dataset and published model repository declare AGPL-3.0 licensing.
