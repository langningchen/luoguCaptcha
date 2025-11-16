# Luogu Captcha Predict

## Introduction

Recognize [Luogu Captcha](https://www.luogu.com.cn/lg4/captcha) using an AI model.

## Usage

1. Ensure you have [TamperMonkey](https://www.tampermonkey.net/) or another UserScript manager installed in your browser.
2. Install the UserScript by downloading the file [predict.user.js](https://github.com/langningchen/luoguCaptcha/raw/refs/heads/main/predict.user.js).

## Model and Dataset

The [Model](https://huggingface.co/langningchen/luogu-captcha-model) and [Dataset](https://huggingface.co/datasets/langningchen/luogu-captcha-dataset) are hosted on HuggingFace.

### Current Model Information

- Training data is located in the [`data`](data) folder.
  **Note: The folder contains 1000 files, so exercise caution when opening it in your browser!**
- Training history visualization:
![Training history](trainHistory.png)

### Data Generator

The data generator consists of [`generate.php`](generate.php) and its Python wrapper [`generate.py`](generate.py).

- `generate.php`
  - Without arguments: Generates a captcha, outputs the captcha answer to `stdout`, and saves the image as `captcha.jpg`.
  - With two arguments (`tot`, `seed`): Both arguments must be integers. The program generates `tot` images using the random seed `seed`, concatenates all image data, and outputs it to `stdout`. Each image is formatted as follows:
    - First 2 bytes (`len`): Length of the image data
    - Next 4 bytes: Captcha answer
    - Next `len` bytes: Binary image data
- `generate.py`
  - Requires three arguments: `TotalImages` and `WorkersCount`.
    It generates `TotalImages` image batch files in the [`data`](data) directory, formatted for HuggingFace (`data/luogu_captcha_dataset`) and TensorFlow (`data/luogu_captcha_tfrecord`).

### Model Training

The script [`train.py`](train.py) trains the model using TensorFlow with data from the `data/luogu_captcha_tfrecord` folder. The trained model is saved as `models/luoguCaptcha.keras`.

### Predicting Captchas

The script [`predict.py`](predict.py) is used for captcha prediction.

- With one argument (`port`): Starts an HTTP server on the specified port, providing a single API endpoint.
  - **URL**: `/`
  - **Request Method**: `POST`
  - **Request Body**: JSON in the following format:
    ```json
    {
        "image": "base64 encoded image file"
    }
    ```
  - **Response Body**: JSON in the following format:
    ```json
    {
        "prediction": "the captcha answer"
    }
    ```

## License

This project is licensed under the terms of the GNU General Public License v3.0.
