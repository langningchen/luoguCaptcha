# predict.py
import numpy as np
import tensorflow as tf
from keras.api.models import load_model
from PIL import Image
import sys
import os
import http.server
import io
import json
import base64

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

model_path = "/home/cyezoi/luoguCaptcha/luoguCaptcha.keras"

# 加载模型
try:
    model = load_model(model_path)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    print(f"Please ensure the model is saved at {model_path}")
    sys.exit(1)


def predict_captcha(image_path):
    """预测单个图像文件"""
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        sys.exit(1)

    # 预处理输入图像 (与 generate.py 保持一致)
    # 灰度转换 -> NumPy -> 归一化 -> 添加通道 -> 添加 Batch 维度
    # 目标形状: (1, 35, 90, 1)
    image_np = np.array(img.convert("L"), dtype=np.float32) / 255.0
    image_np = np.expand_dims(image_np, axis=-1)
    image_np = np.expand_dims(image_np, axis=0)  # 添加 Batch 维度

    # 预测
    pred_probabilities = model.predict(image_np)

    # 解码：找到每个字符位置的最高概率索引 (ASCII 值)
    # pred_probabilities 形状: (1, 4, 256)
    predicted_ascii_codes = tf.math.argmax(pred_probabilities, axis=-1).numpy()[0]

    # 转换为字符
    predicted_captcha = "".join(map(chr, predicted_ascii_codes))
    return predicted_captcha


if __name__ == "__main__":
    if len(sys.argv) == 2 and not sys.argv[1].isdigit():
        port = int(sys.argv[1])
        print(f"Starting HTTP server on port {port}...")

        class CaptchaHandler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                try:
                    # 读取请求数据
                    length = int(self.headers["Content-Length"])
                    data = json.loads(self.rfile.read(length))

                    # 解码 base64 图像
                    image_data = base64.b64decode(data["image"])
                    image = Image.open(io.BytesIO(image_data))

                    # 预处理图像
                    image_np = np.array(image.convert("L"), dtype=np.float32) / 255.0
                    image_np = np.expand_dims(image_np, axis=-1)
                    image_np = np.expand_dims(image_np, axis=0)

                    # 预测
                    pred_probabilities = model.predict(image_np)
                    predicted_ascii_codes = tf.math.argmax(
                        pred_probabilities, axis=-1
                    ).numpy()[0]
                    predicted_captcha = "".join(map(chr, predicted_ascii_codes))

                    # 返回 JSON 响应
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    response = json.dumps({"prediction": predicted_captcha})
                    self.wfile.write(response.encode())

                except Exception as e:
                    # 错误处理
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    error_response = json.dumps({"error": str(e)})
                    self.wfile.write(error_response.encode())

        # 启动服务器
        server = http.server.HTTPServer(("", port), CaptchaHandler)
        print(f"Server running at http://localhost:{port}")
        print('Send POST request with JSON: {"image": "base64_encoded_image"}')
        server.serve_forever()

    else:
        print("Usage:")
        print("  python predict.py <image_path>          # Predict single image")
        print("  python predict.py <port>                # Start HTTP server")
        print("                                           (port must be a number)")
        print("")
        print("HTTP Server Usage:")
        print("  POST http://localhost:<port>/")
        print('  Body: {"image": "base64_encoded_image"}')
        print('  Response: {"prediction": "captcha_text"}')
