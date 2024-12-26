import numpy as np
import tensorflow as tf
from keras.api.models import load_model
from PIL import Image
import sys

import http.server
import io
import json
import base64

if __name__ == "__main__":
    Model = load_model("luoguCaptcha.keras")
    Model.summary()

    if len(sys.argv) == 1:
        Image = Image.open("./captcha.jpg")
        ImageArray = np.array(Image) / 255.0

        prediction = Model.predict(np.array([ImageArray]))
        prediction = tf.math.argmax(prediction, axis=-1)

        print("".join(map(chr, map(int, prediction[0]))))
        # print(prediction[0])

    else:
        # 运行 HTTP 服务器，接收 POST 请求为图片，返回预测结果
        # 端口号为第一个参数
        # 请求格式为 JSON，包含一个名为 "image" 的字符串，内容为 base64 编码的图片
        # 相应格式为 JSON，包含一个名为 "prediction" 的字符串，内容为预测结果
        Port = int(sys.argv[1])

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                Length = int(self.headers["Content-Length"])
                Data = json.loads(self.rfile.read(Length))
                image = Image.open(io.BytesIO(base64.b64decode(Data["image"])))
                ImageArray = np.array(image) / 255.0

                prediction = Model.predict(np.array([ImageArray]))
                prediction = tf.math.argmax(prediction, axis=-1)

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps(
                        {"prediction": "".join(map(chr, map(int, prediction[0])))}
                    ).encode()
                )

        http.server.HTTPServer(("", Port), Handler).serve_forever()
