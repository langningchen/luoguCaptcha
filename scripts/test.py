"""
Interactive captcha tester that pulls images from Luogu API and shows
the model's top predictions for manual inspection.

Usage:
  python scripts/test.py [--top 10] [--interval 0.0]

Controls while running:
  Enter: fetch next captcha
  q + Enter: quit
  s + Enter: save current captcha image to ./captchas/<timestamp>.png
"""

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

import argparse
import io
import os
import sys
import time
from datetime import datetime
from itertools import product
from typing import List, Tuple

import numpy as np
import requests
import tensorflow as tf
from keras.models import load_model
from matplotlib import pyplot as plt
from PIL import Image


# Model/input config (must match training/predict.py)
IMG_HEIGHT, IMG_WIDTH = 35, 90
CHARS_PER_LABEL = 4
CHAR_SIZE = 256  # per predict.py
MODEL_PATH = os.path.join("models", "luoguCaptcha.keras")
LUOGU_CAPTCHA_URL = "https://www.luogu.com.cn/api/verify/captcha"


def setup_device():
    physical_devices = tf.config.list_physical_devices("GPU")
    if physical_devices:
        try:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
            print("Using GPU:", physical_devices[0])
        except Exception:
            print(
                "GPU available but failed to set memory growth; using default settings."
            )
    else:
        print("Using CPU")


def load_model_or_exit(path: str):
    try:
        return load_model(path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Please ensure the model exists at {path}.")
        sys.exit(1)


def fetch_captcha(timeout: float = 10.0) -> Image.Image:
    """Fetch captcha image from Luogu API (returns PIL Image)."""
    # Add minimal headers to simulate a browser; some services require a Referer/User-Agent
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Referer": "https://www.luogu.com.cn/lg4/captcha",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    }
    r = requests.get(LUOGU_CAPTCHA_URL, headers=headers, timeout=timeout)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("L")  # grayscale
    return img


def preprocess(img: Image.Image) -> np.ndarray:
    """Convert PIL image to model input shape (1, H, W, 1) normalized to [0,1]."""
    # In case the fetched size differs, resize to training size
    if img.size != (IMG_WIDTH, IMG_HEIGHT):
        img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    x = np.array(img, dtype=np.float32) / 255.0
    x = np.expand_dims(x, axis=-1)  # (H, W, 1)
    x = np.expand_dims(x, axis=0)  # (1, H, W, 1)
    return x


def topk_per_position(prob: np.ndarray, k: int) -> List[List[Tuple[int, float]]]:
    """Return top-k (index, prob) per of the 4 positions.

    prob shape: (4, CHAR_SIZE)
    """
    tops: List[List[Tuple[int, float]]] = []
    for i in range(CHARS_PER_LABEL):
        p = prob[i]
        # argsort ascending; take last k; reverse to descending
        idxs = np.argsort(p)[-k:][::-1]
        tops.append([(int(idx), float(p[idx])) for idx in idxs])
    return tops


def combine_topk(
    tops: List[List[Tuple[int, float]]], max_results: int
) -> List[Tuple[str, float]]:
    """Combine per-position top-k into global top strings sorted by product prob.

    Returns list of (string, score) with scores as product of per-position probs.
    We compute in log-space for numerical stability, then exp for display.
    """
    candidates: List[Tuple[str, float]] = []
    for combo in product(*tops):
        chars = [chr(i) for i, _ in combo]
        probs = [max(1e-12, p) for _, p in combo]
        logp = float(np.sum(np.log(probs)))
        candidates.append(("".join(chars), logp))
    # sort by log prob desc
    candidates.sort(key=lambda x: x[1], reverse=True)
    # convert log prob to prob for display
    results = [(s, float(np.exp(lp))) for s, lp in candidates[:max_results]]
    return results


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Luogu captcha manual tester")
    parser.add_argument(
        "--top", type=int, default=10, help="Top-N combined predictions to show"
    )
    parser.add_argument(
        "--k", type=int, default=5, help="Top-k per position to consider when combining"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.0,
        help="Auto-advance interval seconds; 0 to wait for input",
    )
    parser.add_argument(
        "--save-dir",
        default="captchas",
        help="Directory to save images when pressing 's'",
    )
    args = parser.parse_args()

    setup_device()
    model = load_model_or_exit(MODEL_PATH)

    # Prepare interactive plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(4, 2))
    img_artist = None
    ax.axis("off")
    try:
        fig.canvas.manager.set_window_title("Luogu Captcha Tester")
    except Exception:
        pass

    ensure_dir(args.save_dir)

    idx = 0
    try:
        while True:
            idx += 1
            # 1) Fetch image
            try:
                pil_img = fetch_captcha()
            except Exception as e:
                print(f"[#{idx}] Fetch failed: {e}")
                if args.interval > 0:
                    time.sleep(min(args.interval, 2.0))
                else:
                    input("Press Enter to retry...")
                continue

            # 2) Show image
            disp_img = np.array(pil_img)
            if img_artist is None:
                img_artist = ax.imshow(disp_img, cmap="gray")
            else:
                img_artist.set_data(disp_img)
            ax.set_title(f"Captcha #{idx}")
            fig.canvas.draw_idle()
            plt.pause(0.001)

            # 3) Predict
            x = preprocess(pil_img)
            probs = model.predict(x, verbose=0)  # shape: (1, 4, 256)
            probs = np.squeeze(probs, axis=0)

            # 4) Top-k per position and combined top-N
            per_pos = topk_per_position(probs, args.k)
            combined = combine_topk(per_pos, args.top)

            # 5) Print results
            print(f"\n[#{idx}] Top-{args.top} predictions (probability):")
            for rank, (s, p) in enumerate(combined, 1):
                print(f"  {rank:2d}. {s}  (p={p:.8f})")

            # 6) Wait for user or auto-advance
            if args.interval > 0:
                time.sleep(args.interval)
                continue

            cmd = input("[Enter=next | s=save | q=quit] > ").strip().lower()
            if cmd == "q":
                break
            if cmd == "s":
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                path = os.path.join(args.save_dir, f"captcha_{ts}.png")
                pil_img.save(path)
                print(f"Saved to {path}")
            # else: fetch next

    except KeyboardInterrupt:
        print("\nInterrupted. Bye.")
    finally:
        plt.ioff()
        try:
            plt.close(fig)
        except Exception:
            pass


if __name__ == "__main__":
    main()
