#!/usr/bin/env python3
"""
preview.py — randomly preview samples from the Luogu-captcha TFRecord dataset,
either from a local folder (as produced by generate.py) or downloaded on the
fly from a Hugging Face Hub dataset repo (as used in the training notebook).

Shard naming / format assumptions (must match generate.py):
  - files named part_00000.tfrecord, part_00001.tfrecord, ...
  - each record has:
      "image": bytes  (raw JPEG bytes, 90x35)
      "label": int64 list of length 4 (ASCII codes of the 4 captcha chars)

Usage examples:
  # preview 24 random samples from a local folder
  python preview.py --source local --data-dir data --count 24 --out preview.png

  # preview 24 random samples, downloading shards on demand from HF Hub
  python preview.py --source hub --repo-id langningchen/luogu-captcha-dataset \
      --count 24 --out preview.png

  # just dump full-size individual images instead of a grid
  python preview.py --source local --data-dir data --count 10 --mode files --out-dir samples/
"""
import argparse
import io
import math
import random
from pathlib import Path

import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont

CAPTCHA_LENGTH = 4
SAMPLES_PER_TFRECORD = 5000


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["local", "hub"], default="local",
                     help="load shards from a local folder, or download from HF Hub")
    ap.add_argument("--data-dir", default="data",
                     help="local folder containing part_*.tfrecord (source=local), "
                          "or local cache folder to download into (source=hub)")
    ap.add_argument("--repo-id", default="langningchen/luogu-captcha-dataset",
                     help="HF Hub dataset repo id (source=hub only)")
    ap.add_argument("--num-shards", type=int, default=1,
                     help="how many shard files to sample from (source=hub: how many "
                          "to download; source=local: how many of the found files to use, "
                          "0 = all)")
    ap.add_argument("--count", type=int, default=24, help="number of samples to preview")
    ap.add_argument("--seed", type=int, default=None, help="random seed (default: random)")
    ap.add_argument("--mode", choices=["grid", "files"], default="grid",
                     help="grid: one combined image; files: dump individual jpgs")
    ap.add_argument("--out", default="preview.png", help="output path for grid mode")
    ap.add_argument("--out-dir", default="preview_samples",
                     help="output dir for files mode")
    ap.add_argument("--cols", type=int, default=6, help="columns in grid mode")
    ap.add_argument("--scale", type=int, default=4,
                     help="upscale factor for each 90x35 image in grid mode "
                          "(captchas are tiny, this makes them legible)")
    return ap.parse_args()


def find_local_shards(data_dir: Path, num_shards: int):
    shards = sorted(data_dir.glob("part_*.tfrecord"))
    if not shards:
        raise SystemExit(f"No part_*.tfrecord files found in {data_dir}")
    if num_shards and num_shards > 0:
        shards = shards[:num_shards]
    return [str(p) for p in shards]


def download_hub_shards(repo_id: str, cache_dir: Path, num_shards: int, count: int):
    from huggingface_hub import hf_hub_download

    # we don't know total dataset size up front without hitting the API, so just
    # grab enough shards to comfortably satisfy `count` random samples, unless the
    # user explicitly asked for more via --num-shards.
    needed = max(num_shards, math.ceil(count / SAMPLES_PER_TFRECORD) + 1)
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(needed):
        fname = f"part_{i:05d}.tfrecord"
        local_path = cache_dir / fname
        if not local_path.exists():
            print(f"Downloading {fname} ...")
            try:
                downloaded = hf_hub_download(
                    repo_id=repo_id, repo_type="dataset", filename=fname,
                    local_dir=str(cache_dir),
                )
                paths.append(downloaded)
            except Exception as e:
                print(f"  stopped at shard {i} ({e}); using {len(paths)} shard(s)")
                break
        else:
            paths.append(str(local_path))
    if not paths:
        raise SystemExit("Failed to obtain any shards from the Hub")
    return paths


def parse_record(example_proto):
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([CAPTCHA_LENGTH], tf.int64),
    }
    parsed = tf.io.parse_single_example(example_proto, features)
    label = "".join(chr(c) for c in parsed["label"].numpy())
    return parsed["image"].numpy(), label


def reservoir_sample(shards, k, seed):
    """
    Single-pass reservoir sampling over all records across all shards so we don't
    need to materialize the whole dataset in memory, and every record has an
    equal chance of being picked regardless of shard/order.
    """
    rng = random.Random(seed)
    ds = tf.data.TFRecordDataset(shards, num_parallel_reads=tf.data.AUTOTUNE)

    reservoir = []
    n_seen = 0
    for rec in ds:
        img_bytes, label = parse_record(rec)
        n_seen += 1
        if len(reservoir) < k:
            reservoir.append((img_bytes, label))
        else:
            j = rng.randint(0, n_seen - 1)
            if j < k:
                reservoir[j] = (img_bytes, label)
    if n_seen == 0:
        raise SystemExit("No records found in the given shard(s)")
    if n_seen < k:
        print(f"Warning: only {n_seen} records available, requested {k}")
    return reservoir


def make_grid(samples, cols, scale, out_path):
    n = len(samples)
    rows = math.ceil(n / cols)

    imgs = []
    for img_bytes, label in samples:
        im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
        imgs.append((im, label))

    cell_w, cell_h = imgs[0][0].size
    pad = 8
    label_h = 20
    canvas_w = cols * (cell_w + pad) + pad
    canvas_h = rows * (cell_h + label_h + pad) + pad
    canvas = Image.new("RGB", (canvas_w, canvas_h), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", 16)
    except Exception:
        font = ImageFont.load_default()

    for idx, (im, label) in enumerate(imgs):
        r, c = divmod(idx, cols)
        x = pad + c * (cell_w + pad)
        y = pad + r * (cell_h + label_h + pad)
        canvas.paste(im, (x, y))
        draw.text((x, y + cell_h + 2), label, fill=(255, 255, 0), font=font)

    canvas.save(out_path)
    print(f"Saved grid preview ({n} samples) to {out_path}")


def dump_files(samples, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (img_bytes, label) in enumerate(samples):
        fname = out_dir / f"{i:03d}_{label}.jpg"
        fname.write_bytes(img_bytes)
    print(f"Saved {len(samples)} individual images to {out_dir}/")


def main():
    args = parse_args()
    seed = args.seed if args.seed is not None else random.randrange(2**31)
    print(f"Using random seed: {seed}")

    data_dir = Path(args.data_dir)
    if args.source == "local":
        shards = find_local_shards(data_dir, args.num_shards)
    else:
        shards = download_hub_shards(args.repo_id, data_dir, args.num_shards, args.count)

    print(f"Sampling {args.count} record(s) from {len(shards)} shard(s):")
    for s in shards:
        print(f"  {s}")

    samples = reservoir_sample(shards, args.count, seed)

    # print labels to console too, handy for quickly eyeballing dup patterns
    for i, (_, label) in enumerate(samples):
        print(f"  [{i:03d}] {label}")

    if args.mode == "grid":
        make_grid(samples, args.cols, args.scale, args.out)
    else:
        dump_files(samples, args.out_dir)


if __name__ == "__main__":
    main()
