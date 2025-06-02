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

import re
import subprocess
import threading
from io import BytesIO
from math import ceil
from os import listdir, path, makedirs
from sys import argv

import pandas as pd
from numpy import array
from PIL import Image

DATA_DIR = "data"
if not path.exists(DATA_DIR):
    makedirs(DATA_DIR)


def Subprocess(GenerateNumber, WorkerID, ImageList):
    Process = subprocess.Popen(
        "php generate.php {}".format(GenerateNumber),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        shell=True,
    )
    for i in range(GenerateNumber):
        # if i * 100 // GenerateNumber != (i - 1) * 100 // GenerateNumber:
        #     print(f"Worker {WorkerID}: {i}/{GenerateNumber} ({i * 100 // GenerateNumber}%)")
        ImageSize = Process.stdout.read(2)
        Label = str(Process.stdout.read(4), encoding="utf-8")
        RealSize = ImageSize[0] * 256 + ImageSize[1]
        ImageData = array(Image.open(BytesIO(Process.stdout.read(RealSize))))
        ImageList.append((Label, ImageData))


if __name__ == "__main__":
    if len(argv) != 4:
        print("Usage: python generate.py <BatchNumber> <WorkersCount> <BatchSize>")
        exit(1)
    BatchNumber = int(argv[1])
    WorkersCount = int(argv[2])
    BatchSize = int(argv[3])
    GenerateNumber = BatchNumber * BatchSize
    EachWorkerGenerate = GenerateNumber // WorkersCount
    ImageLists = [[] for i in range(WorkersCount)]
    WorkerThreads = []
    for i in range(WorkersCount - 1):
        WorkerThreads.append(
            threading.Thread(
                target=Subprocess, args=(EachWorkerGenerate, i, ImageLists[i])
            )
        )
    WorkerThreads.append(
        threading.Thread(
            target=Subprocess,
            args=(
                GenerateNumber - EachWorkerGenerate * (WorkersCount - 1),
                WorkersCount - 1,
                ImageLists[-1],
            ),
        )
    )

    for Thread in WorkerThreads:
        Thread.start()
    for Thread in WorkerThreads:
        Thread.join()

    DataFrame = pd.DataFrame(
        [j for i in ImageLists for j in i], columns=["Label", "Image"]
    )
    ImageOffset = max(
        [
            int((re.match(r"Data_([0-9]+)\.pkl", f) or [None, -1])[1])
            for f in listdir(DATA_DIR)
        ]
        or [-1]
    )
    print(f"Succeed to generate {len(DataFrame)} images, offset: {ImageOffset}")
    for i in range(ceil(len(DataFrame) / BatchSize)):
        DataFrame.iloc[i * BatchSize : (i + 1) * BatchSize].to_pickle(
            path.join(DATA_DIR, f"Data_{i + ImageOffset + 1}.pkl")
        )
