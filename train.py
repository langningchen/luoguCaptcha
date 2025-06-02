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

from glob import glob

import matplotlib
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from keras import layers, models
from keras.api.utils import to_categorical

matplotlib.use("QtAgg")

CharSize = 128
CharsPerLabel = 4  # num_of_chars_per_image
EPOCHS = 8

# create a pandas data frame of images and labels
trainFiles = glob("data/Data_*.pkl")
trainIds, validationIds, testIds = np.split(
    np.random.permutation(len(trainFiles)),
    [int(0.8 * len(trainFiles)), int(0.9 * len(trainFiles))],
)


def GetDataGenerator(Files, indices, repeat=1):
    for _ in range(repeat):
        for i in indices:
            DataFrame = pd.read_pickle(Files[i])
            Images = np.array([Image for Image in DataFrame["Image"]]) / 255.0
            Labels = np.array(
                [
                    [
                        np.array(to_categorical(ord(LabelChar), CharSize))
                        for LabelChar in Label.lower()
                    ]
                    for Label in DataFrame["Label"]
                ]
            )
            yield Images, Labels


input = tf.keras.Input(shape=(35, 90, 3), name="input")
con1 = layers.Conv2D(32, 3, activation="relu", name="con1")(input)
pol1 = layers.MaxPooling2D((2, 2), name="pol1")(con1)
con2 = layers.Conv2D(64, 3, activation="relu", name="con2")(pol1)
pol2 = layers.MaxPooling2D((2, 2), name="pol2")(con2)
con3 = layers.Conv2D(128, 3, activation="relu", name="con3")(pol2)
pol3 = layers.MaxPooling2D((2, 2), name="pol3")(con3)
flat = layers.Flatten(name="flat")(pol3)
drp = layers.Dropout(0.5, name="drp")(flat)
des1 = layers.Dense(1024, activation="relu", name="des1")(drp)
des2 = layers.Dense(CharsPerLabel * CharSize, activation="softmax", name="des2")(des1)
output = layers.Reshape((CharsPerLabel, CharSize), name="output")(des2)

Model = models.Model(inputs=input, outputs=output, name="LuoguCaptcha")
Model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
Model.summary()

trainHistory = Model.fit(
    GetDataGenerator(trainFiles, trainIds, EPOCHS),
    steps_per_epoch=len(trainIds),
    epochs=EPOCHS,
    validation_data=GetDataGenerator(trainFiles, validationIds, EPOCHS),
    validation_steps=len(validationIds),
    callbacks=[
        tf.keras.callbacks.TensorBoard(histogram_freq=1, log_dir="logs"),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=3, min_lr=0.0001
        ),
    ],
)

_, Axes = plt.subplots(1, 2, figsize=(15, 5))

Axes[0].plot(trainHistory.history["accuracy"], label="Training accuracy")
Axes[0].plot(trainHistory.history["val_accuracy"], label="Validation accuracy")
Axes[0].set_xlabel("Epochs")
Axes[0].legend()

Axes[1].plot(trainHistory.history["loss"], label="Training loss")
Axes[1].plot(trainHistory.history["val_loss"], label="Validation loss")
Axes[1].set_xlabel("Epochs")
Axes[1].legend()

print(pd.DataFrame(trainHistory.history))
plt.savefig("trainHistory.png")
plt.show()

# evaluate loss and accuracy in test dataset
TestDataGenerator = GetDataGenerator(trainFiles, testIds)
Loss, Accuracy = Model.evaluate(TestDataGenerator, steps=len(testIds))
print("Test loss: %.4f, accuracy: %.2f%%" % (Loss, Accuracy * 100))

Model.save("luoguCaptcha.keras")
