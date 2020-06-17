from os import listdir
from os.path import isfile, join

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import matplotlib.pyplot as plt

import presskey as pk
from imagegrab import get_tm_hwnd, get_img

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

WIDTH = 200
HEIGHT = 150

ACTIONS = [
    [],
    [pk.LEFT_ARROW],
    [pk.RIGHT_ARROW],
    [pk.UP_ARROW],
    [pk.UP_ARROW, pk.LEFT_ARROW],
    [pk.UP_ARROW, pk.RIGHT_ARROW],
    [pk.DOWN_ARROW],
    [pk.DOWN_ARROW, pk.LEFT_ARROW],
    [pk.DOWN_ARROW, pk.RIGHT_ARROW],
    [pk.UP_ARROW, pk.DOWN_ARROW],
    [pk.UP_ARROW, pk.DOWN_ARROW, pk.LEFT_ARROW],
    [pk.UP_ARROW, pk.DOWN_ARROW, pk.RIGHT_ARROW]
]

# left, up, right, down
NEW_ACTIONS = [
    [0, 0, 0, 0],
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, 1, 0, 0],
    [1, 1, 0, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 1],
    [1, 0, 0, 1],
    [0, 0, 1, 1],
    [0, 1, 0, 1],
    [1, 1, 0, 1],
    [0, 1, 1, 1]
]

model = Sequential()
model.add(Conv2D(32, kernel_size=3, activation="relu", input_shape=(WIDTH, HEIGHT, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, kernel_size=3, activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, kernel_size=5, activation="relu"))
model.add(MaxPooling2D((3, 3)))
model.add(Conv2D(128, kernel_size=5, activation="relu"))
model.add(MaxPooling2D((3, 3)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(4, activation="sigmoid"))
model.summary()


def load_training_data():
    training_files = [f for f in listdir("training") if isfile(join("training", f))]
    X = None
    y = None
    for f in training_files:
        a = np.load(join("training", f))
        X = np.concatenate((X, a["images"]), axis=0) if X is not None else a["images"]
        y = np.concatenate((y, a["actions"]), axis=0) if y is not None else a["actions"]

    X = X.swapaxes(1, 2)

    y_transformed = []
    for action in y:
        y_transformed.append(NEW_ACTIONS[action])

    y = np.array(y_transformed)
    print(y[0])

    return train_test_split(X, y, test_size=0.2)


def main():
    X_train, X_test, y_train, y_test = load_training_data()

    model.compile(optimizer="adam", loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=6)

    input("Ready?")

    fig = plt.figure()
    ax = fig.add_subplot(111)

    def update_graph(data):
        ax.clear()
        ax.set_ylim([0, 1])
        ax.bar(list(range(4)), data)
        plt.xticks(list(range(4)), ('Left', 'Up', 'Right', 'Down'))
        plt.pause(0.001)

    update_graph([0.5, 0.5, 0.5, 0.5])
    fig.show()
    plt.pause(3)

    hwnd = get_tm_hwnd()
    while True:
        img = get_img(hwnd)
        img = cv2.resize(img, (WIDTH, HEIGHT))
        img = img.reshape((1, WIDTH, HEIGHT, 3))
        y_pred = model.predict(img)[0]
        update_graph(y_pred)

        # for i, k in enumerate([pk.LEFT_ARROW, pk.UP_ARROW, pk.RIGHT_ARROW, pk.DOWN_ARROW]):
        #     if y_pred[i] > 0.5:
        #         pk.PressKey(k)
        #     else:
        #         pk.ReleaseKey(k)


if __name__ == "__main__":
    main()
