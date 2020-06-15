from imagegrab import get_tm_hwnd, get_img
import presskey as pk
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
import numpy as np
from scipy.special import softmax
import cv2
import time
import random

WIDTH = 200
HEIGHT = 150
EXPLORATION_RATE = 0.1

ACTIONS = [
    [],
    [pk.UP_ARROW],
    [pk.UP_ARROW, pk.LEFT_ARROW],
    [pk.UP_ARROW, pk.RIGHT_ARROW],
    [pk.DOWN_ARROW],
    [pk.DOWN_ARROW, pk.LEFT_ARROW],
    [pk.DOWN_ARROW, pk.RIGHT_ARROW]
]

model = Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(WIDTH, HEIGHT, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(7))


def main():

    time.sleep(3)
    hwnd = get_tm_hwnd()
    current_pressed_keys = []

    while True:
        img = get_img(hwnd)
        img = cv2.resize(img, (WIDTH, HEIGHT))
        img = img.reshape(1, WIDTH, HEIGHT, 3)
        prediction = softmax(model.predict(img)[0])
        action = np.argmax(prediction)

        if random.random() < EXPLORATION_RATE:
            action = random.randrange(0, len(ACTIONS))

        for key in current_pressed_keys:
            pk.ReleaseKey(key)

        current_pressed_keys = ACTIONS[action]
        for key in current_pressed_keys:
            pk.PressKey(key)

        print(action)


main()
