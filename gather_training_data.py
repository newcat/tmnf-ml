import os
import cv2
import numpy as np
from imagegrab import get_tm_hwnd, get_img
import presskey as pk

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

VK_Q = 0x51
VK_R = 0x52
VK_SPACE = 0x020

WIDTH = 200
HEIGHT = 150


def get_action():
    pressed = {}
    pressed[pk.LEFT_ARROW] = pk.GetKey(pk.LEFT_ARROW)
    pressed[pk.UP_ARROW] = pk.GetKey(pk.UP_ARROW)
    pressed[pk.RIGHT_ARROW] = pk.GetKey(pk.RIGHT_ARROW)
    pressed[pk.DOWN_ARROW] = pk.GetKey(pk.DOWN_ARROW) or pk.GetKey(VK_SPACE, is_vk=True)

    scores = []
    for i in range(len(ACTIONS)):
        action_valid = True
        for k in ACTIONS[i]:
            if not pressed[k]:
                action_valid = False
                break
        scores.append(len(ACTIONS[i]) if action_valid else 0)

    return np.argmax(scores)


if __name__ == "__main__":

    # wait for recording start command
    while True:
        if pk.GetKey(VK_R, is_vk=True):
            break
        elif pk.GetKey(VK_Q, is_vk=True):
            exit(0)

    # Record
    print("Recording")
    hwnd = get_tm_hwnd()
    recorded_images = []
    recorded_actions = []
    while True:

        if pk.GetKey(VK_Q, is_vk=True):
            break

        img = get_img(hwnd)
        img = cv2.resize(img, (WIDTH, HEIGHT))
        recorded_images.append(img)
        recorded_actions.append(get_action())

    assert len(recorded_images) == len(recorded_actions)

    print("Recording stopped")
    print(f"Recorded {len(recorded_images)} frames")

    def onTrackbarUpdate(pos):
        cv2.imshow("win", recorded_images[pos])

    win = cv2.namedWindow("win", flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_EXPANDED)
    cv2.createTrackbar("Position", "win", 0, len(recorded_images) - 1, onTrackbarUpdate)
    cv2.imshow("win", recorded_images[0])

    beginning = 0
    end = len(recorded_images) - 1

    while True:
        key = cv2.waitKey(100) & 0xFF
        if key == ord('b'):
            beginning = cv2.getTrackbarPos("Position", "win")
            print(f"Set beginning to {beginning}")
        if key == ord('e'):
            end = cv2.getTrackbarPos("Position", "win")
            print(f"Set end to {end}")
        if key == ord('s'):
            cv2.destroyAllWindows()
            break

    recorded_images = recorded_images[beginning:end+1]
    recorded_actions = recorded_actions[beginning:end+1]

    name = input("Name? ").strip()
    output_file = os.path.join("training", name + ".npz")
    with open(output_file, "wb") as f:
        np.savez_compressed(output_file, images=recorded_images, actions=recorded_actions)

    print(f"Saved {output_file}")
