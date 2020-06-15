import numpy as np
from PIL import ImageGrab
import win32gui
import cv2

WINDOW_TITLE = "trackmania united forever"
BORDER_OFFSETS = (8, 32, 8, 7)


def screen_record(hwnd):
    while(True):
        img = get_img(hwnd)
        cv2.imshow('window', img)
        print(img)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break


def get_hwnd(target_title):
    toplist, winlist = [], []

    def enum_cb(hwnd, results):
        winlist.append((hwnd, win32gui.GetWindowText(hwnd)))

    win32gui.EnumWindows(enum_cb, toplist)

    window = [(hwnd, title) for hwnd, title in winlist if target_title.lower() in title.lower()]

    if len(window) == 0:
        print("No window found")
        exit(1)

    # just grab the hwnd for first window matching
    window = window[0]
    hwnd = window[0]

    return hwnd


def get_tm_hwnd():
    return get_hwnd(WINDOW_TITLE)


def get_img(hwnd):
    bbox = win32gui.GetWindowRect(hwnd)
    bbox = (bbox[0] + BORDER_OFFSETS[0], bbox[1] + BORDER_OFFSETS[1], bbox[2] - BORDER_OFFSETS[2], bbox[3] - BORDER_OFFSETS[3])
    printscreen = np.array(ImageGrab.grab(bbox=bbox))
    return cv2.cvtColor(printscreen, cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    hwnd = get_tm_hwnd()
    win32gui.SetForegroundWindow(hwnd)
    screen_record(hwnd)
