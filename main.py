"""Standalone desktop entry point for MathCam.

The recommended experience is the Flask UI (`python app.py`), but this script is
kept for users who want a simple OpenCV window without the web interface.
"""

import os
from pathlib import Path

import cv2 as cv
import numpy as np

import handTrack as ht

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
CANVAS_PATH = BASE_DIR / "saved_canvas.jpg"
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
HEADER_HEIGHT = 125
BRUSH_THICKNESS = 15
ERASER_THICKNESS = 100


def load_overlays():
    overlays = []
    for image_path in sorted(ASSETS_DIR.glob("*.png")):
        image = cv.imread(str(image_path))
        if image is not None:
            overlays.append(cv.resize(image, (FRAME_WIDTH, HEADER_HEIGHT)))
    if len(overlays) < 4:
        raise RuntimeError("Expected at least four PNG toolbar images in assets/.")
    return overlays


def main():
    overlays = load_overlays()
    active_overlay = 0
    draw_color = (0, 0, 255)
    xp, yp = 0, 0
    img_canvas = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), np.uint8)

    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    detector = ht.handDetector(detectionCon=0.85)

    while True:
        success, img = cap.read()
        if not success:
            print("Camera not available.")
            break

        img = cv.flip(img, 1)
        img = detector.findHands(img)
        lm_list = detector.findPosition(img, draw=False)

        if lm_list:
            x1, y1 = lm_list[8][1:]
            x2, y2 = lm_list[12][1:]
            fingers = detector.fingersUp()

            if fingers == [1, 1, 0, 0, 0]:
                cv.imwrite(str(CANVAS_PATH), img_canvas)

            if fingers[1] and fingers[2]:
                xp, yp = 0, 0
                cv.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), draw_color, cv.FILLED)

                if y1 < HEADER_HEIGHT:
                    if 250 < x1 < 450:
                        active_overlay = 0
                        draw_color = (0, 0, 255)
                    elif 550 < x1 < 750:
                        active_overlay = 1
                        draw_color = (255, 0, 0)
                    elif 800 < x1 < 950:
                        active_overlay = 2
                        draw_color = (0, 255, 0)
                    elif 1050 < x1 < 1200:
                        active_overlay = 3
                        draw_color = (0, 0, 0)

            if fingers[1] and not fingers[2]:
                cv.circle(img, (x1, y1), 15, draw_color, cv.FILLED)
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1

                thickness = ERASER_THICKNESS if draw_color == (0, 0, 0) else BRUSH_THICKNESS
                cv.line(img, (xp, yp), (x1, y1), draw_color, thickness)
                cv.line(img_canvas, (xp, yp), (x1, y1), draw_color, thickness)
                xp, yp = x1, y1

        img_gray = cv.cvtColor(img_canvas, cv.COLOR_BGR2GRAY)
        _, img_inv = cv.threshold(img_gray, 50, 255, cv.THRESH_BINARY_INV)
        img_inv = cv.cvtColor(img_inv, cv.COLOR_GRAY2BGR)
        img = cv.bitwise_and(img, img_inv)
        img = cv.bitwise_or(img, img_canvas)
        img[0:HEADER_HEIGHT, 0:FRAME_WIDTH] = overlays[active_overlay]

        cv.imshow("MathCam", img)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    cv.imwrite(str(CANVAS_PATH), img_canvas)
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_MSMF", "0")
    main()
