"""Flask application for MathCam.

MathCam lets a user draw equations in the air with their index finger, capture the
canvas, and send the drawing to Gemini for a plain-text solution.
"""

from __future__ import annotations

import os
from pathlib import Path
from threading import Lock

import cv2 as cv
import google.generativeai as genai
import numpy as np
from dotenv import load_dotenv
from flask import Flask, Response, jsonify, render_template

import handTrack as ht

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
ASSETS_DIR = BASE_DIR / "assets"
CANVAS_PATH = BASE_DIR / "saved_canvas.jpg"
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
HEADER_HEIGHT = 125
BRUSH_THICKNESS = 15
ERASER_THICKNESS = 100
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-pro")

app = Flask(__name__)

_camera = None
_detector = None
_state_lock = Lock()
_xp, _yp = 0, 0
_canvas = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), np.uint8)
_draw_color = (0, 0, 255)
_overlays: list[np.ndarray] = []
_active_overlay_index = 0


def _load_overlays() -> list[np.ndarray]:
    """Load toolbar images from the assets directory in a deterministic order."""
    if not ASSETS_DIR.exists():
        raise FileNotFoundError(f"Missing assets directory: {ASSETS_DIR}")

    overlays = []
    for image_path in sorted(ASSETS_DIR.glob("*.png")):
        image = cv.imread(str(image_path))
        if image is not None:
            overlays.append(cv.resize(image, (FRAME_WIDTH, HEADER_HEIGHT)))

    if len(overlays) < 4:
        raise RuntimeError("Expected at least four PNG toolbar images in assets/.")

    return overlays


def _get_overlays() -> list[np.ndarray]:
    global _overlays
    if not _overlays:
        _overlays = _load_overlays()
    return _overlays


def _get_camera():
    global _camera
    if _camera is None:
        _camera = cv.VideoCapture(0)
        _camera.set(cv.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        _camera.set(cv.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    return _camera


def _get_detector():
    global _detector
    if _detector is None:
        _detector = ht.handDetector(detectionCon=0.85)
    return _detector


def _blank_frame(message: str) -> np.ndarray:
    frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), np.uint8)
    frame[:] = (40, 42, 54)
    cv.putText(
        frame,
        message,
        (80, FRAME_HEIGHT // 2),
        cv.FONT_HERSHEY_SIMPLEX,
        1.2,
        (248, 248, 242),
        3,
        cv.LINE_AA,
    )
    return frame


def _save_canvas() -> None:
    with _state_lock:
        cv.imwrite(str(CANVAS_PATH), _canvas)


def _clear_canvas() -> None:
    global _canvas, _xp, _yp
    with _state_lock:
        _canvas = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), np.uint8)
        _xp, _yp = 0, 0
        cv.imwrite(str(CANVAS_PATH), _canvas)


def _select_tool(x: int) -> None:
    """Select the active drawing tool based on a header x-coordinate."""
    global _active_overlay_index, _draw_color

    if 250 < x < 450:
        _active_overlay_index = 0
        _draw_color = (0, 0, 255)
    elif 550 < x < 750:
        _active_overlay_index = 1
        _draw_color = (255, 0, 0)
    elif 800 < x < 950:
        _active_overlay_index = 2
        _draw_color = (0, 255, 0)
    elif 1050 < x < 1200:
        _active_overlay_index = 3
        _draw_color = (0, 0, 0)


def _process_frame(frame: np.ndarray) -> np.ndarray:
    """Apply hand tracking, drawing gestures, and canvas compositing."""
    global _xp, _yp, _canvas

    detector = _get_detector()
    overlays = _get_overlays()

    frame = cv.flip(frame, 1)
    frame = detector.findHands(frame)
    lm_list = detector.findPosition(frame, draw=False)

    with _state_lock:
        if lm_list:
            x1, y1 = lm_list[8][1:]
            x2, y2 = lm_list[12][1:]
            fingers = detector.fingersUp()

            # Thumb + index saves the current canvas without requiring a mouse.
            if fingers == [1, 1, 0, 0, 0]:
                cv.imwrite(str(CANVAS_PATH), _canvas)

            # Index + middle finger: selection mode.
            if fingers[1] and fingers[2]:
                _xp, _yp = 0, 0
                cv.rectangle(frame, (x1, y1 - 25), (x2, y2 + 25), _draw_color, cv.FILLED)
                if y1 < HEADER_HEIGHT:
                    _select_tool(x1)

            # Index finger only: drawing mode.
            if fingers[1] and not fingers[2]:
                cv.circle(frame, (x1, y1), 15, _draw_color, cv.FILLED)
                if _xp == 0 and _yp == 0:
                    _xp, _yp = x1, y1

                thickness = ERASER_THICKNESS if _draw_color == (0, 0, 0) else BRUSH_THICKNESS
                cv.line(frame, (_xp, _yp), (x1, y1), _draw_color, thickness)
                cv.line(_canvas, (_xp, _yp), (x1, y1), _draw_color, thickness)
                _xp, _yp = x1, y1

        img_gray = cv.cvtColor(_canvas, cv.COLOR_BGR2GRAY)
        _, img_inv = cv.threshold(img_gray, 50, 255, cv.THRESH_BINARY_INV)
        img_inv = cv.cvtColor(img_inv, cv.COLOR_GRAY2BGR)
        frame = cv.bitwise_and(frame, img_inv)
        frame = cv.bitwise_or(frame, _canvas)
        frame[0:HEADER_HEIGHT, 0:FRAME_WIDTH] = overlays[_active_overlay_index]

    return frame


def gen_frames():
    """Yield MJPEG frames for the browser video stream."""
    camera = _get_camera()

    while True:
        success, frame = camera.read()
        if success:
            output = _process_frame(frame)
        else:
            output = _blank_frame("Camera not available. Check webcam permissions and refresh.")

        ok, buffer = cv.imencode(".jpg", output)
        if not ok:
            continue

        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/save", methods=["POST"])
def save_canvas():
    _save_canvas()
    return jsonify({"ok": True, "message": "Canvas saved."})


@app.route("/clear", methods=["POST"])
def clear_canvas():
    _clear_canvas()
    return jsonify({"ok": True, "message": "Canvas cleared."})


@app.route("/solve", methods=["POST", "GET"])
def solve():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        return (
            jsonify(
                {
                    "ok": False,
                    "message": "Add GOOGLE_API_KEY to your .env file before solving with Gemini.",
                }
            ),
            400,
        )

    if not CANVAS_PATH.exists():
        _save_canvas()

    genai.configure(api_key=api_key)
    sample_file = genai.upload_file(path=str(CANVAS_PATH), display_name="MathCam canvas")
    model = genai.GenerativeModel(model_name=GEMINI_MODEL)
    response = model.generate_content(
        [
            sample_file,
            (
                "The image contains a handwritten math question or equation. "
                "Solve it. Start with the final answer, then give a concise explanation. "
                "Return plain text only."
            ),
        ]
    )
    return jsonify({"ok": True, "message": response.text})


@app.route("/gemini")
def gemini():
    """Backward-compatible text endpoint used by older versions of the UI."""
    response = solve()
    if isinstance(response, tuple):
        body, status = response
        return body.get_json().get("message", "Gemini request failed."), status
    return response.get_json().get("message", "")


if __name__ == "__main__":
    app.run(host=os.getenv("FLASK_RUN_HOST", "127.0.0.1"), port=int(os.getenv("PORT", "5000")), debug=os.getenv("FLASK_DEBUG", "0") == "1")
