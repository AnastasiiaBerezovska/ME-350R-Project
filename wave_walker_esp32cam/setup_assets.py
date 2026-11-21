"""Made the printable marker and downloading any missing MediaPipe models."""

import hashlib
from pathlib import Path
import struct
import urllib.request
import zlib

ROOT = Path(__file__).resolve().parent
# I got the models running!!
MODELS = {
    "gesture_recognizer.task": (
        "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
        "gesture_recognizer/float16/1/gesture_recognizer.task",
        "97952348cf6a6a4915c2ea1496b4b37ebabc50cbbf80571435643c455f2b0482",
    ),
    "pose_landmarker_lite.task": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a",
    ),
}


def main():
    import cv2
    import numpy as np
    import json

    settings = json.loads((ROOT / "settings.json").read_text())
    (ROOT / "models").mkdir(exist_ok=True)

    for name, (url, expected) in MODELS.items():
        path = ROOT / "models" / name
        if not path.exists():
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
            if expected and hashlib.sha256(data).hexdigest() != expected:
                raise RuntimeError(f"Model checksum mismatch: {name}")
            path.write_bytes(data)

        if expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Model checksum mismatch: {name}")
        print(name, "ready")

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    marker = cv2.aruco.generateImageMarker(dictionary, settings["marker_id"], 960)
    page = np.full((1300, 1200), 255, dtype=np.uint8)
    
    page[160:1120, 120:1080] = marker

    cv2.arrowedLine(page, (600, 125), (600, 55), 0, 5, tipLength=0.3)
    cv2.putText(page, "ROBOT FRONT", (390, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, 0, 2)
    cv2.putText(
        page,
        f"Black square: {settings['marker_side_m'] * 100:.1f} cm each side",
        (180, 1210),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.05,
        0,
        2,
    )

    ok, encoded = cv2.imencode(".png", page)
    if not ok:
        raise RuntimeError("Could not encode the marker image")

    # Imm setting the print scale so the 960 pixel black square matches marker_side_m
    ppm = round(960 / settings["marker_side_m"])
    payload = b"pHYs" + struct.pack(">IIB", ppm, ppm, 1)
    chunk = struct.pack(">I", 9) + payload + struct.pack(">I", zlib.crc32(payload) & 0xFFFFFFFF)
    png = encoded.tobytes()
    (ROOT / "robot_marker.png").write_bytes(png[:33] + chunk + png[33:])
    print("robot_marker.png ready; print it and measure the black square before using it")


if __name__ == "__main__":
    main()
