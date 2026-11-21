"""I'm tracking the robot and standing person using maps for their different heights."""

from collections import deque
from http.client import HTTPConnection
from ipaddress import IPv4Address
from math import atan2, radians
from threading import Event, Lock, Thread
from time import monotonic, sleep

import cv2
import numpy as np

from control import Pose, distance


class Camera:
    def __init__(self, settings):
        self.settings = settings
        self.lock, self.stop = Lock(), Event()
        self.latest, self.error, self.number = None, None, 0
        self.thread = Thread(target=self._read, daemon=True)
        self.thread.start()

    def _read(self):
        try:
            for stamp, frame in self._frames():
                with self.lock:
                    self.number += 1
                    self.latest = (self.number, stamp, frame)
        except Exception as exc:
            self.error = str(exc)

    def _frames(self):
        capture = cv2.VideoCapture(self.settings["camera_index"])
        try:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings["camera_width"])
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings["camera_height"])
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not capture.isOpened():
                raise RuntimeError("Cannot open webcam. Close webcam_test.py and other camera apps.")

            while not self.stop.is_set():
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError("Webcam stopped supplying frames")
                yield monotonic(), frame
        finally:
            capture.release()

    def latest_after(self, previous):
        if self.error:
            raise RuntimeError(self.error)
        with self.lock:
            if self.latest is not None and self.latest[0] > previous:
                return self.latest
        return None

    def next(self, previous, timeout=0.25):
        deadline = monotonic() + timeout
        while monotonic() < deadline:
            sample = self.latest_after(previous)
            if sample is not None:
                return sample
            sleep(0.002)
        raise RuntimeError("No fresh camera frame")

    def close(self):
        self.stop.set()
        self.thread.join(timeout=0.6)


class ESP32Camera(Camera):
    """reading new JPEGs from CameraWebServers /capture endpoint."""

    def __init__(self, ip):
        self.ip = str(IPv4Address(ip))
        super().__init__(None)

    def _frames(self):
        max_size = 2 * 1024 * 1024
        while not self.stop.is_set():
            connection = HTTPConnection(self.ip, timeout=1.0)
            stamp = monotonic()


            try:
                connection.request("GET", "/capture", headers={"Cache-Control": "no-cache"})
                with connection.getresponse() as response:
                    if response.status != 200:
                        raise ValueError(f"HTTP {response.status} from /capture")
                    jpeg = response.read(max_size + 1)
                if not 0 < len(jpeg) <= max_size:
                    raise ValueError("Unexpected JPEG size")
                frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError("Camera did not return a readable JPEG")
            except (OSError, ValueError) as exc:
                raise RuntimeError(
                    f"ESP32-CAM at {self.ip}: {exc}. Check the camera's Wi-Fi and /capture."
                ) from exc
            finally:
                connection.close()


            yield stamp, frame
            self.stop.wait(0.04)


def map_points(points, matrix):
    a = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    h = np.asarray(matrix, dtype=np.float64)
    if h.shape != (3, 3) or not np.isfinite(a).all() or not np.isfinite(h).all():
        raise ValueError("Invalid plane calibration")

    q = np.c_[a, np.ones(len(a))] @ h.T
    if np.any(np.abs(q[:, 2]) < 1e-9):
        raise ValueError("Projection approaches the horizon")

    q = q[:, :2] / q[:, 2, None]
    if not np.isfinite(q).all():
        raise ValueError("Invalid projected coordinates")

    return q


def fit_plane(pixels, world, max_error):
    pixels, world = np.asarray(pixels, dtype=float), np.asarray(world, dtype=float)
    if pixels.shape != (6, 2) or world.shape != (6, 2):
        raise ValueError("Use four corners plus two independent check points")

    contour = pixels[:4].astype(np.float32)
    if not cv2.isContourConvex(contour) or abs(cv2.contourArea(contour)) < 1000:
        raise ValueError("Corners are crossed, nearly collinear, or too small")

    matrix = cv2.getPerspectiveTransform(contour, world[:4].astype(np.float32))
    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError("Singular plane calibration")

    errors = np.linalg.norm(map_points(pixels[4:], matrix) - world[4:], axis=1)
    if errors.max() > max_error:
        raise ValueError(
            f"Independent check error {errors.max():.3f} m exceeds {max_error:.3f} m; "
            "I need to finish calibrating again with both check points within the limit"
        )

    return {
        "matrix": matrix.tolist(),
        "pixels": pixels.tolist(),
        "validation_errors_m": errors.tolist(),
    }


class MarkerTracker:
    def __init__(self, settings, calibration):
        self.s, self.c = settings, calibration
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    def detect(self, frame):
        corners, ids, _ = self.detector.detectMarkers(frame)
        if ids is None:
            return None, None

        matches = np.flatnonzero(ids.ravel() == self.s["marker_id"])
        if len(matches) != 1:
            return None, None

        pixels = corners[matches[0]].reshape(4, 2)
        h, w = frame.shape[:2]
        lengths = np.linalg.norm(np.roll(pixels, -1, axis=0) - pixels, axis=1)
        if (
            lengths.min() < 24
            or np.any(pixels[:, 0] < 4)
            or np.any(pixels[:, 0] > w - 4)
            or np.any(pixels[:, 1] < 4)
            or np.any(pixels[:, 1] > h - 4)
        ):
            return None, None

        world = map_points(pixels, self.c["robot"]["matrix"])
        sides = np.linalg.norm(np.roll(world, -1, axis=0) - world, axis=1)
        if np.any(abs(sides / self.s["marker_side_m"] - 1) > 0.30):
            return None, None

        centre = world.mean(axis=0)
        forward = world[:2].mean(axis=0) - centre
        heading = atan2(forward[1], forward[0]) + radians(self.s["marker_heading_offset_deg"])
        return Pose(float(centre[0]), float(centre[1]), heading), pixels


def feet_position(landmarks, width, height, surface_matrix):
    # checkig both heels and toe tips are visible on a standing person
    indices = (29, 30, 31, 32)
    for i in indices:
        p = landmarks[i]
        if (
            p.visibility is None
            or p.visibility < 0.75
            or not 0.01 < p.x < 0.99
            or not 0.01 < p.y < 0.99
        ):
            return None

    pixels = [(landmarks[i].x * width, landmarks[i].y * height) for i in indices]
    world = map_points(pixels, surface_matrix)
    if distance(world[0], world[1]) > 0.9:
        return None

    return tuple(world.mean(axis=0))


def matches_wrist(wrist, pose_landmarks, width, height):
    shoulder = np.mean(
        [[pose_landmarks[i].x * width, pose_landmarks[i].y * height] for i in (11, 12)],
        axis=0,
    )
    hip = np.mean(
        [[pose_landmarks[i].x * width, pose_landmarks[i].y * height] for i in (23, 24)],
        axis=0,
    )
    threshold = max(18, min(70, 0.30 * float(np.linalg.norm(shoulder - hip))))

    for i in (15, 16):
        p = pose_landmarks[i]
        if p.visibility is not None and p.visibility >= 0.65:
            if np.linalg.norm(wrist - [p.x * width, p.y * height]) <= threshold:
                return True

    return False


class StationaryPerson:
    def __init__(self):
        self.history = deque()

    def update(self, point, now):
        if point is None:
            self.history.clear()
            return False

        self.history.append((now, point))
        while self.history and now - self.history[0][0] > 0.7:
            self.history.popleft()

        return (
            len(self.history) >= 4
            and now - self.history[0][0] >= 0.45
            and all(distance(point, old) < 0.05 for _, old in self.history)
        )
