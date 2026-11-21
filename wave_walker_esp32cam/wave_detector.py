""" recognizing an open palm and checking its back-and-forth motion"""

from dataclasses import dataclass
from math import atan2, pi, radians, sqrt

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision as hand_vision

PALM_JOINTS = (0, 5, 9, 13, 17)


@dataclass
class Hand:
    center: np.ndarray
    span: float
    angle: float
    open_enough: bool
    wrist: np.ndarray | None = None
    gesture: str = ""


def count_straight_fingers(landmarks):
    """counting straight fingers when the gesture model isn't sure."""

    points = np.array([(point.x, point.y, point.z) for point in landmarks])
    palm = points[9] - points[0]
    length = np.linalg.norm(palm)
    if length <= 1e-6 or not np.isfinite(points).all():
        return 0

    axis = palm / length
    count = 0

    for base in (5, 9, 13, 17):
        finger = points[base:base + 4]
        segments = np.diff(finger, axis=0)
        sizes = np.linalg.norm(segments, axis=1)
        aligned = np.sum(segments[:-1] * segments[1:], axis=1)
        if (
            np.all(sizes > 1e-6)
            and np.all(aligned > 0.65 * sizes[:-1] * sizes[1:])
            and np.dot(finger[-1] - finger[0], axis) > 0.27 * length
            and (
                np.linalg.norm(finger[-1] - points[0]) - np.linalg.norm(finger[0] - points[0])
                > 0.40 * length
            )
        ):
            count += 1

    return count


def measure_hand(result, width, height, zone):
    if len(result.hand_landmarks) != 1 or len(result.gestures) != 1 or not result.gestures[0]:
        return None

    category = max(result.gestures[0], key=lambda gesture: gesture.score)
    points = np.array([(point.x * width, point.y * height) for point in result.hand_landmarks[0]])
    center = points[list(PALM_JOINTS)].mean(axis=0)
    wrist = points[0]
    span = max(
        float(np.linalg.norm(points[5] - points[17])),
        0.8 * float(np.linalg.norm(wrist - points[9])),
    )
    if (
        span < 12
        or not np.isfinite(points).all()
        or not (0 <= center[0] < width and 0 <= center[1] < height)
    ):
        return None

    x, y, crop_width, crop_height = zone
    scale = np.array([crop_width / width, crop_height / height])
    offset = np.array([x, y])
    axis = (points[9] - wrist) * scale
    is_open = category.category_name == "Open_Palm" and category.score >= 0.65
    if not is_open and category.category_name in ("None", "Open_Palm") and result.hand_world_landmarks:
        is_open = count_straight_fingers(result.hand_world_landmarks[0]) >= 3

    return Hand(
        center * scale + offset,
        span * sqrt(scale.prod()),
        atan2(axis[0], -axis[1]),
        is_open,
        wrist * scale + offset,
        category.category_name,
    )


def load_hand_detector(model):
    options = hand_vision.GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=str(model)),
        running_mode=hand_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.55,
        min_tracking_confidence=0.55,
    )
    return hand_vision.GestureRecognizer.create_from_options(options)


def read_hand(recognizer, frame, timestamp, zoom):
    height, width = frame.shape[:2]
    crop_width, crop_height = round(width / zoom), round(height / zoom)
    x, y = (width - crop_width) // 2, (height - crop_height) // 2
    zone = (x, y, crop_width, crop_height)
    crop = frame[y:y + crop_height, x:x + crop_width]
    if zoom > 1:
        crop = cv2.resize(crop, (width, height), interpolation=cv2.INTER_LINEAR)

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    result = recognizer.recognize_for_video(
        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp,
    )

    return measure_hand(result, width, height, zone), len(result.hand_landmarks), zone


def show_hand(frame, hand, count, zone, row):
    from calibrate import label

    x, y, width, height = zone
    if x or y:
        cv2.rectangle(frame, (x, y), (x + width - 1, y + height - 1), (0, 230, 255), 2)

    text = "Show one hand" if count > 1 else "No clear hand in wave area"
    if hand is not None:
        text = "Open palm" if hand.open_enough else "Show an open palm"
        if hand.gesture == "Closed_Fist":
            text = "Closed fist"
        color = (0, 220, 0) if hand.open_enough else (0, 130, 255)
        cv2.circle(frame, tuple(hand.center.astype(int)), 5, color, -1)

    label(frame, text, row)


def angle_difference(current, previous):
    return (current - previous + pi) % (2 * pi) - pi


class NaturalWave:
    def __init__(self):
        self.reset()

    def reset(self):
        self.start = None
        self.last = None
        self.moving_since = None
        self.samples = 0

    def begin(self, hand, now):
        self.start = (hand, now)
        self.last = (hand, now)
        self.unwrapped_angle = hand.angle
        self.positions = [float(hand.center[0]), hand.angle]
        self.peaks = self.positions.copy()
        self.directions = [0, 0]
        self.samples = 1
        return "start"

    def update(self, hand, now):
        if hand is None:
            if self.last is not None and now - self.last[1] > 0.30:
                self.reset()
            return "idle"
        if not hand.open_enough:
            self.reset()
            return "idle"

        if self.start is None:
            return self.begin(hand, now)

        previous, last_seen = self.last
        starting_hand, started_at = self.start
        jump = np.linalg.norm(hand.center - previous.center)
        motion_expired = self.moving_since is not None and now - self.moving_since > 1.8
        if now - last_seen > 0.30 or motion_expired or jump > max(22.0, 2.2 * previous.span):
            self.reset()
            return self.begin(hand, now)

        #keeping the wrist angle continuous when it wraps past ==> -180 or 180 degrees
        self.unwrapped_angle += angle_difference(hand.angle, previous.angle)
        self.last = (hand, now)
        self.samples += 1

        # counting either a sideways sweep or a wrist tilt as wave motion
        positions = (float(hand.center[0]), self.unwrapped_angle)
        thresholds = (max(6.0, 0.23 * starting_hand.span), radians(12))
        for i, (position, threshold) in enumerate(zip(positions, thresholds)):
            filtered = 0.75 * position + 0.25 * self.positions[i]
            
            self.positions[i] = filtered
            displacement = filtered - self.peaks[i]
            direction = self.directions[i]

            if direction == 0:
                if abs(displacement) >= threshold:
                    self.directions[i] = 1 if displacement > 0 else -1
                    self.peaks[i] = filtered
                    if self.moving_since is None:
                        self.moving_since = now
                continue

            if direction * displacement > 0:
                self.peaks[i] = filtered
            elif (
                direction * displacement <= -threshold
                and self.samples >= 4
                and now - started_at >= 0.15
            ):
                self.reset()
                return "confirmed"

        if self.moving_since is None and now - started_at >= 1.2:
            self.reset()
            return self.begin(hand, now)
        return "idle"


def hand_moved(hand, reference):
    if hand is None or reference is None:
        return hand is not reference

    travel = np.linalg.norm(hand.center - reference.center)
    turn = abs(angle_difference(hand.angle, reference.angle))
    return travel >= max(6, 0.14 * reference.span) or turn >= radians(7)


