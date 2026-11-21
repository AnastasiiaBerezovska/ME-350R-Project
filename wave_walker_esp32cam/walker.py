#!/usr/bin/env python3
"""Software contribution - Anastasiia Brund, ME-350R-Project
   Contributed by calibrating the camera, detecting an open-hand wave, 
   tracking the robot's marker, and sending motor commands over USB to the Arduino Uno/Raspberry Pi Pico. 

   In addition, built the simulator and tested how the controller stops the 
   walker when tracking is lost or it crosses the allowed boundary.
"""

"""I am using a wave to call the walker to a table spot or a person on the floor."""

import argparse
from contextlib import ExitStack
import json
from math import isfinite
from pathlib import Path
import sys
from time import monotonic, sleep

from control import Controller

ROOT = Path(__file__).resolve().parent


def load_settings(path):
    settings = json.loads(Path(path).read_text())
    numeric = (
        "marker_side_m",
        "robot_radius_m",
        "stop_distance_m",
        "table_goal_tolerance_m",
        "max_travel_m",
        "max_trip_seconds",
        "no_progress_seconds",
        "max_frame_age_seconds",
        "person_motion_limit_m",
    )
    if any(
        not isinstance(settings[k], (int, float)) or not isfinite(settings[k]) or settings[k] <= 0
        for k in numeric
    ):
        raise ValueError("Settings must contain positive finite distances and time limits")
    if not 0 < settings["max_frame_age_seconds"] <= 0.25:
        raise ValueError("Frame age limit must not exceed 0.25 seconds")
    if not 0 < settings["minimum_pwm"] <= settings["maximum_pwm"] <= 120:
        raise ValueError("Require 0 < minimum_pwm <= maximum_pwm <= 120")
    if not 0 < settings["turn_pwm"] <= 120:
        raise ValueError("turn_pwm must be in 1..120")

    for key in ("minimum_pwm", "maximum_pwm", "turn_pwm", "camera_index", "marker_id"):
        if type(settings[key]) is not int:
            raise ValueError(f"{key} must be an integer")

    if not 0 <= settings["marker_id"] < 50:
        raise ValueError("marker_id must be in 0..49")
    if any(settings[k] not in (-1, 1) for k in ("left_motor_sign", "right_motor_sign")):
        raise ValueError("Motor signs must be +1 or -1")
    if settings["robot_radius_m"] < 0.28:
        raise ValueError(
            "The CAD needs at least 0.28 m swept clearance; measure before changing this bound"
        )
    if not isfinite(settings["marker_heading_offset_deg"]):
        raise ValueError("Invalid marker heading offset")

    return settings

# understanding if the motors are connected.
def check_motors(args, settings):
    from usb_link import USBLink

    link = USBLink(args.port)
    try:
        print(
            f"UNO connected. Motor enable {'closed' if link.enabled else 'OPEN'}. Limit {link.max_pwm} PWM."
        )
        if args.command == "check":
            print("USB protocol is ready. Motors were not commanded.")
            return
        if not args.drive:
            raise ValueError(
                "A motor pulse requires --drive; raise the robot so its feet cannot touch the surface"
            )
        if not 1 <= args.ms <= 200 or not 0 < args.pwm <= link.max_pwm:
            raise ValueError("Pulse must be 1..200 milliseconds and PWM 1..120")

        left = args.pwm * settings["left_motor_sign"] if args.side == "left" else 0
        
        right = args.pwm * settings["right_motor_sign"] if args.side == "right" else 0
        link.arm()
        started = monotonic()

        while monotonic() - started < args.ms / 1000:
            link.drive(left, right)
            sleep(min(0.04, max(0.0, args.ms / 1000 - (monotonic() - started))))

        link.halt()
        print(f"Completed {args.side} pulse; motor commands stopped.")
    finally:
        link.close()


class WaveGate:
    def __init__(self):
        from wave_detector import NaturalWave

        self.detector = NaturalWave()
        self.waiting = False
        self.quiet_since = None
        self.quiet_hand = None

    def update(self, hand, now):
        from wave_detector import hand_moved

        if self.waiting:
            if self.quiet_since is None or hand_moved(hand, self.quiet_hand):
                self.quiet_since, self.quiet_hand = now, hand
            elif now - self.quiet_since >= 0.65:
                self.waiting = False
                self.detector.reset()
            return False

        if self.detector.update(hand, now) == "confirmed":
            self.waiting, self.quiet_since = True, None
            return True

        return False


def run(args, settings):
   # additional
   
    import cv2
    import mediapipe as mp
    import numpy as np
    from mediapipe.tasks.python import BaseOptions, vision

    from calibrate import label, load, check_clearance
    from tracking import (
        Camera,
        ESP32Camera,
        MarkerTracker,
        StationaryPerson,
        feet_position,
        matches_wrist,
    )
    from wave_detector import load_hand_detector, read_hand, show_hand

    if args.wave_camera_ip and args.mode != "table":
        raise ValueError(
            "Use the ESP32-CAM wave input with table mode. Floor mode tracks feet and hand in one view."
        )


    config = load(args.calibration)
    if config.get("version") != 1 or config.get("mode") != args.mode:
        raise ValueError("Calibration does not match the selected floor/table mode")
    for name in ("surface", "robot"):
        if name not in config:
            raise ValueError(f"Missing {name} calibration; run the two calibration steps first")


    if args.mode == "table":
        if "serving_point_m" not in config:
            raise ValueError("Choose the table serving point with calibrate --plane serve")
        if args.drive and "edge_clearance_m" not in config:
            raise ValueError(
                "Measure the tabletop clearance and save it with "
                "calibrate --plane serve --table-clearance-m"
            )
        margin = check_clearance(
            config.get("edge_clearance_m", settings["robot_radius_m"]),
            config["width_m"],
            config["depth_m"],
        )
        settings = {**settings, "robot_radius_m": margin}

    if settings["camera_index"] != config["camera_index"]:
        raise ValueError("I'm stopping because the camera changed; calibrating again is required")

    model = ROOT / "models" / "gesture_recognizer.task"
    pose_model = ROOT / "models" / "pose_landmarker_lite.task"
    if not model.exists() or (args.mode == "floor" and not pose_model.exists()):
        raise ValueError("Models missing; run python setup_assets.py")

    controller = Controller(settings, config["width_m"], config["depth_m"], args.mode)
    tracker = MarkerTracker(settings, config)
    gate, stable = WaveGate(), StationaryPerson()
    
    window = f"Wave walker - {args.mode} - {'MOTORS ENABLED' if args.drive else 'PREVIEW ONLY'}"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    count, number, wave_number, timestamp_ms = 0, 0, 0, -1
    
    status = "Wave at the ESP32-CAM" if args.wave_camera_ip else "Show one hand and wave"
    previous_message, previous_command_time = None, 0.0

    # I'm closing cameras and have to fix the motors even if image processing fails
    with ExitStack() as stack:
        link = None
        if args.drive:
            if not args.port:
                raise ValueError("Motor mode requires --port /dev/ttyACM0")
            from usb_link import USBLink

            link = USBLink(args.port)
            stack.callback(link.close)

        camera = Camera(settings)
        stack.callback(camera.close)
        
        stack.callback(cv2.destroyAllWindows)
        wave_camera = None
        if args.wave_camera_ip:
            wave_camera = ESP32Camera(args.wave_camera_ip)
            stack.callback(wave_camera.close)

        hands = stack.enter_context(load_hand_detector(model))
        poses = None
        if args.mode == "floor":
            options = vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(pose_model)),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=2,
                min_pose_detection_confidence=0.65,
                min_pose_presence_confidence=0.65,
                min_tracking_confidence=0.60,
            )
            poses = stack.enter_context(vision.PoseLandmarker.create_from_options(options))

        print(
            "Preview has no motor connection."
            if link is None
            else "Motor mode selected. SPACE stops; Q quits. Keeping the physical power switch reachable.",
            flush=True,
        )

        while True:
            key = cv2.waitKey(1) & 255
            if key in (ord("q"), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key == ord(" "):
                controller.stop("Stopped from keyboard")
                if link:
                    link.halt()
                gate.waiting, gate.quiet_since = True, None

            number, stamp, frame = camera.next(number)
            if list(frame.shape[1::-1]) != config["image_size"]:
                raise RuntimeError("I'm stopping because the camera size changed; calibrating again is required")

            wave_sample = (
                wave_camera.latest_after(wave_number) if wave_camera else (number, stamp, frame)
            )
            hand = None
            if wave_sample is not None:
                wave_number, wave_stamp, wave_frame = wave_sample
                timestamp_ms = max(timestamp_ms + 1, int(wave_stamp * 1000))
                hand, hand_count, zone = read_hand(hands, wave_frame, timestamp_ms, args.wave_zoom)
                if hand_count > 1:
                    gate.detector.reset()

            height, width = frame.shape[:2]
            robot, marker_pixels = tracker.detect(frame)
            person = None
            valid_wave = hand
            if args.mode == "floor":
                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                )
                pose_result = poses.detect_for_video(image, timestamp_ms)
                if len(pose_result.pose_landmarks) == 1:
                   
                    landmarks = pose_result.pose_landmarks[0]
                    person = feet_position(landmarks, width, height, config["surface"]["matrix"])
                    if hand is None or not matches_wrist(hand.wrist, landmarks, width, height):
                        valid_wave = None
                else:
                    valid_wave = None
                person_ready = stable.update(person, stamp)
            else:
                person = tuple(config["serving_point_m"])
                person_ready = True
            now = monotonic()
            if now - stamp > settings["max_frame_age_seconds"]:
                raise RuntimeError("Inference is too slow for motor control; trip stopped")

            was_active = controller.active

            # tracking waves during a trip and waiting until it ends to accept a new call, 
            waved = False
            if wave_sample is not None:
                if wave_camera and now - wave_stamp > 0.5:
                    gate.detector.reset()
                else:
                    waved = gate.update(valid_wave, wave_stamp)

            started = False
            if waved and not was_active:
                count += 1
                if robot is None:
                    status = "Wave seen; robot marker is not visible"
                elif person is None or not person_ready:
                    status = "Wave seen; need one stationary person with visible feet"
                elif controller.start(robot, person, now):
                    if link:
                        link.arm()
                    started = True
                    status = controller.reason
                    print(f"Wave {count}: {status}", flush=True)
                else:
                    status = controller.reason

            if controller.active:
                command = (0, 0) if started else controller.update(robot, person, now)
                status = controller.reason
                if link:
                    if not controller.active:
                        link.halt()
                    else:
                        left = int(command[0]) * settings["left_motor_sign"]
                        right = int(command[1]) * settings["right_motor_sign"]
                        link.drive(left, right)
                        previous_command_time = now
                else:
                    if now - previous_command_time > 0.25:
                        print(
                            f"PREVIEW motor commands left={command[0]} right={command[1]} | {status}",
                            flush=True,
                        )
                        previous_command_time = now

            if was_active and not controller.active:
                gate.waiting, gate.quiet_since = True, None
                print(status, flush=True)
            if status != previous_message:
                print(status, flush=True)
                previous_message = status

            shown = frame.copy()
            if marker_pixels is not None:
                pts = marker_pixels.astype(np.int32)
                cv2.polylines(shown, [pts], True, (0, 220, 0), 2)
                cv2.line(
                    shown,
                    tuple(pts.mean(axis=0).astype(int)),
                    tuple(pts[:2].mean(axis=0).astype(int)),
                    (0, 0, 255),
                    3,
                )

            label(shown, status)
            label(shown, "SPACE: stop | Q: quit | " + ("MOTORS ENABLED" if link else "PREVIEW ONLY"), 1)
            label(shown, "Robot marker: " + ("visible" if robot else "not visible"), 2)
            if args.mode == "floor":
                label(shown, "Feet: " + ("stationary" if person_ready else "missing or moving"), 3)
            if wave_camera is None:
                show_hand(shown, hand, hand_count, zone, 4 if args.mode == "floor" else 3)
            cv2.imshow(window, shown)

            if wave_camera and wave_sample is not None:
                wave_view = wave_frame.copy()
                label(wave_view, "Wave at this camera")
                show_hand(wave_view, hand, hand_count, zone, 1)
                cv2.imshow("ESP32-CAM wave", wave_view)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--settings", default=str(ROOT / "settings.json"))
    commands = p.add_subparsers(dest="command", required=True)

    for name in ("check", "bench"):
        q = commands.add_parser(name)
        q.add_argument("--port", required=True)
        if name == "bench":
            q.add_argument("--side", choices=("left", "right"), required=True)
            q.add_argument("--pwm", type=int, default=65)
            q.add_argument("--ms", type=int, default=150)
            q.add_argument("--drive", action="store_true")

    q = commands.add_parser("calibrate")
    q.add_argument("--mode", choices=("floor", "table"), required=True)
    q.add_argument("--plane", choices=("surface", "robot", "serve"), required=True)
    q.add_argument("--calibration", required=True)
    q.add_argument("--width-m", type=float)
    q.add_argument("--depth-m", type=float)
    q.add_argument("--marker-height-m", type=float)
    q.add_argument(
        "--table-clearance-m",
        type=float,
        help="Measured total edge clearance in metres, used with --mode table --plane serve",
    )

    q = commands.add_parser("run")
    q.add_argument("--mode", choices=("floor", "table"), required=True)
    q.add_argument("--calibration", required=True)
    q.add_argument("--port")
    q.add_argument("--drive", action="store_true")
    q.add_argument("--wave-camera-ip", help="ESP32-CAM IPv4 address for table-mode wave detection")
    q.add_argument(
        "--wave-zoom",
        type=float,
        choices=(1.0, 1.5, 2.0, 2.5, 3.0),
        default=1.0,
        help="Magnify the centre of the wave camera image",
    )

    return p

# main finish up.
def main():
    args = build_parser().parse_args()

    try:
        settings = load_settings(args.settings)
        if args.command in ("check", "bench"):
            check_motors(args, settings)
        elif args.command == "calibrate":
            if args.plane == "serve" and args.mode != "table":
                raise ValueError("Serving points are for table mode")
            if args.table_clearance_m is not None and args.plane != "serve":
                raise ValueError("Set table clearance while calibrating the serving point")
            from calibrate import run as calibrate_run

            calibrate_run(args, settings)
        else:
            run(args, settings)
    except KeyboardInterrupt:
        print("Stopped")
    except Exception as exc:
        print(f"Stopped: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
