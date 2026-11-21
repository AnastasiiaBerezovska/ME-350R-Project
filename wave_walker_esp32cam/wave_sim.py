"""I'm using a camera wave to move a simulated robot, with no motor connection."""

import argparse
from contextlib import ExitStack
from math import cos, sin
from pathlib import Path
from time import monotonic

from control import Controller, Pose


def advance(pose, left, right, dt):
    # using a rough motion model that hasn't been checked against the wooden walker
    def speed(pwm):
        return max(0, pwm - 50) * 0.004

    l, r = speed(left), speed(right)
    omega, velocity = (r - l) / 0.26, (l + r) / 2
    mid = pose.heading + omega * dt / 2
    return Pose(
        pose.x + cos(mid) * velocity * dt,
        pose.y + sin(mid) * velocity * dt,
        pose.heading + omega * dt,
    )


def main():
    import cv2
    import numpy as np

    from calibrate import label
    from tracking import Camera, ESP32Camera
    from walker import ROOT, WaveGate, load_settings
    from wave_detector import load_hand_detector, read_hand, show_hand

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("floor", "table"), default="table")
    parser.add_argument("--wave-camera-ip", help="ESP32-CAM IPv4 address from CameraWebServer")
    parser.add_argument(
        "--wave-zoom",
        type=float,
        choices=(1.0, 1.5, 2.0, 2.5, 3.0),
        default=1.0,
        help="Magnify the centre of the wave camera image",
    )
    args = parser.parse_args()

    settings = load_settings(ROOT / "settings.json")
    w, d = (2.0, 1.2) if args.mode == "table" else (4.0, 3.0)
    initial = Pose(0.5, d / 2, 0.0)
   
    destination = (1.5, d / 2) if args.mode == "table" else (3.2, d / 2)
    pose = initial
   
    controller, gate = Controller(settings, w, d, args.mode), WaveGate()
    previous_time, number, timestamp = monotonic(), 0, -1

    with ExitStack() as stack:
        camera = ESP32Camera(args.wave_camera_ip) if args.wave_camera_ip else Camera(settings)
        stack.callback(camera.close)
      
        stack.callback(cv2.destroyAllWindows)
        detector = stack.enter_context(load_hand_detector(ROOT / "models/gesture_recognizer.task"))
      
        camera_name = "ESP32-CAM" if args.wave_camera_ip else "Webcam"
        print(f"SIMULATION ONLY: wave at the {camera_name}. R resets. SPACE stops. Q quits.", flush=True)

        while True:
            key = cv2.waitKey(1) & 255
            if key in (ord("q"), 27):
                return
            if key == ord("r"):
                pose, controller = initial, Controller(settings, w, d, args.mode)
                gate = WaveGate()
            if key == ord(" "):
                controller.stop("Stopped")
                gate.waiting, gate.quiet_since = True, None

            number, stamp, frame = camera.next(number, timeout=3)
            timestamp = max(timestamp + 1, int(stamp * 1000))
            hand, hand_count, zone = read_hand(detector, frame, timestamp, args.wave_zoom)
            if hand_count > 1:
                gate.detector.reset()

            now = monotonic()
            dt, previous_time = min(0.2, now - previous_time), now
            started = (
                gate.update(hand, stamp)
                and not controller.active
                and controller.start(pose, destination, now)
            )
            if controller.active and not started:
                left, right = controller.update(pose, destination, now)
                pose = advance(pose, left, right, dt)

            map_image = np.full((600, 850, 3), 245, dtype=np.uint8)
            scale = min(730 / w, 410 / d)

            def pixel(point):
                return int(60 + point[0] * scale), int(520 - point[1] * scale)

            cv2.rectangle(map_image, pixel((0, d)), pixel((w, 0)), (50, 50, 50), 3)
            margin = settings["robot_radius_m"]
            cv2.rectangle(
                map_image,
                pixel((margin, d - margin)),
                pixel((w - margin, margin)),
                (160, 180, 160),
                1,
            )
            cv2.circle(map_image, pixel(destination), 11, (40, 120, 240), -1)
            cv2.circle(map_image, pixel(pose.xy), int(0.17 * scale), (220, 100, 30), -1)
            front = pose.x + 0.27 * cos(pose.heading), pose.y + 0.27 * sin(pose.heading)
            cv2.arrowedLine(map_image, pixel(pose.xy), pixel(front), (0, 0, 0), 3)

            label(map_image, "SIMULATION ONLY - no motor connection")
            label(map_image, controller.reason, 1)
            label(map_image, "Idealised motion; physical walking and stopping are untested", 2)

            wave_view = frame.copy()
            label(wave_view, "R resets | Space stops | Q quits")
            show_hand(wave_view, hand, hand_count, zone, 1)
            cv2.imshow(f"{camera_name} wave", wave_view)
            cv2.imshow("Virtual waiter", map_image)


if __name__ == "__main__":
    main()
