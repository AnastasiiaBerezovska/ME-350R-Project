"""I'm calibrating the floor or table, then repeating at the robot marker's height."""

import json
from math import isfinite
from pathlib import Path

import cv2
import numpy as np

from tracking import Camera, fit_plane, map_points


def label(frame, text, row=0):
    y = 25 + row * 25
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4)
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1)


def save(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
  
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def load(path):
    return json.loads(Path(path).read_text())


def check_clearance(clearance, width, depth):
    if (
        type(clearance) not in (int, float)
        or not isfinite(clearance)
        or not 0 < clearance < min(width, depth) / 2
    ):
        raise ValueError("Table clearance must be positive and leave room inside all four table edges")

    return clearance


def run(args, settings):
    path = Path(args.calibration)
  
    data = load(path) if path.exists() else {"version": 1, "mode": args.mode}
    if data.get("mode") != args.mode:
        raise ValueError("I need separate calibration files for floor and table")

    if args.plane == "surface":
        if args.width_m is None or args.depth_m is None:
            raise ValueError("I need measured --width-m and --depth-m for the usable rectangle")
        w, d = args.width_m, args.depth_m
        if min(w, d) <= 2 * settings["robot_radius_m"]:
            raise ValueError("Area is too narrow for the robot's configured turning clearance")
    else:
        if "surface" not in data:
            raise ValueError("I need the surface calibration before calibrating at marker height")
        w, d = data["width_m"], data["depth_m"]
        if data["camera_index"] != settings["camera_index"]:
            raise ValueError("I'm stopping because the camera changed; both planes need calibrating again")
        if args.plane == "robot" and (args.marker_height_m is None or args.marker_height_m <= 0):
            raise ValueError("I need measured --marker-height-m above the floor/table")

    camera = Camera(settings)
    window = "Calibration"
   
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    clicks, frozen = [], None
   
    number = 0

    def click(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN and frozen is not None:
            clicks.append((x, y))

    cv2.setMouseCallback(window, click)
    points = [(0, 0), (w, 0), (w, d), (0, d), (w / 4, d / 2), (3 * w / 4, d / 2)]

    try:
        # I'm taking one point at a time so the robot or target can move between clicks
        target_count = 1 if args.plane == "serve" else 6
        while len(clicks) < target_count:
            number, stamp, live = camera.next(number, timeout=3.0)
            if args.plane != "surface" and list(live.shape[1::-1]) != data["image_size"]:
                raise ValueError(
                    "I'm stopping because the camera resolution changed; both planes need calibrating again"
                )

            shown = (live if frozen is None else frozen).copy()
            if args.plane == "serve":
                text = "I'm choosing where the robot CENTRE should stop, away from the table edge"
            else:
                x, y = points[len(clicks)]
                text = f"{args.plane}: point {len(clicks) + 1}/6 at X={x:.3f}, Y={y:.3f} metres"

            label(shown, text)
            label(shown, "F: freeze current view. Click reference. R: unfreeze. Q: cancel.", 1)
            for xy in clicks:
                cv2.circle(shown, xy, 4, (0, 240, 255), -1)
            cv2.imshow(window, shown)

            before = len(clicks)
            key = cv2.waitKey(10) & 255
            if key in (ord("q"), 27):
                print("Calibration cancelled; previous file was kept")
                return
            if key == ord("f"):
                frozen = live.copy()
            if key == ord("r"):
                frozen = None
            if len(clicks) > before:
                frozen = None

        if args.plane == "serve":
            goal = map_points(clicks, data["surface"]["matrix"])[0]
            margin = check_clearance(
                args.table_clearance_m
                if args.table_clearance_m is not None
                else data.get("edge_clearance_m", settings["robot_radius_m"]),
                w,
                d,
            )
            if not (margin <= goal[0] <= w - margin and margin <= goal[1] <= d - margin):
                raise ValueError("Serving point is too close to an edge")
            data["serving_point_m"] = goal.tolist()
            if args.table_clearance_m is not None:
                data["edge_clearance_m"] = margin
        else:
            fit = fit_plane(clicks, points, 0.02 if args.mode == "table" else 0.04)
            if args.plane == "surface":
                # clearing the old marker map and serving point when recalibrating the surface
                data = {
                    "version": 1,
                    "mode": args.mode,
                    "width_m": w,
                    "depth_m": d,
                    "image_size": list(live.shape[1::-1]),
                    "camera_index": settings["camera_index"],
                    "surface": fit,
                }
            else:
                data["robot"] = fit
                data["marker_height_m"] = args.marker_height_m

        save(path, data)
        print(f"Saved {path}")
    finally:
        camera.close()
        cv2.destroyAllWindows()
