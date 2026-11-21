from pathlib import Path
import sys
import unittest
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calibrate import check_clearance
from tracking import MarkerTracker, feet_position, fit_plane, map_points, matches_wrist
from walker import ROOT, load_settings
from control import Controller, Pose
from wave_sim import advance


class TrackingTests(unittest.TestCase):
    def test_table_clearance_at_edges(self):
        settings = load_settings(ROOT / "settings.json")
        settings["robot_radius_m"] = check_clearance(0.42, 2, 1.2)
        controller = Controller(settings, 2, 1.2, "table")
        pose = Pose(0.55, 0.60, 0)

        self.assertFalse(controller.start(pose, (1.75, 0.60), 0.0))
        self.assertTrue(controller.start(pose, (1.50, 0.60), 0.0))
        self.assertEqual(controller.update(Pose(0.39, 0.60, 0), (1.50, 0.60), 0.02), (0, 0))
        self.assertEqual(controller.reason, "STOP: area boundary")

        with self.assertRaisesRegex(ValueError, "all four table edges"):
            check_clearance(0.60, 2, 1.2)

    def test_map_matches_measured_points(self):
        world = np.array([[0, 0], [2, 0], [2, 1], [0, 1], [0.5, 0.5], [1.5, 0.5]], float)
       
        h = np.array([[200, 20, 100], [10, -180, 400], [0.02, 0.01, 1]], float)
      
        pixels = map_points(world, h)
        fit = fit_plane(pixels, world, 0.02)

        np.testing.assert_allclose(map_points(pixels, fit["matrix"]), world, atol=1e-5)

    def test_bad_check_points_are_rejected(self):
        world = np.array([[0, 0], [2, 0], [2, 1], [0, 1], [0.5, 0.5], [1.5, 0.5]], float)
        pixels = world * 200 + 100
        pixels[-1] += 100

        with self.assertRaisesRegex(ValueError, "check error"):
            fit_plane(pixels, world, 0.02)

    def test_crossed_corners_are_rejected(self):
        world = np.array([[0, 0], [2, 0], [2, 1], [0, 1], [0.5, 0.5], [1.5, 0.5]], float)
        pixels = world * 200 + 100
        pixels[[1, 2]] = pixels[[2, 1]]

        with self.assertRaises(ValueError):
            fit_plane(pixels, world, 0.02)

    def test_marker_detection_and_heading(self):
        settings = load_settings(ROOT / "settings.json")
      
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        marker = cv2.aruco.generateImageMarker(dictionary, 7, 240)
       
        frame = np.full((500, 600), 255, np.uint8)
       
        frame[150:390, 180:420] = marker
        config = {"robot": {"matrix": [[0.0005, 0, 0], [0, -0.0005, 1], [0, 0, 1]]}}
      
        p, _ = MarkerTracker(settings, config).detect(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))

        self.assertIsNotNone(p)
        self.assertAlmostEqual(p.x, 0.15, delta=0.002)
        self.assertAlmostEqual(p.y, 0.865, delta=0.002)
        self.assertAlmostEqual(p.heading, np.pi / 2, delta=0.03)

    def test_no_pose_without_marker(self):
        settings = load_settings(ROOT / "settings.json")
        tracker = MarkerTracker(settings, {"robot": {"matrix": np.eye(3).tolist()}})
        self.assertIsNone(tracker.detect(np.full((400, 400, 3), 255, np.uint8))[0])

    def test_low_confidence_feet_are_rejected(self):
        landmarks = [SimpleNamespace(x=0.5, y=0.5, visibility=0.1) for _ in range(33)]
        self.assertIsNone(feet_position(landmarks, 100, 100, np.eye(3)))

    def test_hand_must_match_person_wrist(self):
        pose = [SimpleNamespace(x=0.5, y=0.5, visibility=1.0) for _ in range(33)]
        hand = np.array([0.9 * 640, 0.1 * 480])
        self.assertFalse(matches_wrist(hand, pose, 640, 480))

        hand = np.array([0.5 * 640, 0.5 * 480])
        self.assertTrue(matches_wrist(hand, pose, 640, 480))

    def test_walker_reaches_goal_using_marker(self):
        settings = load_settings(ROOT / "settings.json")
        image_from_world = np.array([[300, 0, 60], [0, -300, 1000], [0, 0, 1]], float)
      
        config = {"robot": {"matrix": np.linalg.inv(image_from_world).tolist()}}
        tracker = MarkerTracker(settings, config)
        # dict
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        marker = cv2.aruco.generateImageMarker(dictionary, 7, 256)
        source = np.array([[0, 0], [255, 0], [255, 255], [0, 255]], np.float32)

        # I'm pointing the markers top edge toward the front of the robot!!
        local = np.array([[0.06, 0.06], [0.06, -0.06], [-0.06, -0.06], [-0.06, 0.06]])
        pose = Pose(0.7, 1.5, 0.4)
        target = (2.8, 1.5)
        c = Controller(settings, 4, 3)

        for i in range(900):
            angle = pose.heading
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
           
            corners = map_points(local @ rotation.T + pose.xy, image_from_world).astype(np.float32)
            
            homography = cv2.getPerspectiveTransform(source, corners)
           
            frame = cv2.warpPerspective(marker, homography, (1350, 1060), borderValue=255)
            measured, _ = tracker.detect(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
            self.assertIsNotNone(measured)

            if i == 0:
                self.assertTrue(c.start(measured, target, 0))
                continue

            left, right = c.update(measured, target, i * 0.05)
            if not c.active:
                break
            pose = advance(pose, left, right, 0.05)

        self.assertIn("ARRIVED", c.reason)


if __name__ == "__main__":
    unittest.main()
