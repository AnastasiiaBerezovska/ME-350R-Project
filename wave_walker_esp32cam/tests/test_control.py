import json
from math import pi
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from control import Controller, Pose, distance
from wave_sim import advance

SETTINGS = json.loads((Path(__file__).resolve().parents[1] / "settings.json").read_text())


class ControlTests(unittest.TestCase):
    def make_controller(self, mode="floor"):
        return Controller(SETTINGS, 4.0, 3.0, mode)

    def test_wave_starts_one_trip(self):
        c = self.make_controller()

        self.assertTrue(c.start(Pose(0.6, 1.5, 0), (2.8, 1.5), 0))
        self.assertFalse(c.start(Pose(0.6, 1.5, 0), (2.5, 1.0), 0.01))
        self.assertEqual(c.target, (2.8, 1.5))

    def test_stop_when_marker_is_lost(self):
        c = self.make_controller()
        c.start(Pose(0.6, 1.5, 0), (2.8, 1.5), 0)

        self.assertEqual(c.update(None, (2.8, 1.5), 0.1), (0, 0))
        self.assertFalse(c.active)

    def test_stop_when_person_is_lost(self):
        c = self.make_controller()
        c.start(Pose(0.6, 1.5, 0), (2.8, 1.5), 0)

        self.assertEqual(c.update(Pose(0.6, 1.5, 0), None, 0.1), (0, 0))
        self.assertFalse(c.active)

    def test_stop_when_person_moves(self):
        c = self.make_controller()
        c.start(Pose(0.6, 1.5, 0), (2.8, 1.5), 0)

        self.assertEqual(c.update(Pose(0.6, 1.5, 0), (3.1, 1.5), 0.1), (0, 0))
        self.assertFalse(c.active)

    def test_stop_at_edge(self):
        c = self.make_controller()
        c.start(Pose(0.6, 1.5, 0), (2.8, 1.5), 0)

        self.assertEqual(c.update(Pose(0.1, 1.5, 0), (2.8, 1.5), 0.1), (0, 0))
        self.assertIn("boundary", c.reason)

    def test_stale_updates_stop(self):
        c = self.make_controller()
        c.start(Pose(0.6, 1.5, 0), (2.8, 1.5), 0)

        self.assertEqual(c.update(Pose(0.6, 1.5, 0), (2.8, 1.5), 0.4), (0, 0))
        self.assertFalse(c.active)

    def test_slow_motion_counts_toward_travel_limit(self):
        s = dict(SETTINGS, max_travel_m=0.20)
        c = Controller(s, 4, 3)
        self.assertTrue(c.start(Pose(1.2, 1.2, 0), (2.2, 1.2), 0))

        for i in range(1, 70):
            c.update(Pose(1.2, 1.2 + i * 0.005, 0), (2.2, 1.2), i * 0.05)
            if not c.active:
                break

        self.assertIn("travel limit", c.reason)

    def test_trip_time_limit_stops(self):
        s = dict(SETTINGS, max_trip_seconds=1.0)
        c = Controller(s, 4, 3)
        self.assertTrue(c.start(Pose(0.7, 1.5, 0), (2.8, 1.5), 0))

        for i in range(1, 15):
            c.update(Pose(0.7 + i * 0.003, 1.5, 0), (2.8, 1.5), i * 0.1)
            if not c.active:
                break

        self.assertIn("time limit", c.reason)

    def test_stop_on_position_jump(self):
        c = self.make_controller()
        c.start(Pose(0.6, 1.5, 0), (2.8, 1.5), 0)

        c.update(Pose(1.6, 1.5, 0), (2.8, 1.5), 0.1)
        self.assertIn("jump", c.reason)

    def test_nan_is_rejected(self):
        c = self.make_controller()
        self.assertFalse(c.start(Pose(float("nan"), 1.5, 0), (2.8, 1.5), 0))

    def test_nearby_person_does_not_start(self):
        c = self.make_controller()
        self.assertFalse(c.start(Pose(1.5, 1.5, 0), (2.0, 1.5), 0))

    def test_motionless_robot_stops(self):
        c = self.make_controller()
        p = Pose(0.6, 1.5, 0)
        person = (2.8, 1.5)
        c.start(p, person, 0)

        for i in range(1, 40):
            c.update(p, person, i * 0.1)
            if not c.active:
                break

        self.assertIn("no measured motion", c.reason)

    def test_approach_from_different_angles(self):
        for heading in [0, 0.7, -0.7, pi]:
            with self.subTest(heading=heading):
                c = self.make_controller()
                p = Pose(0.7, 1.5, heading)
                target = (2.8, 1.5)
                self.assertTrue(c.start(p, target, 0))

                for i in range(1, 4501):
                    left, right = c.update(p, target, i * 0.01)
                    self.assertTrue(0 <= left <= 120 and 0 <= right <= 120)
                    if not c.active:
                        break
                    p = advance(p, left, right, 0.01)

                self.assertTrue(c.reason.startswith("ARRIVED"), c.reason)
                self.assertLessEqual(distance(p.xy, target), SETTINGS["stop_distance_m"] + 0.04)
                self.assertGreater(distance(p.xy, target), SETTINGS["stop_distance_m"] - 0.03)

    def test_table_trip_stays_inside(self):
        c = Controller(SETTINGS, 2.0, 1.2, "table")
        p = Pose(0.5, 0.6, 0)
        goal = (1.5, 0.6)
        self.assertTrue(c.start(p, goal, 0))

        for i in range(1, 4501):
            left, right = c.update(p, goal, i * 0.01)
            self.assertTrue(c.inside(p.xy, SETTINGS["robot_radius_m"]))
            if not c.active:
                break
            p = advance(p, left, right, 0.01)

        self.assertIn("ARRIVED", c.reason)
        self.assertLess(distance(p.xy, goal), 0.09)

    def test_table_edge_destination_is_rejected(self):
        c = Controller(SETTINGS, 2.0, 1.2, "table")
        self.assertFalse(c.start(Pose(0.5, 0.6, 0), (1.99, 0.6), 0))


if __name__ == "__main__":
    unittest.main()
