from math import radians
import os
from pathlib import Path
import pty
import select
import sys
from threading import Event, Thread
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from usb_link import USBLink
from walker import WaveGate
from wave_detector import Hand, measure_hand


class Wire:
    def __init__(self, drop_drive=False):
        self.master, self.slave = pty.openpty()
        self.port = os.ttyname(self.slave)
        self.stop = Event()
        self.messages = []
        self.drop_drive = drop_drive

        self.thread = Thread(target=self.run, daemon=True)
        self.thread.start()

    def run(self):
        pending = b""
        while not self.stop.is_set():
            if not select.select([self.master], [], [], 0.02)[0]:
                continue

            try:
                pending += os.read(self.master, 4096)
            except OSError:
                return

            while b"\n" in pending:
                raw, pending = pending.split(b"\n", 1)
                line = raw.decode().strip()
                self.messages.append(line)
               
                fields = line.split()
                response = None
               
                if line == "HELLO":
                    response = "READY WALKER 1 1 120"
                if line == "HALT":
                    response = "ACK HALT"
                if fields and fields[0] == "ARM":
                    response = "ACK " + line
                if fields and fields[0] == "DRIVE" and not self.drop_drive:
                    response = "ACK " + " ".join(fields[:3])
                if response:
                    os.write(self.master, (response + "\n").encode())

    def close(self):
        self.stop.set()
        self.thread.join(timeout=0.2)
        os.close(self.master)
        os.close(self.slave)


class WaveUSBTests(unittest.TestCase):
    def test_wave_requires_reversal_and_pause(self):
        gate = WaveGate()

        def hand(x):
            return Hand(np.array([x, 100.0]), 80.0, radians(0), True)

        events = []
        values = [100, 115, 135, 155, 170, 155, 135, 115, 95]
        for i, x in enumerate(values):
            events.append(gate.update(hand(x), i * 0.06))
        self.assertEqual(sum(events), 1)

        for i in range(10):
            self.assertFalse(gate.update(hand(100 + 50 * (i % 2)), 0.7 + i * 0.06))
        for i in range(15):
            gate.update(None, 1.4 + i * 0.06)

        events = [gate.update(hand(x), 2.5 + i * 0.06) for i, x in enumerate(values)]
        self.assertEqual(sum(events), 1)

    def test_stationary_palm_is_not_a_wave(self):
        gate = WaveGate()
    
        h = Hand(np.array([100.0, 100.0]), 80.0, 0.0, True)
        self.assertFalse(any(gate.update(h, i * 0.05) for i in range(80)))

    def test_closing_hand_cancels_wave(self):
        gate = WaveGate()
        values = [100, 115, 135, 155, 170, 155, 135, 115, 95]

        for i, x in enumerate(values):
            hand = Hand(np.array([x, 100.0]), 80.0, 0.0, i < 5)
            self.assertFalse(gate.update(hand, i * 0.06))

    def test_fist_and_uncertain_palm_cannot_trigger(self):
        landmarks = [SimpleNamespace(x=0.5, y=0.5) for _ in range(21)]
        landmarks[5].x = 0.4
      
        landmarks[17].x = 0.6
        landmarks[9].y = 0.4
        result = SimpleNamespace(hand_landmarks=[landmarks], hand_world_landmarks=[], gestures=[[]])

        for gesture, score, is_open in (
            ("Closed_Fist", 0.96, False),
            ("Open_Palm", 0.64, False),
            ("Open_Palm", 0.95, True),
        ):
            result.gestures = [[SimpleNamespace(category_name=gesture, score=score)]]
            hand = measure_hand(result, 640, 480, (160, 120, 320, 240))
        
            self.assertEqual(hand.open_enough, is_open)
            np.testing.assert_allclose(hand.wrist, [320, 240])

    def test_finger_check_keeps_fists_closed(self):
        landmarks = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(21)]
        for finger, base in enumerate((5, 9, 13, 17)):
            for joint in range(4):
                landmarks[base + joint] = SimpleNamespace(
                    x=0.35 + 0.08 * finger, y=0.35 - 0.04 * joint, z=0.0,
                )

        result = SimpleNamespace(
            hand_landmarks=[landmarks], hand_world_landmarks=[landmarks], gestures=[[]],
        )
        zone = (0, 0, 640, 480)
        result.gestures = [[SimpleNamespace(category_name="None", score=0.9)]]
        self.assertTrue(measure_hand(result, 640, 480, zone).open_enough)

        result.gestures = [[SimpleNamespace(category_name="Closed_Fist", score=0.9)]]
        self.assertFalse(measure_hand(result, 640, 480, zone).open_enough)

    def test_small_open_hand_can_wave(self):
        gate = WaveGate()
        moves = [100, 103, 107, 112, 115, 112, 107, 102, 98]
        events = [
            gate.update(Hand(np.array([x, 100.0]), 12.0, 0.0, True), i * 0.06)
            for i, x in enumerate(moves)
        ]
        self.assertEqual(sum(events), 1)

    def connect(self, wire):
        with patch("usb_link.sleep", lambda _: None):
            return USBLink(wire.port)

    def test_usb_drive_and_stop(self):
        wire = Wire()
        link = self.connect(wire)

        try:
            link.arm()
            link.drive(65, 75)
            link.halt()

            self.assertIsNone(link.session)
            self.assertTrue(any(x.startswith("DRIVE ") for x in wire.messages))
        finally:
            link.close()
            wire.close()

    def test_missing_ack_halts_and_does_not_resume(self):
        wire = Wire(drop_drive=True)
        link = self.connect(wire)

        try:
            link.arm()
            with self.assertRaisesRegex(RuntimeError, "acknowledgement"):
                link.drive(65, 75)
            self.assertIsNone(link.session)
            with self.assertRaises(RuntimeError):
                link.drive(65, 75)
        finally:
            link.close()
            wire.close()

    def test_invalid_pwm_disarms_host(self):
        wire = Wire()
        link = self.connect(wire)

        try:
            link.arm()
            with self.assertRaises(ValueError):
                link.drive(121, 75)
            self.assertIsNone(link.session)
        finally:
            link.close()
            wire.close()


if __name__ == "__main__":
    unittest.main()
