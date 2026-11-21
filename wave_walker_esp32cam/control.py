"""I'm steering using camera positions without assuming a fixed walking speed."""

from dataclasses import dataclass
from math import atan2, hypot, isfinite, pi


def distance(a, b):
    return hypot(a[0] - b[0], a[1] - b[1])


def wrap_angle(a):
    return (a + pi) % (2 * pi) - pi


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    heading: float

    @property
    def xy(self):
        return self.x, self.y


class Controller:
    def __init__(self, settings, width, depth, mode="floor"):
        if mode not in ("floor", "table"):
            raise ValueError("Choose floor or table mode")

        self.mode = mode
        self.s = dict(settings)
        if mode == "table":
            self.s["stop_distance_m"] = settings["table_goal_tolerance_m"]

        self.width, self.depth = float(width), float(depth)
        self.active = False
        self.reason = "Waiting for a wave"
        self.target = None
        self.previous = None

    def inside(self, p, margin=0.0):
        return (
            all(isfinite(v) for v in p)
            and margin <= p[0] <= self.width - margin
            and margin <= p[1] <= self.depth - margin
        )

    def stop(self, reason):
        self.active = False
        self.reason = reason
        return 0, 0

    def start(self, pose, person, now):
        if self.active:
            return False
        if not all(isfinite(v) for v in (*pose.xy, pose.heading, *person, now)):
            self.reason = "Invalid position or timestamp"
            return False
        if not self.inside(pose.xy, self.s["robot_radius_m"]) or not self.inside(person):
            self.reason = "Robot or person outside the calibrated area"
            return False

        d = distance(pose.xy, person)
        if d <= self.s["stop_distance_m"] + 0.04:
            self.reason = "Already within the stopping distance"
            return False
        if d - self.s["stop_distance_m"] > self.s["max_travel_m"]:
            self.reason = "Person is beyond the allowed trip distance"
            return False

        # I'm leaving room for the whole robot at the stopping point.
        ratio = self.s["stop_distance_m"] / d
        goal = (
            person[0] + ratio * (pose.x - person[0]),
            person[1] + ratio * (pose.y - person[1]),
        )
        if not self.inside(goal, self.s["robot_radius_m"]):
            self.reason = "Approach endpoint is too close to the area boundary"
            return False

        self.target = tuple(person)
        self.started = self.progress_at = now
        self.progress_pose = pose
        self.previous = (pose, now)
        self.travel = 0.0
        self.travel_anchor = pose.xy
        self.active = True
        self.reason = (
            "Approaching the waving person"
            if self.mode == "floor"
            else "Going to the serving spot"
        )

        return True

    def update(self, pose, person, now):
        if not self.active:
            return 0, 0
        if pose is None or person is None:
            return self.stop("STOP: robot marker or person is not visible")
        if not all(isfinite(v) for v in (*pose.xy, pose.heading, *person, now)):
            return self.stop("STOP: invalid position")
        if not self.inside(pose.xy, self.s["robot_radius_m"]):
            return self.stop("STOP: area boundary")
        if not self.inside(person) or distance(person, self.target) > self.s["person_motion_limit_m"]:
            return self.stop("STOP: person moved; stand still and wave again")
        if now - self.started > self.s["max_trip_seconds"]:
            return self.stop("STOP: trip time limit")

        previous, previous_time = self.previous
        dt = now - previous_time
        if dt <= 0 or dt > self.s["max_frame_age_seconds"]:
            return self.stop("STOP: position updates are too slow")
        increment = distance(previous.xy, pose.xy)
        if (
            increment > 0.05 + 0.7 * dt
            or abs(wrap_angle(pose.heading - previous.heading)) > 0.22 + 4.0 * dt
        ):
            return self.stop("STOP: implausible marker jump")

        # I'm counting movement once it reaches 2 cm so small tracking mistakees don't add up
        segment = distance(self.travel_anchor, pose.xy)
        if segment >= 0.02:
            self.travel += segment
            self.travel_anchor = pose.xy
        self.previous = (pose, now)
        if self.travel > self.s["max_travel_m"]:
            return self.stop("STOP: travel limit")

        if (
            distance(pose.xy, self.progress_pose.xy) > 0.025
            or abs(wrap_angle(pose.heading - self.progress_pose.heading)) > 0.12
        ):
            self.progress_at, self.progress_pose = now, pose
        elif now - self.progress_at > self.s["no_progress_seconds"]:
            return self.stop("STOP: no measured motion")

        d = distance(pose.xy, self.target)
        if d <= self.s["stop_distance_m"] + 0.04:
            return self.stop(
                "ARRIVED: stopped near the person"
                if self.mode == "floor"
                else "ARRIVED: stopped at the serving spot"
            )
        error = wrap_angle(atan2(self.target[1] - pose.y, self.target[0] - pose.x) - pose.heading)
        turn = self.s["turn_pwm"]

        # turning with one side moving forward and the other stopped.
        if abs(error) > 0.40:
            self.reason = "Turning toward the destination"
            return (0, turn) if error > 0 else (turn, 0)

        remaining = max(0.0, d - self.s["stop_distance_m"])
        cruise = int(round(
            self.s["minimum_pwm"]
            + min(1.0, remaining / 0.6) * (self.s["maximum_pwm"] - self.s["minimum_pwm"])
        ))
        slow = max(self.s["minimum_pwm"], int(cruise * (1.0 - 1.5 * abs(error))))
        self.reason = f"Approaching: destination {d:.2f} m from robot centre"
        return (slow, cruise) if error > 0 else (cruise, slow)
