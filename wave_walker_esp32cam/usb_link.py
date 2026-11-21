"""I'm sending USB motor commands and stopping if the UNO doesn't recognize them."""

from secrets import randbelow
from time import monotonic, sleep


class USBLink:
    def __init__(self, port, max_pwm=120):
        import serial

        self.serial = serial.Serial(port, 115200, timeout=0.02, write_timeout=0.1)
        self.session = None
        self.sequence = 0
        self.max_pwm = max_pwm

        try:
            sleep(2.0)  # giving the UNO time to reset after opening USB.
            self.serial.reset_input_buffer()
            self.serial.write(b"HALT\nHELLO\n")
            deadline = monotonic() + 1.0

            while monotonic() < deadline:
                line = self.serial.readline().decode("ascii", "replace").strip()
                if line.startswith("READY WALKER 1 "):
                    fields = line.split()
                    self.enabled = fields[3] == "1"
                    self.max_pwm = min(max_pwm, int(fields[4]))
                    return

            raise RuntimeError(
                "UNO did not identify as WALKER 1. Upload walker_usb.ino and close Serial Monitor."
            )
        except BaseException:
            self.serial.close()
            raise

    def send_command(self, command, expected):
        try:
            self.serial.write((command + "\n").encode("ascii"))
            deadline = monotonic() + 0.12

            while monotonic() < deadline:
                line = self.serial.readline().decode("ascii", "replace").strip()
                if line == expected:
                    return
                if line.startswith("ERR "):
                    raise RuntimeError("UNO: " + line)

            raise RuntimeError("No command acknowledgement from UNO; trip cancelled")
        except BaseException:
            self.halt()
            raise

    def arm(self):
        session = randbelow(2**31 - 1) + 1
        self.send_command(f"ARM {session}", f"ACK ARM {session}")
        self.session, self.sequence = session, 0

    def drive(self, left, right):
        if self.session is None:
            raise RuntimeError("Motor link is not armed")
        if any(type(v) is not int or abs(v) > self.max_pwm for v in (left, right)):
            self.halt()
            raise ValueError("Invalid motor command")

        self.sequence += 1
        self.send_command(
            f"DRIVE {self.session} {self.sequence} {left} {right}",
            f"ACK DRIVE {self.session} {self.sequence}",
        )

    def halt(self):
        self.session = None
        try:
            self.serial.write(b"HALT\n")
        except Exception:
            pass

    def close(self):
        self.halt()
        self.serial.close()
