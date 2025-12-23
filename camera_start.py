import atexit
import signal
import sys
import time
import zmq
from picamera2 import Picamera2
import numpy as np
import struct

ctx = zmq.Context()
sock = ctx.socket(zmq.PUB)
sock.bind("ipc:///tmp/camera.ipc")

picam2 = None

def cleanup():
    if picam2 is not None:
        try:
            picam2.stop()
        except Exception:
            pass
        try:
            picam2.close()
        except Exception:
            pass
    try:
        sock.close(0)
    except Exception:
        pass
    try:
        ctx.term()
    except Exception:
        pass

def handle_signal(signum, frame):
    cleanup()
    sys.exit(0)

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)
atexit.register(cleanup)

def init_camera(retries=8, delay_s=0.7):
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            cam = Picamera2()
            cam.configure(
                cam.create_preview_configuration(
                    main={"size": (640, 480), "format": "BGR888"}
                )
            )
            cam.start()
            return cam
        except Exception as exc:
            last_exc = exc
            try:
                cam.stop()
            except Exception:
                pass
            try:
                cam.close()
            except Exception:
                pass
            time.sleep(delay_s)
    raise RuntimeError(
        f"Kamera baslatilamadi ({retries} deneme). "
        "Baska bir surec kamerayi kullaniyor olabilir."
    ) from last_exc

picam2 = init_camera()

HEADER_FMT = "!III"   # network endian
HEADER_SIZE = struct.calcsize(HEADER_FMT)

while True:
    frame = picam2.capture_array()
    h, w, c = frame.shape

    header = struct.pack(HEADER_FMT, h, w, c)
    payload = frame.tobytes()

    sock.send(header + payload)
