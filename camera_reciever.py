import zmq
import cv2
import numpy as np
import struct

ctx = zmq.Context()
sock = ctx.socket(zmq.SUB)
sock.connect("ipc:///tmp/camera.ipc")
sock.setsockopt(zmq.SUBSCRIBE, b"")

HEADER_FMT = "!III"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

EXPECTED_W = 800
EXPECTED_H = 600
EXPECTED_C = 3
EXPECTED_SIZE = EXPECTED_W * EXPECTED_H * EXPECTED_C

while True:
    data = sock.recv()

    # Header kontrolü
    if len(data) < HEADER_SIZE:
        continue

    h, w, c = struct.unpack(HEADER_FMT, data[:HEADER_SIZE])

    # Güvenlik / senkron kontrolü
    if (h, w, c) != (EXPECTED_H, EXPECTED_W, EXPECTED_C):
        continue

    payload = data[HEADER_SIZE:]

    if len(payload) != EXPECTED_SIZE:
        continue

    # Frame oluştur
    frame = np.frombuffer(payload, dtype=np.uint8).reshape((h, w, c))

    # 🔧 RENK DÜZELTME (IMX708 / libcamera)
    # libcamera pipeline pratikte RGB gibi geliyor
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()