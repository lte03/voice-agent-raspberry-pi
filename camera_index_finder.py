import cv2
import platform

def list_cameras(max_index=20, backend=None, test_frames=5, warmup=2):
    """
    Kameraları index tarayarak listeler.
    - max_index: 0..max_index-1 arası dener
    - backend: None ya da cv2.CAP_DSHOW / cv2.CAP_MSMF / cv2.CAP_V4L2 gibi
    """
    found = []

    for idx in range(max_index):
        if backend is None:
            cap = cv2.VideoCapture(idx)
        else:
            cap = cv2.VideoCapture(idx, backend)

        if not cap.isOpened():
            cap.release()
            continue

        # Bazı cihazlar ilk framelerde boş döner; biraz ısıt
        ok_any = False
        for i in range(test_frames):
            ret, frame = cap.read()
            if i < warmup:
                continue
            if ret and frame is not None and frame.size > 0:
                ok_any = True
                break

        if ok_any:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            found.append({"index": idx, "width": w, "height": h, "fps": fps})

        cap.release()

    return found

def pick_backend():
    sys = platform.system().lower()
    if "windows" in sys:
        # Windows'ta genelde DSHOW daha stabil
        return cv2.CAP_DSHOW
    if "linux" in sys:
        return cv2.CAP_V4L2
    # macOS için None bırakmak çoğu zaman yeterli
    return None

if __name__ == "__main__":
    backend = pick_backend()
    cams = list_cameras(max_index=30, backend=backend)

    if not cams:
        print("Hiç kamera bulunamadı.")
    else:
        print("Bulunan kameralar:")
        for c in cams:
            print(f"- index={c['index']}  {c['width']}x{c['height']}  fps={c['fps']}")

        # İstersen ilk kamerayı önizle
        first = cams[0]["index"]
        cap = cv2.VideoCapture(first, backend) if backend is not None else cv2.VideoCapture(first)
        print("\nÖnizleme: çıkmak için 'q'")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cv2.imshow(f"Camera {first}", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
