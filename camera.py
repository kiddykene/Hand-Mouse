import sys
import cv2

class Camera:
    def __init__(self):
        self.cap = None

    def _open_capture(self, index, backend):
        try:
            return (cv2.VideoCapture(int(index)) if backend == 0 else cv2.VideoCapture(int(index), backend))
        except Exception:
            return None

    def set_camera(self, index, desired_w=1920, desired_h=1080, warmup_frames=8, timeout=8):
        backends = []
        if sys.platform.startswith("win"):
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, 0]
        elif sys.platform.startswith("linux"):
            backends = [cv2.CAP_V4L2, 0]
        else:
            backends = [0]
        last_err = None
        for backend in backends:
            cap = self._open_capture(index, backend)
            if cap is None or not cap.isOpened():
                try:
                    if cap is not None:
                        cap.release()
                except Exception:
                    pass
                last_err = f"open failed backend={backend}"
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(desired_w))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(desired_h))
            observed = {}
            got_frame = False
            for _ in range(warmup_frames + timeout):
                r, frame = cap.read()
                if not r or frame is None:
                    continue
                got_frame = True
                h, w = frame.shape[:2]
                observed[(w, h)] = observed.get((w, h), 0) + 1
                if (w, h) == (desired_w, desired_h) and observed[(w, h)] >= 2:
                    break
            if not got_frame:
                try:
                    cap.release()
                except Exception:
                    pass
                last_err = f"no frames backend={backend}"
                continue
            old = getattr(self, "cap", None)
            if old is not None and old is not cap:
                try:
                    old.release()
                except Exception:
                    pass
            self.cap = cap
            return True
        return False

    def read_frame(self):
        if not hasattr(self, "cap") or self.cap is None or not self.cap.isOpened():
            return False, None
        return self.cap.read()

    def dump_camera_diagnostics(self, label="camera"):
        cap = getattr(self, "cap", None)
        if cap is None:
            print("no cap")
            return
        print(f"--- {label} diagnostics ---")
        try:
            reported_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            reported_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
            print("reported (prop):", reported_w, reported_h, "fps:", fps, "fourcc:", fourcc)
        except Exception as e:
            print("prop read error:", e)
        shapes = {}
        for i in range(5):
            ok, f = cap.read()
            if not ok or f is None:
                print("frame read failed at iter", i)
                continue
            h, w = f.shape[:2]
            shapes.setdefault((w, h), 0)
            shapes[(w, h)] += 1
        print("observed frame shapes and counts:", shapes)
        print("--------------------------")

    def get_camera_resolution(self, max_samples=8):
        try:
            if not hasattr(self, "cap") or self.cap is None or not self.cap.isOpened():
                return 640, 480
            observed = {}
            for _ in range(max_samples):
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    continue
                h, w = frame.shape[:2]
                observed[(w, h)] = observed.get((w, h), 0) + 1
            if observed:
                best = max(observed.items(), key=lambda kv: kv[1])[0]
                return int(best[0]), int(best[1])
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            if w > 0 and h > 0:
                return w, h
            return 640, 480
        except Exception:
            return 640, 480

    def release(self):
        try:
            if hasattr(self, "cap") and self.cap is not None:
                self.cap.release()
        except Exception:
            pass
