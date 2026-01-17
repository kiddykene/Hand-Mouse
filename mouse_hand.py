import cv2
import mediapipe as mp
import numpy as np
import sys
import keyboard
import zhmiscellany
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QSlider, QVBoxLayout, QHBoxLayout, QSizePolicy
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QPolygonF, QImage, QPixmap
from PyQt5.QtCore import Qt, QPoint, QPointF, QTimer

zhmiscellany.misc.die_on_key()

is_c_pressed = False
SMOOTH_ALPHA = 0.2
FAST_MOVEMENT_ALPHA = 0.8
FAST_MOVEMENT_THRESHOLD = 50
smoothed_pos = None
smoothed_translation_pos = None
fov_factor = 0.55
frame_interval_ms = 20

def on_c_key_press(event):
    global is_c_pressed
    if event.name == 'c':
        is_c_pressed = True

keyboard.on_press_key('c', on_c_key_press)

class Overlay(QWidget):
    def __init__(self, screen_size):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setGeometry(0, 0, screen_size.width(), screen_size.height())
        self.pointer_pos = None
        self.quad_points = None
    def set_pointer_pos(self, pos):
        self.pointer_pos = QPoint(int(pos[0]), int(pos[1])) if pos else None
        self.update()
    def set_quad_points(self, points):
        self.quad_points = [tuple(p) for p in points] if points and len(points) == 4 else None
        self.update()
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        if self.quad_points:
            painter.setBrush(QBrush(QColor(173, 216, 230, 80)))
            painter.setPen(QPen(Qt.white, 2, Qt.SolidLine))
            painter.drawPolygon(QPolygonF([QPointF(p[0], p[1]) for p in self.quad_points]))
        if self.pointer_pos:
            painter.setPen(QPen(Qt.red, 4))
            painter.drawEllipse(self.pointer_pos, 20, 20)

def main():
    global is_c_pressed, smoothed_pos, smoothed_translation_pos, SMOOTH_ALPHA, FAST_MOVEMENT_ALPHA, FAST_MOVEMENT_THRESHOLD, fov_factor, frame_interval_ms

    app = QApplication(sys.argv)
    screen = app.primaryScreen()
    W, H = screen.geometry().width(), screen.geometry().height()
    overlay = Overlay(screen.geometry())
    overlay.show()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        sys.exit(app.exec_())

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    main_win = QWidget()
    main_win.setWindowTitle("Camera + Controls")
    cam_label = QLabel()
    cam_label.setFixedSize(cam_w, cam_h)
    cam_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    controls = QWidget()
    vbox = QVBoxLayout(controls)

    def mk_slider(minv, maxv, start, step=1):
        s = QSlider(Qt.Horizontal)
        s.setRange(minv, maxv)
        s.setSingleStep(step)
        s.setValue(start)
        return s

    smooth_label = QLabel(f"SMOOTH_ALPHA: {SMOOTH_ALPHA:.2f}")
    smooth_s = mk_slider(0, 100, int(SMOOTH_ALPHA * 100))
    def upd_smooth(v):
        global SMOOTH_ALPHA
        SMOOTH_ALPHA = v / 100.0
        smooth_label.setText(f"SMOOTH_ALPHA: {SMOOTH_ALPHA:.2f}")
    smooth_s.valueChanged.connect(upd_smooth)

    fast_label = QLabel(f"FAST_MOVEMENT_ALPHA: {FAST_MOVEMENT_ALPHA:.2f}")
    fast_s = mk_slider(0, 100, int(FAST_MOVEMENT_ALPHA * 100))
    def upd_fast(v):
        global FAST_MOVEMENT_ALPHA
        FAST_MOVEMENT_ALPHA = v / 100.0
        fast_label.setText(f"FAST_MOVEMENT_ALPHA: {FAST_MOVEMENT_ALPHA:.2f}")
    fast_s.valueChanged.connect(upd_fast)

    thresh_label = QLabel(f"FAST_MOVEMENT_THRESHOLD: {FAST_MOVEMENT_THRESHOLD}")
    thresh_s = mk_slider(0, 300, int(FAST_MOVEMENT_THRESHOLD))
    def upd_thresh(v):
        global FAST_MOVEMENT_THRESHOLD
        FAST_MOVEMENT_THRESHOLD = int(v)
        thresh_label.setText(f"FAST_MOVEMENT_THRESHOLD: {FAST_MOVEMENT_THRESHOLD}")
    thresh_s.valueChanged.connect(upd_thresh)

    fov_label = QLabel(f"fov_factor: {fov_factor:.2f}")
    fov_s = mk_slider(0, 100, int(fov_factor * 100))
    def upd_fov(v):
        global fov_factor
        fov_factor = v / 100.0
        fov_label.setText(f"fov_factor: {fov_factor:.2f}")
    fov_s.valueChanged.connect(upd_fov)

    interval_label = QLabel(f"interval ms: {frame_interval_ms}")
    interval_s = mk_slider(5, 200, int(frame_interval_ms))
    def upd_interval(v):
        global frame_interval_ms, timer
        frame_interval_ms = int(v)
        interval_label.setText(f"interval ms: {frame_interval_ms}")
        timer.setInterval(frame_interval_ms)
    interval_s.valueChanged.connect(upd_interval)

    vbox.addWidget(smooth_label); vbox.addWidget(smooth_s)
    vbox.addWidget(fast_label); vbox.addWidget(fast_s)
    vbox.addWidget(thresh_label); vbox.addWidget(thresh_s)
    vbox.addWidget(fov_label); vbox.addWidget(fov_s)
    vbox.addWidget(interval_label); vbox.addWidget(interval_s)
    vbox.addStretch(1)

    layout = QHBoxLayout(main_win)
    layout.addWidget(cam_label)
    layout.addWidget(controls)
    main_win.show()

    hands = mp.solutions.hands.Hands(model_complexity=0, min_detection_confidence=0.7, min_tracking_confidence=0.7, max_num_hands=1)

    image_fingertip_points = []
    is_calibrated = False
    homography_matrix = None
    calibrated_quad_points = None
    last_prompt_idx = -1
    calibration_prompts = [
        "Point to the TOP-LEFT corner and press 'c'",
        "Point to the TOP-RIGHT corner and press 'c'",
        "Point to the BOTTOM-RIGHT corner and press 'c'",
        "Point to the BOTTOM-LEFT corner and press 'c'",
    ]

    initial_calibration_points = None
    initial_translation_point = None

    def update_homograph(new_image_points):
        nonlocal homography_matrix, calibrated_quad_points
        screen_corners = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
        homography_matrix, _ = cv2.findHomography(new_image_points, screen_corners)
        if homography_matrix is not None:
            calibrated_quad_points = new_image_points
            return True
        return False

    def get_scaling_factor(current_pos, cam_w_local, cam_h_local):
        center_x, center_y = cam_w_local / 2, cam_h_local / 2
        max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
        dist_from_center = np.linalg.norm(current_pos - np.array([center_x, center_y]))
        normalized_dist = min(dist_from_center / max_dist, 1.0)
        return 1.0 + (fov_factor * (1.0 - normalized_dist)) - fov_factor

    def process_frame():
        nonlocal is_calibrated, homography_matrix, last_prompt_idx, calibrated_quad_points, initial_calibration_points, initial_translation_point
        global is_c_pressed, smoothed_pos, smoothed_translation_pos

        success, image = cap.read()
        if not success:
            return

        image = cv2.flip(image, 1)
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        fingertip_pos = None
        raw_translation_point_pos = None

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            cam_h_local, cam_w_local, _ = image.shape

            index_fingertip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            wrist = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.WRIST]
            index_mcp = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP]

            fingertip_pos = np.array([index_fingertip.x * cam_w_local, index_fingertip.y * cam_h_local])

            avg_x = (wrist.x + index_mcp.x) / 2
            avg_y = (wrist.y + index_mcp.y) / 2
            raw_translation_point_pos = np.array([avg_x * cam_w_local, avg_y * cam_h_local])

            if smoothed_translation_pos is None:
                smoothed_translation_pos = raw_translation_point_pos
            else:
                speed = np.linalg.norm(raw_translation_point_pos - smoothed_translation_pos)
                alpha = FAST_MOVEMENT_ALPHA if speed > FAST_MOVEMENT_THRESHOLD else SMOOTH_ALPHA
                smoothed_translation_pos = np.array([alpha * raw_translation_point_pos[0] + (1 - alpha) * smoothed_translation_pos[0],
                                                     alpha * raw_translation_point_pos[1] + (1 - alpha) * smoothed_translation_pos[1]])

            mp.solutions.drawing_utils.draw_landmarks(image, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            cv2.circle(image, (int(fingertip_pos[0]), int(fingertip_pos[1])), 8, (0, 255, 0), -1)
            cv2.circle(image, (int(smoothed_translation_pos[0]), int(smoothed_translation_pos[1])), 8, (255, 0, 0), -1)

        if not is_calibrated:
            prompt_idx = len(image_fingertip_points)
            if prompt_idx < len(calibration_prompts) and prompt_idx != last_prompt_idx:
                print(f"\n--- CALIBRATION STEP {prompt_idx + 1}/{len(calibration_prompts)} ---")
                print(calibration_prompts[prompt_idx])
                last_prompt_idx = prompt_idx

            if is_c_pressed and fingertip_pos is not None and smoothed_translation_pos is not None:
                image_fingertip_points.append(fingertip_pos)
                is_c_pressed = False
                print(f"Captured point {len(image_fingertip_points)}: {fingertip_pos}")
                if len(image_fingertip_points) == 4:
                    if update_homograph(np.array(image_fingertip_points, dtype=np.float32)):
                        is_calibrated = True
                        initial_calibration_points = np.array(image_fingertip_points, dtype=np.float32)
                        initial_translation_point = smoothed_translation_pos
                        print("Calibration successful. Starting tracking.")
                    else:
                        print("Calibration failed. Please try again.")
                        image_fingertip_points.clear()
                        last_prompt_idx = -1
        else:
            if initial_calibration_points is not None and smoothed_translation_pos is not None and initial_translation_point is not None:
                cam_h_local, cam_w_local, _ = image.shape
                scaling_factor = get_scaling_factor(smoothed_translation_pos, cam_w_local, cam_h_local)
                translation_vector = smoothed_translation_pos - np.array(initial_translation_point)
                center_of_quad = np.mean(initial_calibration_points, axis=0)
                scaled_points = (initial_calibration_points - center_of_quad) * scaling_factor + center_of_quad
                new_image_points = scaled_points + translation_vector
                update_homograph(new_image_points)

            if calibrated_quad_points is not None:
                quad_points_int = calibrated_quad_points.astype(np.int32)
                overlay_fill = image.copy()
                cv2.fillPoly(overlay_fill, [quad_points_int], (230, 216, 173))
                image = cv2.addWeighted(overlay_fill, 0.3, image, 0.7, 0)
                cv2.polylines(image, [quad_points_int], True, (255, 255, 255), 2)

                if fingertip_pos is not None and homography_matrix is not None:
                    fingertip_np = np.array([[fingertip_pos]], dtype=np.float32)
                    screen_pos_transformed = cv2.perspectiveTransform(fingertip_np, homography_matrix)
                    if screen_pos_transformed is not None:
                        sx, sy = screen_pos_transformed[0, 0]
                        if smoothed_pos is None:
                            smoothed_pos = (sx, sy)
                        else:
                            speed = np.linalg.norm(np.array(smoothed_pos) - np.array([sx, sy]))
                            alpha = FAST_MOVEMENT_ALPHA if speed > FAST_MOVEMENT_THRESHOLD else SMOOTH_ALPHA
                            smoothed_pos = (alpha * sx + (1 - alpha) * smoothed_pos[0], alpha * sy + (1 - alpha) * smoothed_pos[1])
                        zhmiscellany.misc.click_pixel(smoothed_pos, act_start=False, act_end=False)
                        overlay.set_pointer_pos(smoothed_pos)
                    else:
                        overlay.set_pointer_pos(None)
                else:
                    overlay.set_pointer_pos(None)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        cam_label.setPixmap(pix)

    timer = QTimer()
    timer.timeout.connect(process_frame)
    timer.setInterval(frame_interval_ms)
    timer.start()

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
