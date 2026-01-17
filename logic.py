import cv2
import mediapipe as mp
import numpy as np
import zhmiscellany, time
from threading import Thread
from camera import Camera
from input_handler import InputHandler
from smoothing import smooth_translation, smooth_pointer_time
import utils
import draw
from mp_reader import MPReader
import math
from ctypes import windll
from collections import deque
zhmiscellany.misc.click_pixel((0, 0), relative=True, click=False)

class Logic:
    def __init__(self, cfg):
        self.cfg = cfg
        self.camera = Camera()
        self.input = InputHandler(capture_key='f8')
        self.mp_reader = MPReader(
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75
        )
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.white = (255, 255, 255)
        self.finger_tips = {'Thumb': 4,'Index': 8,'Middle': 12,'Ring': 16,'Pinky': 20}
        self.is_right_hand = deque(maxlen=30)
        self.smoothed_pos = (0, 0)
        self.old_smoothed_pos = (0, 0)
        self.smoothed_translation_pos = None
        self.smoothed_translation_pos_fist = None
        self.image_fingertip_points = []
        self.image_fist_points = []
        self.image_fist_center = None
        self.is_calibrated = False
        self.homography_matrix = None
        self.calibrated_quad_points = None
        self.last_prompt_idx = -1
        self.initial_calibration_points = None
        self.initial_translation_point = None
        self.initial_dist_from_center = None
        self.initial_angle = None
        self.initial_offset_vector = None
        self.initial_hand_span = None
        self.raw_mode = False
        self.pinky_init = False
        self.thumb_init = False
        self.raw_thumb_init = False
        self.middle_init = False
        self.raw_mode_init = False
        self.pinky_clicking = False
        self.thumb_clicking = False
        self.raw_thumb_clicking = False
        self.raw_pinky_clicking = False
        self.pinky_time = 0
        self.thumb_time = 0
        self.middle_time = 0
        self.raw_mode_time = 0
        self.finger_lowers_n = {}
        self.finger_lowers_r = {}
        self.active_buttons = set()
        self._pi = math.pi
        self._eps = 1e-9
        self.thumb_curl = 0.0
        self.pinky_curl = 0.0
        self.user32 = windll.user32
        self.dis_x, self.dis_y = zhmiscellany.misc.get_actual_screen_resolution()
        SWP_NOSIZE = 0x0001
        SWP_NOZORDER = 0x0004
        SWP_NOACTIVATE = 0x0010
        self.move_to = None
        self.last_move_to = None
        self._flags_move = SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE
        self.SW_MAXIMIZE = 3
        self.SW_RESTORE = 9
        self.MONITOR_DEFAULTTONEAREST = 2
        self._last_time = cv2.getTickCount()
        self._fps = 0.0
        self._last_pointer_time = self._last_time
        Thread(target=self.move_cursor).start()
        self.interframe_lerp = getattr(cfg, "interframe_lerp", 0.5)
        self._hand_scale_history = deque(maxlen=12)
        self._hand_peak_history = deque(maxlen=90)
        self._smoothed_estimated_radius = None
        self.prev_move_to = None
        # yes i add the ✊ emoji, i hope you like it
        self.calibration_prompts = [
            "Point to the TOP-LEFT corner 🡬 and press 'c'",
            "Point to the TOP-RIGHT corner 🡭 and press 'c'",
            "Point to the BOTTOM-RIGHT corner 🡮 and press 'c'",
            "Point to the BOTTOM-LEFT corner 🡯 and press 'c'",
            f"Point fist at TOP-LEFT corner ✊🡬 and press 'c'",
            f"Point fist at TOP-RIGHT ✊🡭 corner and press 'c'",
            f"Point fist at BOTTOM-RIGHT corner ✊🡮 and press 'c'",
            f"Point fist at BOTTOM-LEFT corner ✊🡯 and press 'c'",
            f"Point fist at CENTER ✊𖧋 and press 'c'",
        ]
    
        self.c_fing = {"Right Hand - Thumb", "Right Hand - Pinky"}
    
    @property
    def capture_key(self):
        return getattr(self.input, "capture_key", None)
    
    @capture_key.setter
    def capture_key(self, v):
        self.input.set_keybind(v)
        self.calibration_prompts = [
            f"Point to the TOP-LEFT corner 🡬 and press '{self.input.capture_key}'",
            f"Point to the TOP-RIGHT corner 🡭 and press '{self.input.capture_key}'",
            f"Point to the BOTTOM-RIGHT corner 🡮 and press '{self.input.capture_key}'",
            f"Point to the BOTTOM-LEFT corner 🡯 and press '{self.input.capture_key}'",
            f"Point fist at TOP-LEFT corner ✊🡬 and press '{self.input.capture_key}'",
            f"Point fist at TOP-RIGHT corner ✊🡭 and press '{self.input.capture_key}'",
            f"Point fist at BOTTOM-RIGHT corner ✊🡮 and press '{self.input.capture_key}'",
            f"Point fist at BOTTOM-LEFT corner ✊🡯 and press '{self.input.capture_key}'",
            f"Point fist at CENTER ✊𖧋 and press '{self.input.capture_key}'",
        ]
    
    @property
    def capture_requested(self):
        return getattr(self.input, "capture_requested", False)
    
    @capture_requested.setter
    def capture_requested(self, v):
        self.input.capture_requested = bool(v)
    
    def set_camera(self, index, desired_w=1920, desired_h=1080, warmup_frames=8, timeout=8):
        return self.camera.set_camera(index, desired_w, desired_h, warmup_frames, timeout)
    
    def get_camera_resolution(self, max_samples=8):
        return self.camera.get_camera_resolution(max_samples)
    
    def set_keybind(self, new_key):
        self.input.set_keybind(new_key)
        self.calibration_prompts = [
            f"Point to the TOP-LEFT corner 🡬 and press '{self.input.capture_key}'",
            f"Point to the TOP-RIGHT corner 🡭 and press '{self.input.capture_key}'",
            f"Point to the BOTTOM-RIGHT corner 🡮 and press '{self.input.capture_key}'",
            f"Point to the BOTTOM-LEFT corner 🡯 and press '{self.input.capture_key}'",
            f"Point fist at TOP-LEFT corner ✊🡬 and press '{self.input.capture_key}'",
            f"Point fist at TOP-RIGHT corner ✊🡭 and press '{self.input.capture_key}'",
            f"Point fist at BOTTOM-RIGHT corner ✊🡮 and press '{self.input.capture_key}'",
            f"Point fist at BOTTOM-LEFT corner ✊🡯 and press '{self.input.capture_key}'",
            f"Point fist at CENTER ✊𖧋 and press '{self.input.capture_key}'",
        ]
        
    def move_cursor(self):
        last_time = time.time()
        
        current_pos = None
        target_pos = None
        t = 1.0
        while True:
            now = time.time()
            dt = now - last_time
            last_time = now
            if self.raw_mode and self.move_to:
                if target_pos != self.move_to:
                    current_pos = self.last_move_to or self.move_to
                    target_pos = self.move_to
                    t = 0.0
                    self.last_move_to = target_pos
                
                if current_pos is not None and target_pos is not None:
                    t += dt / max(self.dt, 1e-6)
                    if t > 1.0:
                        t = 1.0
                    
                    x = current_pos[0] + (target_pos[0] - current_pos[0]) * t
                    y = current_pos[1] + (target_pos[1] - current_pos[1]) * t
                    x, y = self.move_to
                    clicked = False
                    for finger, low_val in self.finger_lowers_r.items():
                        thr = self.cfg.r_thresholds.get(finger, 1.0)
                        start = low_val >= thr
                        pressed = self.cfg.r_keybinds[finger] in self.active_buttons
                        if self.cfg.r_keybinds[finger] == "mouse:left":
                            if not pressed and start:
                                zhmiscellany.misc.click_pixel((x, y), relative=True, act_end=False)
                                self.active_buttons.add("mouse:left")
                                clicked = True
                            elif pressed and not start:
                                zhmiscellany.misc.click_pixel((x, y), relative=True, act_start=False)
                                self.active_buttons.discard("mouse:left")
                                clicked = True
                        elif self.cfg.r_keybinds[finger] == "mouse:right":
                            if not pressed and start:
                                zhmiscellany.misc.click_pixel((x, y), relative=True, act_end=False, right_click=True)
                                self.active_buttons.add("mouse:right")
                                clicked = True
                            elif pressed and not start:
                                zhmiscellany.misc.click_pixel((x, y), relative=True, act_start=False, right_click=True)
                                self.active_buttons.discard("mouse:right")
                                clicked = True
                        elif self.cfg.r_keybinds[finger] == "mouse:scroll_up":
                            if not pressed and start:
                                zhmiscellany.misc.scroll(1)
                                self.active_buttons.add("mouse:scroll_up")
                            elif pressed and not start:
                                self.active_buttons.discard("mouse:scroll_up")
                        elif self.cfg.r_keybinds[finger] == "mouse:scroll_down":
                            if not pressed and start:
                                zhmiscellany.misc.scroll(-1)
                                self.active_buttons.add("mouse:scroll_down")
                            elif pressed and not start:
                                self.active_buttons.discard("mouse:scroll_down")
                        else:
                            try:
                                key = self.cfg.r_keybinds[finger]
                                if not pressed and start:
                                    zhmiscellany.macro.press_key_directinput(key, act_end=False)
                                    self.active_buttons.add(key)
                                elif pressed and not start:
                                    zhmiscellany.macro.press_key_directinput(key, act_start=False)
                                    self.active_buttons.discard(key)
                            except:
                                pass
                    if not clicked:
                        zhmiscellany.misc.click_pixel((x, y), relative=True, act_start=False, act_end=False)
            time.sleep(0.001)
    
    def update_homograph(self, new_image_points, screen_w, screen_h):
        H = utils.update_homograph(new_image_points, screen_w, screen_h)
        if H is not None:
            self.homography_matrix = H
            self.calibrated_quad_points = new_image_points
            return True
        return False
    
    def _v(self, a, b):
        return (b.x - a.x, b.y - a.y, b.z - a.z)
    
    def _len(self, v):
        return math.hypot(v[0], v[1], v[2])
    
    def _ang(self, a, b, c):
        ab, cb = self._v(b, a), self._v(b, c)
        dot = ab[0] * cb[0] + ab[1] * cb[1] + ab[2] * cb[2]
        na, nc = self._len(ab), self._len(cb)
        if na < self._eps or nc < self._eps: return self._pi
        return math.acos(max(-1.0, min(1.0, dot / (na * nc))))
    
    def get_finger_curl(self, hand, lhand, finger_name,
                        weights=(0.18, 0.54, 0.28),
                        low_clip=0.07,
                        high_clip=0.92,
                        concave=0.48,
                        sharpen_scale=2.6,
                        proximity_boost=0.95,
                        min_prox_ratio=0.08,
                        baseline_trim=0.02):
        
        target_hand = None
        if "Right Hand" in finger_name:
            target_hand = hand
        elif "Left Hand" in finger_name:
            target_hand = lhand
        
        if target_hand is None:
            return 0.0
        
        landmark_map = {
            "Thumb": (0, 1, 2, 3, 4),
            "Index": (0, 5, 6, 7, 8),
            "Middle": (0, 9, 10, 11, 12),
            "Ring": (0, 13, 14, 15, 16),
            "Pinky": (0, 17, 18, 19, 20)
        }
        
        target_indices = None
        is_thumb = False
        
        for key, indices in landmark_map.items():
            if key in finger_name:
                target_indices = indices
                if key == "Thumb":
                    is_thumb = True
                break
        
        if target_indices is None:
            return 0.0
        
        score = self._calculate_bend_score(target_hand, target_indices, weights, concave, sharpen_scale, low_clip, high_clip)
        
        if is_thumb:
            idx_tip_lm = 8
            thumb_tip_lm = 4
            
            idx_tip = target_hand.landmark[idx_tip_lm]
            tip = target_hand.landmark[thumb_tip_lm]
            
            prox = self._len(self._v(tip, idx_tip))
            scale_ref = self._len(self._v(target_hand.landmark[0], target_hand.landmark[9])) or 1.0
            prox_ratio = prox / scale_ref
            
            if prox_ratio < min_prox_ratio:
                boost = proximity_boost * (1.0 - prox_ratio / min_prox_ratio)
                score = max(score, min(1.0, score + boost))
        
        if baseline_trim:
            score = max(0.0, min(1.0, (score - baseline_trim) / (1.0 - baseline_trim)))
        
        return score
    
    def _calculate_bend_score(self, hand, landmarks, weights, concave, sharpen_scale, low_clip, high_clip, init_margin=0.25):
        wrist, cmc, mcp, ip, tip = (hand.landmark[i] for i in landmarks)
        
        raws = [max(0.0, min(1.0, (self._pi - self._ang(wrist, cmc, mcp)) / self._pi)),
                max(0.0, min(1.0, (self._pi - self._ang(cmc, mcp, ip)) / self._pi)),
                max(0.0, min(1.0, (self._pi - self._ang(mcp, ip, tip)) / self._pi))]
        
        min_b = [max(0.0, x - init_margin) for x in raws]
        max_b = [min(1.0, x + init_margin) for x in raws]
        
        def normalize(r, lo, hi):
            span = hi - lo
            n = 0.0 if span <= 1e-5 else (r - lo) / span
            n = max(0.0, min(1.0, n))
            n = (n - low_clip) / (high_clip - low_clip)
            return max(0.0, min(1.0, n))
        
        normed = [normalize(r, min_b[i], max_b[i]) for i, r in enumerate(raws)]
        normed = [x ** concave for x in normed]
        normed = [(math.tanh((x - 0.5) * sharpen_scale) + 1.0) / 2.0 for x in normed]
        
        wsum = sum(w * n for w, n in zip(weights, normed))
        return max(0.0, min(1.0, wsum / (sum(weights) or 1.0)))
    
    def calculate_thumb_bend(self, hand,
                             weights=(0.18, 0.54, 0.28),
                             low_clip=0.07,
                             high_clip=0.92,
                             concave=0.48,
                             sharpen_scale=2.6,
                             proximity_boost=0.95,
                             min_prox_ratio=0.08,
                             baseline_trim=0.02):
        thumb_lms = (0, 1, 2, 3, 4)
        thumb_score = self._calculate_bend_score(hand, thumb_lms, weights, concave, sharpen_scale, low_clip, high_clip)
        idx_tip_lm = hand.landmark[8]
        thumb_tip_lm = hand.landmark[4]
        prox = self._len(self._v(thumb_tip_lm, idx_tip_lm))
        scale_ref = self._len(self._v(hand.landmark[0], hand.landmark[9])) or 1.0
        prox_ratio = prox / scale_ref
        
        if prox_ratio < min_prox_ratio:
            boost = proximity_boost * (1.0 - prox_ratio / min_prox_ratio)
            thumb_score = max(thumb_score, min(1.0, thumb_score + boost))
        if baseline_trim:
            thumb_score = max(0.0, min(1.0, (thumb_score - baseline_trim) / (1.0 - baseline_trim)))
            
        return thumb_score
    
    def calculate_pinky_bend(self, hand,
                             weights=(0.18, 0.54, 0.28),
                             low_clip=0.07,
                             high_clip=0.92,
                             concave=0.48,
                             sharpen_scale=2.6,
                             baseline_trim=0.02):
        pinky_lms = (0, 17, 18, 19, 20)
        pinky_score = self._calculate_bend_score(hand, pinky_lms, weights, concave, sharpen_scale, low_clip, high_clip)
        if baseline_trim:
            pinky_score = max(0.0, min(1.0, (pinky_score - baseline_trim) / (1.0 - baseline_trim)))
        
        return pinky_score
    
    def _best_align(self, src, mapv):
        src_norm = np.linalg.norm(src)
        map_norm = np.linalg.norm(mapv)
        if src_norm < 1e-6 or map_norm < 1e-6:
            return mapv.copy()
        best = None
        best_ang = float("inf")
        mx, my = mapv[0], mapv[1]
        for swap in (False, True):
            for sx in (1.0, -1.0):
                for sy in (1.0, -1.0):
                    if swap:
                        cand = np.array([sx * my, sy * mx], dtype=np.float64)
                    else:
                        cand = np.array([sx * mx, sy * my], dtype=np.float64)
                    dot = np.dot(src, cand)
                    denom = src_norm * np.linalg.norm(cand)
                    if denom <= 1e-12:
                        ang = float("inf")
                    else:
                        cos = np.clip(dot / denom, -1.0, 1.0)
                        ang = abs(np.arccos(cos))
                    if ang < best_ang:
                        best_ang = ang
                        best = cand
        return best
    
    def process_frame(self, screen_w, screen_h):
        '''this is where everything happens'''
        if not hasattr(self.camera, "cap") or self.camera.cap is None or not getattr(self.camera.cap, "isOpened", lambda: False)():
            ok = self.camera.set_camera(0)
            if not ok:
                return None, None, None, None
        
        success, image = self.camera.read_frame()
        if not success or image is None:
            return None, None, None, None
        instruction = None
        image = cv2.flip(image, 1)
        cam_h, cam_w, _ = image.shape
        line_thickness = max(1, int(min(cam_w, cam_h) * 0.005))
        point_radius = max(3, int(min(cam_w, cam_h) * 0.008))
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        raw = self.mp_reader.hands.process(image_rgb)
        
        results_left = None
        results_right = None
        
        detected_hands = []
        
        if raw.multi_hand_landmarks:
            for hand_info, hand_landmarks in zip(raw.multi_handedness, raw.multi_hand_landmarks):
                wrist_x = hand_landmarks.landmark[0].x
                
                detected_hands.append((wrist_x, hand_info, hand_landmarks))
        
        detected_hands.sort(key=lambda x: x[0])
        if len(detected_hands) == 1:
            _, hand_info, hand_landmarks = detected_hands[0]
            label = hand_info.classification[0].label
            
            dummy = type("MPResult", (), {})()
            dummy.multi_hand_landmarks = [hand_landmarks]
            dummy.multi_handedness = [hand_info]
            
            if label == "Left":
                results_left = dummy
            else:
                results_right = dummy
        
        elif len(detected_hands) == 2:
            _, hand_info_left, hand_landmarks_left = detected_hands[0]
            
            dummy_left = type("MPResult", (), {})()
            dummy_left.multi_hand_landmarks = [hand_landmarks_left]
            dummy_left.multi_handedness = [hand_info_left]
            results_left = dummy_left
            
            _, hand_info_right, hand_landmarks_right = detected_hands[1]
            
            dummy_right = type("MPResult", (), {})()
            dummy_right.multi_hand_landmarks = [hand_landmarks_right]
            dummy_right.multi_handedness = [hand_info_right]
            results_right = dummy_right
        
        r_exist = results_right is not None
        self.is_right_hand.append(r_exist)
        required_true_count = 0.4 * len(self.is_right_hand)
        
        is_right = sum(self.is_right_hand) >= required_true_count
        fingertip_pos = None
        raw_translation_point_pos = None
        current_angle = None
        current_hand_span = None
        landmark_0 = None
        landmark_5 = None
        lhand, hand = None, None
        if image is not None:
            H, W, _ = image.shape
        else:
            H, W = 480, 640
        FONT_SCALE = H / 620
        THICKNESS = max(1, int(FONT_SCALE * 3))
        if results_left is not None:
            lhand = results_left.multi_hand_landmarks[0]
            mp.solutions.drawing_utils.draw_landmarks(image, lhand, mp.solutions.hands.HAND_CONNECTIONS)
            if results_right is None and self.raw_mode:
                l_finger_lowers = {}
                for finger, key in self.cfg.r_keybinds.items():
                    if key is not None and "Right Hand" not in finger:
                        l_finger_lowers[finger] = self.get_finger_curl(hand, lhand, finger)
                self.finger_lowers_r = l_finger_lowers
                self.move_to = (0.0, 0.0)
            for key, value in self.finger_lowers_r.items():
                if not key.startswith('Left Hand'):
                    continue
                
                try:
                    finger = key.split(' - ')[1]
                    tip_idx = self.finger_tips.get(finger)
                    
                    if tip_idx is None:
                        continue
                    tip_landmark = lhand.landmark[tip_idx]
                    tip_x_pixel = int(tip_landmark.x * W)
                    tip_y_pixel = int(tip_landmark.y * H)
                    text_to_draw = f"{value:.2f}"
                    position = (tip_x_pixel + 10, tip_y_pixel)
                    cv2.putText(image,text_to_draw,position,self.font,FONT_SCALE,self.white,THICKNESS,cv2.LINE_AA)
                except Exception as e:
                    pass
        if results_right is not None and is_right:
            hand = results_right.multi_hand_landmarks[0]
            
            idx_tip = hand.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            knuckle = hand.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_PIP]
            wrist = hand.landmark[mp.solutions.hands.HandLandmark.WRIST]
            idx_mcp = hand.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_MCP]
            
            landmark_0 = np.array([hand.landmark[0].x * cam_w, hand.landmark[0].y * cam_h])
            landmark_5 = np.array([hand.landmark[5].x * cam_w, hand.landmark[5].y * cam_h])
            
            fingertip_pos = np.array([idx_tip.x * cam_w, idx_tip.y * cam_h])
            knuckle_pos = np.array([knuckle.x * cam_w, knuckle.y * cam_h])
            wrist_pos = np.array([wrist.x * cam_w, wrist.y * cam_h])
            raw_translation_point_pos = np.array([hand.landmark[0].x * cam_w, hand.landmark[0].y * cam_h])
            raw_translation_point_pos_fist = np.array([hand.landmark[9].x * cam_w, hand.landmark[9].y * cam_h])
            landmark_11_pos = np.array([hand.landmark[11].x * cam_w, hand.landmark[11].y * cam_h])
            
            angle_vector = landmark_5 - landmark_0
            current_angle = np.arctan2(angle_vector[1], angle_vector[0])
            current_hand_span = np.linalg.norm(landmark_5 - landmark_0)
            
            if self.smoothed_translation_pos is None:
                self.smoothed_translation_pos = raw_translation_point_pos
                self.smoothed_translation_pos_fist = raw_translation_point_pos_fist
            else:
                self.smoothed_translation_pos = smooth_translation(
                    self.smoothed_translation_pos, raw_translation_point_pos, self.cfg, cam_w, cam_h
                )
                self.smoothed_translation_pos_fist = smooth_translation(
                    self.smoothed_translation_pos_fist, raw_translation_point_pos, self.cfg, cam_w, cam_h
                )
            
            pinky_tip = np.array([hand.landmark[20].x * cam_w, hand.landmark[20].y * cam_h])
            index_tip_y = hand.landmark[8].y * cam_h
            
            thumb_tip = np.array([hand.landmark[4].x * cam_w, hand.landmark[4].y * cam_h])
            middle_tip = np.array([hand.landmark[12].x * cam_w, hand.landmark[12].y * cam_h])
            index_tip = np.array([hand.landmark[8].x * cam_w, hand.landmark[8].y * cam_h])
            
            all_ys = np.array([lm.y * cam_h for lm in hand.landmark])
            bbox_height = all_ys.max() - all_ys.min()
            
            wrist = hand.landmark[0]
            middle_mcp = hand.landmark[9]
            wrist_to_middle = np.linalg.norm([
                (wrist.x - middle_mcp.x) * cam_w,
                (wrist.y - middle_mcp.y) * cam_h
            ])
            
            scale = max(bbox_height, wrist_to_middle, 1.0)
            
            raw_metric_pinky = (pinky_tip[1] - index_tip_y) / scale
            raw_metric_thumb = (thumb_tip[1] - index_tip_y) / scale
            raw_metric_middle = (middle_tip[1] - index_tip_y) / scale
            idx_mcp_y = idx_mcp.y * cam_h
            raw_metric_index = (index_tip[1] - idx_mcp_y) / scale
            self.finger_lowers_n = {"Pinky": float(np.clip(raw_metric_pinky, 0.0, 1.0)), "Thumb": float(np.clip(raw_metric_thumb, 0.0, 1.0)), "Middle": float(np.clip(raw_metric_middle, 0.0, 1.0))}
            lowers = self.finger_lowers_r if self.raw_mode else self.finger_lowers_n
            for key, value in lowers.items():
                if self.raw_mode and not key.startswith('Right Hand'):
                    continue
                try:
                    if self.raw_mode:
                        key = key.split(' - ')[1]
                    tip_idx = self.finger_tips.get(key)
                    if tip_idx is None:
                        continue
                    tip_landmark = hand.landmark[tip_idx]
                    tip_x_pixel = int(tip_landmark.x * W)
                    tip_y_pixel = int(tip_landmark.y * H)
                    text_to_draw = f"{value:.2f}"
                    position = (tip_x_pixel + 10, tip_y_pixel)
                    cv2.putText(image, text_to_draw, position, self.font, FONT_SCALE, self.white, THICKNESS, cv2.LINE_AA)
                except Exception as e:
                    pass
            ct = time.time()
            current_time = cv2.getTickCount()
            self.dt = (current_time - getattr(self, "_last_time", current_time)) / cv2.getTickFrequency()
            self._last_time = current_time
            mp.solutions.drawing_utils.draw_landmarks(image, hand, mp.solutions.hands.HAND_CONNECTIONS)
            
            if self.is_calibrated:
                sum_x = 0
                sum_y = 0
                for point in self.calibrated_quad_points:
                    sum_x += point[0]
                    sum_y += point[1]
                
                center_x = sum_x / len(self.calibrated_quad_points)
                center_y = sum_y / len(self.calibrated_quad_points)
                
                lm = hand.landmark
                eps = 1e-6
                
                def pt(i):
                    return np.array([lm[i].x * cam_w, lm[i].y * cam_h])
                
                all_pts = np.array([pt(i) for i in range(len(lm))])
                dists = np.linalg.norm(all_pts[:, None, :] - all_pts[None, :, :], axis=2)
                max_hand_dim = np.max(dists) + eps
                palm_centroid = np.mean([pt(0), pt(5), pt(9), pt(13), pt(17)], axis=0)
                fingertip_pts = np.array([pt(8), pt(12), pt(16), pt(20)])
                fingertip_dists = np.linalg.norm(fingertip_pts - palm_centroid, axis=1)
                finger_to_palm_proximity_score = 1.0 - (np.mean(fingertip_dists) / max_hand_dim)
                
                def get_finger_curl_ratio(mcp, tip, pip, dip):
                    straight_dist = np.linalg.norm(pt(tip) - pt(mcp))
                    joint_dists = np.linalg.norm(pt(pip) - pt(mcp)) + np.linalg.norm(pt(dip) - pt(pip)) + np.linalg.norm(pt(tip) - pt(dip))
                    return np.clip(joint_dists / (straight_dist + eps), 1.0, 2.0) - 1.0
                
                idx_curl = get_finger_curl_ratio(5, 8, 6, 7)
                mid_curl = get_finger_curl_ratio(9, 12, 10, 11)
                ring_curl = get_finger_curl_ratio(13, 16, 14, 15)
                pink_curl = get_finger_curl_ratio(17, 20, 18, 19)
                finger_curl_score = np.mean([idx_curl, mid_curl, ring_curl, pink_curl])
                thumb_to_index_tip_dist = np.linalg.norm(pt(4) - pt(8))
                thumb_position_score = 1.0 - (thumb_to_index_tip_dist / max_hand_dim)
                bbox_w = max(all_pts[:, 0].max() - all_pts[:, 0].min(), 1.0)
                bbox_h = max(all_pts[:, 1].max() - all_pts[:, 1].min(), 1.0)
                bbox_area = bbox_w * bbox_h
                hull = cv2.convexHull(all_pts.astype(np.float32))
                hull_area = max(cv2.contourArea(hull), 0.0)
                hand_compaction_score = 1.0 - (hull_area / (bbox_area + eps))
                w_proximity = 0.40
                w_curl = 0.30
                w_thumb = 0.20
                w_compaction = 0.10
                raw_score = (
                        w_proximity * finger_to_palm_proximity_score +
                        w_curl * finger_curl_score +
                        w_thumb * thumb_position_score +
                        w_compaction * hand_compaction_score
                )
                fist_confidence = np.clip((raw_score - 0.2) / (0.9 - 0.2), 0.0, 1.0)
                threshold_met = fist_confidence >= self.cfg.raw_mode_threshold
                
                if self.raw_mode != threshold_met and not self.raw_mode_init:
                    self.raw_mode_init = True
                    self.raw_mode_time = ct
                
                if self.raw_mode_init:
                    if self.raw_mode == threshold_met:
                        self.raw_mode_init = False
                        self.raw_mode_time = False
                    elif ct >= self.raw_mode_time + self.cfg.raw_mode_switch_time:
                        self.raw_mode = not self.raw_mode
                        for button in self.active_buttons:
                            if button == "mouse:left":
                                zhmiscellany.misc.click_pixel((0, 0), relative=True, act_start=False)
                            elif button == "mouse:right":
                                zhmiscellany.misc.click_pixel((0, 0), relative=True, act_start=False)
                            else:
                                try:
                                    zhmiscellany.macro.press_key_directinput(button, act_start=False)
                                except:
                                    pass
                        self.active_buttons = set()
                        self.raw_mode_init = False
                        self.raw_mode_time = False
                sp = tuple(map(int, np.round(self.smoothed_pos)))
                
                if self.raw_mode:
                    def _to_xy(pt):
                        if hasattr(pt, "x") and hasattr(pt, "y"):
                            cw, ch = getattr(self, "cam_w", None), getattr(self, "cam_h", None)
                            sx, sy = (float(cw), float(ch)) if cw is not None and ch is not None else (1.0, 1.0)
                            return np.array([float(pt.x) * sx, float(pt.y) * sy], dtype=np.float64)
                        a = np.asarray(pt, dtype=np.float64).reshape(-1)
                        if a.size < 2:
                            raise ValueError("point-like object must have at least two elements")
                        return a[:2].astype(np.float64)
                    
                    raw_pts = [_to_xy(p) for p in self.initial_calibration_fist_points]
                    if len(raw_pts) != 4:
                        raise ValueError("initial_calibration_fist_points must contain exactly 4 points")
                    pts = np.vstack(raw_pts).astype(np.float64)
                    center_src = _to_xy(wrist_pos)
                    knuckle_src = _to_xy(knuckle_pos)
                    initial_center_src = _to_xy(self.initial_calibration_fist_center)
                    wrist_offset_raw = getattr(self, "initial_fist_wrist_dist", None)
                    if wrist_offset_raw is None:
                        orig_wrist_offset = np.array([0.0, 0.0], dtype=np.float64)
                    else:
                        orig_wrist_offset = np.asarray(wrist_offset_raw, dtype=np.float64).reshape(-1)[:2].astype(np.float64)
                    
                    try:
                        ih, iw = image.shape[:2]
                        img_w, img_h = float(iw), float(ih)
                    except Exception:
                        cw, ch = getattr(self, "cam_w", None), getattr(self, "cam_h", None)
                        if cw is not None and ch is not None:
                            img_w, img_h = float(cw), float(ch)
                        else:
                            img_w, img_h = float(self.dis_x), float(self.dis_y)
                    
                    image_center = np.array([img_w / 2.0, img_h / 2.0], dtype=np.float64)
                    
                    r_current = np.linalg.norm(center_src - image_center)
                    r_initial = np.linalg.norm(initial_center_src - image_center)
                    max_radius = np.hypot(img_w / 2.0, img_h / 2.0)
                    if max_radius <= 0:
                        max_radius = 1.0
                    
                    fov_factor = float(getattr(self.cfg, "fov_factor", 0.0))
                    scale = 1.0 + fov_factor * ((r_initial - r_current) / max_radius)
                    scale = float(np.clip(scale, 0.5, 1.5))
                    
                    new_wrist_offset = orig_wrist_offset * scale
                    wrist_offset = new_wrist_offset.astype(np.float64)
                    adjusted_center_src = center_src - wrist_offset
                    
                    dis_x, dis_y = float(self.dis_x), float(self.dis_y)
                    if dis_x <= 0 or dis_y <= 0:
                        raise ValueError("dis_x and dis_y must be positive")
                    
                    s = pts[:, 0] + pts[:, 1]
                    d = pts[:, 0] - pts[:, 1]
                    tl, br = pts[np.argmin(s)], pts[np.argmax(s)]
                    tr, bl = pts[np.argmin(d)], pts[np.argmax(d)]
                    pts_src_ordered = np.vstack([tl, tr, br, bl]).astype(np.float64)
                    
                    offset = adjusted_center_src - initial_center_src
                    moved_pts_src_ordered = (pts_src_ordered + offset).astype(np.float64)
                    
                    src_ctrl = np.vstack([pts_src_ordered, adjusted_center_src])
                    dst_ctrl = np.array([
                        [0.0, 0.0],
                        [dis_x, 0.0],
                        [dis_x, dis_y],
                        [0.0, dis_y],
                        [dis_x / 2.0, dis_y / 2.0],
                    ], dtype=np.float64)
                    
                    def _tps_fit(src, dst, reg=0.0):
                        n = src.shape[0]
                        d2 = np.sum((src[:, None, :] - src[None, :, :]) ** 2, axis=2)
                        K = d2 * np.log(d2 + 1e-20)
                        P = np.hstack([np.ones((n, 1)), src])
                        L = np.zeros((n + 3, n + 3), dtype=np.float64)
                        L[:n, :n] = K + reg * np.eye(n)
                        L[:n, n:] = P
                        L[n:, :n] = P.T
                        vx = np.concatenate([dst[:, 0], np.zeros(3)], axis=0)
                        vy = np.concatenate([dst[:, 1], np.zeros(3)], axis=0)
                        px = np.linalg.solve(L, vx)
                        py = np.linalg.solve(L, vy)
                        return {"src": src, "w_x": px[:n], "a_x": px[n:], "w_y": py[:n], "a_y": py[n:]}
                    
                    def _tps_transform(params, points):
                        src = params["src"]
                        pts = np.atleast_2d(points).astype(np.float64)
                        dif = pts[:, None, :] - src[None, :, :]
                        r2 = np.sum(dif ** 2, axis=2)
                        U = r2 * np.log(r2 + 1e-20)
                        ax, ay = params["a_x"], params["a_y"]
                        mx = ax[0] + ax[1] * pts[:, 0] + ax[2] * pts[:, 1] + U.dot(params["w_x"])
                        my = ay[0] + ay[1] * pts[:, 0] + ay[2] * pts[:, 1] + U.dot(params["w_y"])
                        mapped = np.vstack([mx, my]).T
                        return mapped[0] if mapped.shape[0] == 1 else mapped
                    
                    tps = _tps_fit(src_ctrl, dst_ctrl, reg=0.001)
                    
                    mapped_center = np.atleast_1d(_tps_transform(tps, adjusted_center_src)).astype(np.float64)
                    mapped_knuckle = np.atleast_1d(_tps_transform(tps, knuckle_src)).astype(np.float64)
                    mapped_dx = mapped_knuckle[0] - mapped_center[0]
                    mapped_dy = mapped_knuckle[1] - mapped_center[1]
                    src_dx = knuckle_src[0] - adjusted_center_src[0]
                    src_dy = knuckle_src[1] - adjusted_center_src[1]
                    
                    src_vec = np.array([src_dx, src_dy], dtype=np.float64)
                    map_vec = np.array([mapped_dx, mapped_dy], dtype=np.float64)
                    
                    
                    
                    aligned_map = self._best_align(src_vec, map_vec)
                    x_cand, y_cand = float(aligned_map[0]), float(aligned_map[1])
                    
                    hx, hy = dis_x / 2.0, dis_y / 2.0
                    x = float(np.clip(x_cand, -hx, hx))
                    y = float(np.clip(y_cand, -hy, hy))
                    
                    deadzone_ratio = float(getattr(self.cfg, "raw_mode_deadzone", 0.0))
                    if deadzone_ratio > 0.0:
                        cam_w = float(getattr(self, "cam_w", 1.0))
                        cam_h = float(getattr(self, "cam_h", 1.0))
                        threshold_x = deadzone_ratio * dis_x * (cam_w / max(cam_w, cam_h))
                        threshold_y = deadzone_ratio * dis_y * (cam_h / max(cam_w, cam_h))
                        
                        if threshold_x > 1e-6 and threshold_y > 1e-6:
                            if (x / threshold_x) ** 2 + (y / threshold_y) ** 2 < 1.0:
                                x = 0.0
                                y = 0.0
                        elif np.hypot(x, y) < deadzone_ratio * max(dis_x, dis_y):
                            x = 0.0
                            y = 0.0
                    
                    x *= float(getattr(self.cfg, "raw_mode_sensitivity_x", 1.0))
                    y *= float(getattr(self.cfg, "raw_mode_sensitivity_y", 1.0))
                    
                    l_finger_lowers = {}
                    thumb_c = self.calculate_thumb_bend(hand)
                    pinky_c = self.calculate_pinky_bend(hand)
                    l_finger_lowers["Right Hand - Thumb"] = thumb_c
                    l_finger_lowers["Right Hand - Pinky"] = pinky_c
                    for finger, key in self.cfg.r_keybinds.items():
                        if key is not None and finger not in self.c_fing:
                            l_finger_lowers[finger] = self.get_finger_curl(hand, lhand, finger)
                    self.finger_lowers_r = l_finger_lowers
                    self.move_to = (round(x, 2), round(y, 2))
                    
                    try:
                        image = draw.draw_overlay(image, moved_pts_src_ordered)
                    except Exception:
                        try:
                            moved_int = np.round(moved_pts_src_ordered).astype(np.int32).reshape((-1, 1, 2))
                            cv2.polylines(image, [moved_int], isClosed=True, color=(0, 0, 255), thickness=2)
                        except Exception:
                            pass
                    
                    try:
                        cx, cy = int(round(adjusted_center_src[0])), int(round(adjusted_center_src[1]))
                        cv2.circle(image, (cx, cy), 6, (0, 255, 0), -1)
                    except Exception:
                        pass
                    
                else:
                    clicked = False
                    for finger, thresh in self.cfg.n_thresholds.items():
                        if self.finger_lowers_n[finger] >= thresh:
                            start = True
                        else:
                            start = False
                        pressed = self.cfg.n_keybinds[finger] in self.active_buttons
                        match self.cfg.n_keybinds[finger]:
                            case "mouse:left":
                                if not pressed and start:
                                    zhmiscellany.misc.click_pixel(sp, act_end=False)
                                    self.active_buttons.add("mouse:left")
                                    clicked = True
                                elif pressed and not start:
                                    zhmiscellany.misc.click_pixel(sp, act_start=False)
                                    self.active_buttons.discard("mouse:left")
                                    clicked = True
                            case "mouse:right":
                                if not pressed and start:
                                    zhmiscellany.misc.click_pixel(sp, act_end=False, right_click=True)
                                    self.active_buttons.add("mouse:right")
                                    clicked = True
                                elif pressed and not start:
                                    zhmiscellany.misc.click_pixel(sp, act_start=False, right_click=True)
                                    self.active_buttons.discard("mouse:right")
                                    clicked = True
                            case "mouse:middle":
                                if start:
                                    if not getattr(self, "middle_setup_done", False):
                                        utils.do_setup_for_middle(self, sp)
                                    if getattr(self, "middle_hwnd", None) and self.middle_setup_done:
                                        new_left, new_top, target_x, target_y = utils.compute_new_pos(self, sp)
                                        w, h = self._middle_cached_size
                                        if new_top <= self._middle_snap_threshold:
                                            self._will_maximize = True
                                        else:
                                            self._will_maximize = False
                                        self.user32.SetWindowPos(self.middle_hwnd, 0, new_left, new_top, 0, 0, self._flags_move)
                                else:
                                    if getattr(self, "middle_hwnd", None):
                                        new_left, new_top, target_x, target_y = utils.compute_new_pos(self, sp)
                                        w, h = self._middle_cached_size
                                        if getattr(self, "_will_maximize", False) or new_top <= self._middle_snap_threshold:
                                            self.user32.ShowWindow(self.middle_hwnd, self.SW_MAXIMIZE)
                                        else:
                                            self.user32.SetWindowPos(self.middle_hwnd, 0, new_left, new_top, 0, 0, self._flags_move)
                                    utils.reset_middle_state(self)
                        try:
                            key = self.cfg.n_keybinds[finger]
                            if not pressed and start:
                                zhmiscellany.macro.press_key_directinput(key, act_end=False)
                                self.active_buttons.add(key)
                            elif pressed and not start:
                                zhmiscellany.macro.press_key_directinput(key, act_start=False)
                                self.active_buttons.discard(key)
                        except:
                            pass
                    if not clicked:
                        zhmiscellany.misc.click_pixel(sp, act_start=False, act_end=False)

                if not self.raw_mode:
                    cv2.circle(image, (int(fingertip_pos[0]), int(fingertip_pos[1])), point_radius, (0, 255, 0), -1)
                    cv2.circle(image, (int(raw_translation_point_pos[0]), int(raw_translation_point_pos[1])), point_radius, (255, 0, 0), -1)
                    
                    image = draw.draw_overlay(image, self.calibrated_quad_points)
                else:
                    cv2.circle(image, (int(knuckle_pos[0]), int(knuckle_pos[1])), point_radius, (0, 255, 255), -1)
            if self.input.capture_requested:
                if self.is_calibrated:
                    self.reset_calibration()
                    instruction = "Calibration reset, press the keybind to start again"
                    self.input.capture_requested = False
                else:
                    if fingertip_pos is not None and knuckle_pos is not None and self.smoothed_translation_pos is not None:
                        if len(self.image_fingertip_points) != 4:
                            self.image_fingertip_points.append(fingertip_pos)
                        elif len(self.image_fist_points) != 4:
                            self.image_fist_points.append(knuckle_pos)
                        elif self.image_fist_center is None:
                            self.image_fist_center = knuckle_pos
                            self.image_wrist = wrist_pos
                            ok = self.update_homograph(np.array(self.image_fingertip_points, dtype=np.float32), screen_w, screen_h)
                            if ok:
                                self.is_calibrated = True
                                self.initial_calibration_points = np.array(self.image_fingertip_points, dtype=np.float32)
                                self.initial_calibration_fist_points = np.array(self.image_fist_points, dtype=np.float32)
                                self.initial_translation_point = self.smoothed_translation_pos
                                self.initial_translation_point_fist = self.smoothed_translation_pos_fist
                                self.initial_dist_from_center = np.linalg.norm(
                                    self.initial_translation_point - np.array([cam_w / 2, cam_h / 2])
                                )
                                self.initial_dist_from_center_fist = np.linalg.norm(
                                    self.initial_translation_point - np.array([cam_w / 2, cam_h / 2])
                                )
                                if landmark_0 is not None and landmark_5 is not None:
                                    angle_vector = landmark_5 - landmark_0
                                    self.initial_angle = np.arctan2(angle_vector[1], angle_vector[0])
                                    self.initial_hand_span = np.linalg.norm(landmark_5 - landmark_0)
                                center_of_quad = np.mean(self.initial_calibration_points, axis=0)
                                self.initial_offset_vector = self.initial_translation_point - center_of_quad
                                center_of_quad_fist = np.mean(self.initial_calibration_fist_points, axis=0)
                                self.initial_fist_offset_vector_fist = self.initial_translation_point_fist - center_of_quad_fist
                                pts = np.asarray(self.initial_calibration_fist_points, dtype=np.float32)
                                self.initial_calibration_fist_center = np.asarray(self.image_fist_center, dtype=np.float32)
                                
                                self.initial_fist_wrist_dist = np.asarray(self.image_wrist, dtype=np.float32) - self.initial_calibration_fist_center
                                instruction = "Calibration worked, started tracking"
                                print("Calibration worked, started tracking")
                            else:
                                instruction = "Calibration failed tf"
                                print("Calibration failed tf")
                                self.image_fingertip_points.clear()
                                self.last_prompt_idx = -1
                        self.input.capture_requested = False
            
            if not self.is_calibrated:
                if len(self.image_fingertip_points) < 4:
                    draw.draw_calibration_preview(image, self.image_fingertip_points, fingertip_pos, point_radius, line_thickness)
                elif len(self.image_fingertip_points) == 4:
                    cv2.polylines(image, [np.int32(self.image_fingertip_points + [self.image_fingertip_points[0]])], isClosed=True, color=(255, 255, 255), thickness=line_thickness)
                    if len(self.image_fist_points) < 4:
                        draw.draw_calibration_preview(image, self.image_fist_points, knuckle_pos, point_radius, line_thickness)
                    elif len(self.image_fist_points) == 4:
                        cv2.polylines(image, [np.int32(self.image_fist_points + [self.image_fist_points[0]])], isClosed=True, color=(255, 255, 255), thickness=line_thickness)
                if self.image_fist_center is None and len(self.image_fingertip_points) + len(self.image_fist_points) == 8:
                    prompt_idx = 8
                else:
                    prompt_idx = len(self.image_fingertip_points) + len(self.image_fist_points)
                if prompt_idx < len(self.calibration_prompts) and prompt_idx != self.last_prompt_idx:
                    instruction = f"{self.calibration_prompts[prompt_idx]} {prompt_idx + 1}/{len(self.calibration_prompts)}"
                    print(f"\n--- CALIBRATION STEP {prompt_idx + 1}/{len(self.calibration_prompts)} ---")
                    print(self.calibration_prompts[prompt_idx])
                    self.last_prompt_idx = prompt_idx
            else:
                if (self.initial_calibration_points is not None
                        and self.smoothed_translation_pos is not None
                        and self.initial_translation_point is not None
                        and self.initial_angle is not None
                        and current_angle is not None
                        and self.initial_hand_span is not None
                        and current_hand_span is not None):
                    scaling_factor = utils.get_scaling_factor(self.initial_dist_from_center, self.smoothed_translation_pos, cam_w, cam_h, self.cfg)
                    rotation_angle = utils.normalize_angle_diff(current_angle, self.initial_angle) * self.cfg.rotation_factor
                    cos_theta = np.cos(rotation_angle)
                    sin_theta = np.sin(rotation_angle)
                    rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
                    hand_span_ratio = current_hand_span / self.initial_hand_span
                    span_translation_factor = 0.5 * (1.0 - hand_span_ratio)
                    translation_vector_from_span = self.smoothed_translation_pos - self.initial_translation_point
                    adjusted_translation_vector = translation_vector_from_span * span_translation_factor
                    center_of_quad = np.mean(self.initial_calibration_points, axis=0)
                    scaled_points = (self.initial_calibration_points - center_of_quad) * scaling_factor
                    rotated_points = np.dot(scaled_points, rotation_matrix.T)
                    rotated_offset_vector = np.dot(self.initial_offset_vector, rotation_matrix.T)
                    new_quad_center = self.smoothed_translation_pos - rotated_offset_vector + adjusted_translation_vector
                    new_image_points = rotated_points + new_quad_center
                    
                    self.standard_quad_points = new_image_points
                    
                    if not self.raw_mode:
                        self.update_homograph(new_image_points, screen_w, screen_h)
                
                if self.calibrated_quad_points is not None:
                    if fingertip_pos is not None and self.homography_matrix is not None:
                        fingertip_np = np.array([[fingertip_pos]], dtype=np.float32)
                        screen_pos_transformed = cv2.perspectiveTransform(fingertip_np, self.homography_matrix)
                        if screen_pos_transformed is not None:
                            sx, sy = screen_pos_transformed[0, 0]
                            now = cv2.getTickCount()
                            dt_pointer = (now - getattr(self, "_last_pointer_time", self._last_time)) / cv2.getTickFrequency()
                            self._last_pointer_time = now
                            
                            pinky_strength = float(getattr(self, "pinky_lower", 0.0))
                            offset_deg = float(getattr(self.cfg, "click_offset_deg", 180.0))
                            offset_amount = float(getattr(self.cfg, "click_offset_amount", 0.01))
                            rad = math.radians(offset_deg)
                            dir_vec = np.array([math.cos(rad), -math.sin(rad)], dtype=np.float32)
                            offset_pixels = offset_amount * np.array([screen_w, screen_h], dtype=np.float32) * dir_vec * pinky_strength
                            
                            sx_off = sx + float(offset_pixels[0])
                            sy_off = sy + float(offset_pixels[1])
                            
                            prev_smoothed = None if self.smoothed_pos is None else np.array(self.smoothed_pos, dtype=np.float32)
                            new_smoothed = smooth_pointer_time(
                                prev_smoothed,
                                sx_off, sy_off,
                                self.cfg,
                                screen_w, screen_h,
                                dt_pointer,
                                base_step_hz=getattr(self.cfg, "pointer_time_hz", 240.0),
                                max_substeps=getattr(self.cfg, "pointer_max_substeps", 12),
                                use_avg=getattr(self.cfg, "pointer_use_avg", True),
                                avg_samples=getattr(self.cfg, "pointer_avg_samples", 21)
                            )
                            self.smoothed_pos = np.array(new_smoothed, dtype=np.float32)
                            
                            if prev_smoothed is None:
                                display_pos = self.smoothed_pos
                            else:
                                frac = float(getattr(self.cfg, "interframe_lerp", getattr(self, "interframe_lerp", 0.5)))
                                frac = np.clip(frac, 0.0, 1.0)
                                display_pos = prev_smoothed + frac * (self.smoothed_pos - prev_smoothed)
                            
                            pointer = np.array(display_pos, dtype=np.float32)
                            return image, zhmiscellany.misc.get_mouse_xy(), self.calibrated_quad_points, instruction
        if results_right is None and results_left is None:
            self.move_to = (0.0, 0.0)
        return image, None, self.calibrated_quad_points, instruction
    
    def reset_calibration(self):
        self.is_calibrated = False
        self.calibrated_quad_points = None
        self.homography_matrix = None
        self.image_fingertip_points = []
        self.image_fist_points = []
        self.image_fist_center = None
        self.last_prompt_idx = -1
        self.initial_calibration_points = None
        self.initial_translation_point = None
        self.initial_dist_from_center = None
        self.initial_angle = None
        self.initial_offset_vector = None
        self.initial_offset_vector_fist = None
        self.initial_hand_span = None
        self.smoothed_pos = None
        self.move_to = (0.0, 0.0)
        print("Calibration reset, press the keybind to start again")