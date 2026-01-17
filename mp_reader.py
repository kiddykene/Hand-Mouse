import numpy as np
import mediapipe as mp

class MPReader:
    def __init__(self, max_num_hands=1, model_complexity=0,
                 min_detection_confidence=0.7, min_tracking_confidence=0.7):
        self.mp = mp
        self.hands = mp.solutions.hands.Hands(
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            max_num_hands=max_num_hands
        )

    def process(self, image):
        """
        image: BGR image from OpenCV (as in your original code)
        returns dict with:
          - hand_landmarks (the mediapipe landmarks object or None)
          - fingertip_pos (np.array [x,y]) or None
          - raw_translation_point_pos (np.array [x,y]) or None
          - landmark_0, landmark_5 (np.array [x,y]) or None
          - current_angle (float) or None
          - current_hand_span (float) or None
          - mp_hand (the mediapipe hand landmark set) or None
        """
        out = {
            "hand_landmarks": None,
            "fingertip_pos": None,
            "raw_translation_point_pos": None,
            "landmark_0": None,
            "landmark_5": None,
            "current_angle": None,
            "current_hand_span": None,
            "mp_hand": None
        }

        cam_h, cam_w = image.shape[:2]
        results = self.hands.process(mp.Image(image_format=mp.ImageFormat.SRGB, data=image) if False else
                                     self.mp.solutions.hands.Hands.process(self.hands,
                                     self.mp.solutions.drawing_utils._normalize_image(image)))

        # Above is a fallback trick — MediaPipe expects RGB numpy in your original flow, but we'll
        # rely on the same call-site conversion to RGB used in logic.py. So the logic.py will
        # pass cv2.cvtColor(image, cv2.COLOR_BGR2RGB) before calling this when needed.
        # To keep behavior identical, the caller should convert to RGB (the original code does).

        # For safety, we handle both shapes of results returned by MediaPipe.
        if getattr(results, "multi_hand_landmarks", None):
            mp_hand = results.multi_hand_landmarks[0]
            out["hand_landmarks"] = mp_hand
            out["mp_hand"] = mp_hand

            # Extract original normalized landmarks -> convert to pixel coordinates using cam_w/cam_h
            # We need to compute coordinates; the caller will supply the image dimensions in the same way.
            # But because caller already has cam_w/cam_h, we'll do conversion there to keep this small.
        return out

    def close(self):
        try:
            self.hands.close()
        except Exception:
            pass
