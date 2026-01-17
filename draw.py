import cv2
import numpy as np
import zhmiscellany

def draw_calibration_preview(image, image_fingertip_points, fingertip_pos, point_radius, line_thickness):
    if len(image_fingertip_points) > 0:
        pts = [tuple(map(int, p)) for p in image_fingertip_points]
        for p in pts:
            cv2.circle(image, p, point_radius, (255, 255, 255), -1)
        for i in range(len(pts) - 1):
            cv2.line(image, pts[i], pts[i + 1], (255, 255, 255), line_thickness, lineType=cv2.LINE_AA)
        if fingertip_pos is not None and len(pts) > 0:
            last = pts[-1]
            cur = (int(fingertip_pos[0]), int(fingertip_pos[1]))
            cv2.line(image, last, cur, (255, 255, 255), line_thickness, lineType=cv2.LINE_AA)

def draw_overlay(image, calibrated_quad_points, anchor_point=None):
    if calibrated_quad_points is None:
        return image
    quad = calibrated_quad_points.astype(np.float32).copy()
    if anchor_point is not None:
        anchor = np.array(anchor_point, dtype=np.float32)
        quad_center = quad.mean(axis=0)
        quad = quad + (anchor - quad_center)
    quad_int = quad.astype(np.int32)
    overlay_fill = image.copy()
    cv2.fillPoly(overlay_fill, [quad_int], (230, 216, 173))
    image = cv2.addWeighted(overlay_fill, 0.3, image, 0.7, 0)
    cam_h, cam_w = image.shape[:2]
    thickness = max(1, int(min(cam_w, cam_h) * 0.005))
    cv2.polylines(image, [quad_int], True, (255, 255, 255), thickness)
    return image
