import numpy as np
import cv2, win32gui, zhmiscellany
import ctypes
from ctypes import wintypes, byref
import utils as _utils_module


def update_homograph(new_image_points, screen_w, screen_h):
    screen_corners = np.array([
        [0, 0],
        [screen_w - 1, 0],
        [screen_w - 1, screen_h - 1],
        [0, screen_h - 1]
    ], dtype=np.float32)
    H, _ = cv2.findHomography(new_image_points, screen_corners)
    return H

def get_scaling_factor(initial_dist_from_center, current_pos, cam_w, cam_h, cfg):
    eps = 1e-6
    center = np.array([cam_w / 2.0, cam_h / 2.0])
    current_dist = np.linalg.norm(current_pos - center)
    if initial_dist_from_center is None or initial_dist_from_center <= eps:
        return 1.0
    current_dist = max(current_dist, eps)
    ratio = float(initial_dist_from_center) / current_dist
    fov_factor = float(getattr(cfg, "fov_factor", 1.0))
    scale = ratio ** fov_factor
    min_scale = float(getattr(cfg, "min_fov_scale", 0.25))
    max_scale = float(getattr(cfg, "max_fov_scale", 4.0))
    return float(np.clip(scale, min_scale, max_scale))

def normalize_angle_diff(a, b):
    diff = a - b
    return np.arctan2(np.sin(diff), np.cos(diff))

def get_focused_window_info():
    hwnd = win32gui.GetForegroundWindow()
    if not hwnd: return None, None
    try:
        rect = win32gui.GetWindowRect(hwnd)
    except Exception as e:
        print(f'Error in get_focused_window_info: {e}')
        return (0, 0), zhmiscellany.misc.get_actual_screen_resolution()
    position = (rect[0], rect[1])
    size = (rect[2] - rect[0], rect[3] - rect[1])
    return position, size


class MONITORINFOEX(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("rcMonitor", wintypes.RECT),
        ("rcWork", wintypes.RECT),
        ("dwFlags", wintypes.DWORD),
        ("szDevice", ctypes.c_wchar * 32),
    ]


class WINDOWPLACEMENT(ctypes.Structure):
    _fields_ = [
        ("length", wintypes.UINT),
        ("flags", wintypes.UINT),
        ("showCmd", wintypes.UINT),
        ("ptMinPosition", wintypes.POINT),
        ("ptMaxPosition", wintypes.POINT),
        ("rcNormalPosition", wintypes.RECT),
    ]






def compute_new_pos(obj, sp):
    dx = sp[0] - obj.middle_sp_at_press[0]
    dy = sp[1] - obj.middle_sp_at_press[1]
    target_x = int(obj.middle_anchor[0] + dx)
    target_y = int(obj.middle_anchor[1] + dy)
    w, h = obj._middle_cached_size
    new_left = target_x - (w // 2)
    new_top = target_y - int(h * 0.01)
    return new_left, new_top, target_x, target_y

def reset_middle_state(obj):
    obj.middle_init = False
    obj.middle_anchor = None
    obj.middle_sp_at_press = None
    obj.middle_hwnd = None
    obj._middle_cached_size = None
    obj._middle_snap_threshold = None
    obj._will_maximize = False
    obj._restored_size = None
    obj.middle_setup_done = False

def do_setup_for_middle(obj, sp):
    hwnd = obj.user32.GetForegroundWindow()
    obj.middle_hwnd = int(hwnd) if hwnd else None
    if not obj.middle_hwnd:
        obj._will_maximize = False
        obj._middle_cached_size = None
        obj._middle_snap_threshold = None
        return
    was_maximized = bool(obj.user32.IsZoomed(obj.middle_hwnd))
    rect = wintypes.RECT()
    obj.user32.GetWindowRect(obj.middle_hwnd, byref(rect))
    w = rect.right - rect.left
    h = rect.bottom - rect.top
    if was_maximized:
        wp = _utils_module.WINDOWPLACEMENT() if hasattr(_utils_module, "WINDOWPLACEMENT") else None
        if wp is not None:
            wp.length = ctypes.sizeof(wp)
            obj.user32.GetWindowPlacement(obj.middle_hwnd, byref(wp))
            nr = wp.rcNormalPosition
            nw = (nr.right - nr.left) or w
            nh = (nr.bottom - nr.top) or h
            w, h = int(nw), int(nh)
            obj._restored_size = (w, h)
            new_left = sp[0] - (w // 2)
            new_top = sp[1] - int(h * 0.01)
            obj.user32.ShowWindow(obj.middle_hwnd, obj.SW_RESTORE)
            obj.user32.SetWindowPos(obj.middle_hwnd, 0, new_left, new_top, 0, 0, obj._flags_move)
    obj._middle_cached_size = (w, h)
    mi = _utils_module.MONITORINFOEX()
    mi.cbSize = ctypes.sizeof(mi)
    obj.user32.GetMonitorInfoW(obj.user32.MonitorFromWindow(obj.middle_hwnd, obj.MONITOR_DEFAULTTONEAREST), byref(mi))
    mon_top = mi.rcWork.top
    obj._middle_snap_threshold = mon_top + 10
    win_pos, win_dim = _utils_module.get_focused_window_info()
    anchor_x = int(win_pos[0] + win_dim[0] / 2)
    anchor_y = int(win_pos[1] + win_dim[1] * 0.01)
    obj.middle_anchor = (anchor_x, anchor_y)
    obj.middle_sp_at_press = sp
    obj._will_maximize = False
    obj.middle_setup_done = True
