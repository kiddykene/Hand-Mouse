import numpy as np
import math

def _compute_alpha(speed, max_dist, cfg):
    threshold = cfg.FAST_MOVEMENT_THRESHOLD * max_dist
    t = np.clip(speed / (threshold + 1e-6), 0.0, 1.0)
    eased_t = t * t * (3 - 2 * t)
    return cfg.SMOOTH_ALPHA * (1 - eased_t) + cfg.FAST_MOVEMENT_ALPHA * eased_t

def smooth_translation(prev, raw, cfg, cam_w, cam_h):
    if prev is None:
        return raw
    speed = np.linalg.norm(raw - prev)
    max_dist = math.hypot(cam_w / 2, cam_h / 2)
    alpha = _compute_alpha(speed, max_dist, cfg)
    return prev + alpha * (raw - prev)

def smooth_pointer(prev, sx, sy, cfg, screen_w, screen_h):
    raw = np.array([sx, sy], dtype=np.float32)
    if prev is None:
        return raw
    prev = np.array(prev, dtype=np.float32)
    speed = np.linalg.norm(raw - prev)
    max_dist = math.hypot(screen_w / 2, screen_h / 2)
    alpha = _compute_alpha(speed, max_dist, cfg)
    return prev + alpha * (raw - prev)

def smooth_pointer_avg(prev, sx, sy, cfg, screen_w, screen_h, samples=11):
    raw = np.array([sx, sy], dtype=np.float32)
    if prev is None:
        return raw
    prev = np.array(prev, dtype=np.float32)
    max_dist = math.hypot(screen_w / 2, screen_h / 2)

    dif = raw - prev
    fracs = np.linspace(0.0, 1.0, samples)[:, None]
    intermediates = prev + fracs * dif
    speeds = np.linalg.norm(intermediates - prev, axis=1)
    alphas = np.array([_compute_alpha(s, max_dist, cfg) for s in speeds])[:, None]
    smoothed_samples = prev + alphas * (intermediates - prev)
    out = smoothed_samples.mean(axis=0)
    return np.array(out, dtype=np.float32)

def smooth_pointer_time(prev, sx, sy, cfg, screen_w, screen_h, dt, base_step_hz=120.0, max_substeps=8, use_avg=False, avg_samples=11):
    raw = np.array([sx, sy], dtype=np.float32)
    if prev is None:
        return raw
    prev = np.array(prev, dtype=np.float32)
    if dt <= 0:
        if use_avg:
            return smooth_pointer_avg(prev, sx, sy, cfg, screen_w, screen_h, samples=avg_samples)
        return smooth_pointer(prev, sx, sy, cfg, screen_w, screen_h)

    base_step = 1.0 / float(base_step_hz)
    steps = min(max_substeps, max(1, int(math.ceil(dt / base_step))))
    cur = prev.copy()
    for i in range(steps):
        frac = (i + 1) / steps
        intermediate_target = prev + frac * (raw - prev)
        if use_avg:
            cur = smooth_pointer_avg(cur, intermediate_target[0], intermediate_target[1], cfg, screen_w, screen_h, samples=avg_samples)
        else:
            cur = smooth_pointer(cur, intermediate_target[0], intermediate_target[1], cfg, screen_w, screen_h)
    return cur
