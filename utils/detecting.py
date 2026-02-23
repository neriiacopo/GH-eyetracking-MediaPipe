import cv2
import numpy as np


# =============================
# Landmark indices
# =============================
IDX_CHIN = 152
IDX_FOREHEAD = 10
IDX_LEFT_CHEEK = 234
IDX_RIGHT_CHEEK = 454
IDX_NOSE_TIP = 1

IDX_L_EYE_OUTER = 33
IDX_L_EYE_INNER = 133
IDX_R_EYE_INNER = 362
IDX_R_EYE_OUTER = 263

LEFT_IRIS_IDXS = [468, 469, 470, 471, 472]
RIGHT_IRIS_IDXS = [473, 474, 475, 476, 477]


def _lm_to_np(lm):
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)


def _safe_norm(v, eps=1e-8):
    n = float(np.linalg.norm(v))
    if n < eps:
        return v * 0.0
    return v / n


def _to_pixel_xy(lm, w, h):
    return np.array([lm.x * w, lm.y * h], dtype=np.float32)


def _estimate_depth_from_eyes_px(face_landmarks, w, h, eps=1e-6):
    l_outer = face_landmarks[IDX_L_EYE_OUTER]
    l_inner = face_landmarks[IDX_L_EYE_INNER]
    r_inner = face_landmarks[IDX_R_EYE_INNER]
    r_outer = face_landmarks[IDX_R_EYE_OUTER]

    l_center = 0.5 * (_to_pixel_xy(l_outer, w, h) + _to_pixel_xy(l_inner, w, h))
    r_center = 0.5 * (_to_pixel_xy(r_outer, w, h) + _to_pixel_xy(r_inner, w, h))

    eye_px = float(np.linalg.norm(r_center - l_center))
    depth_rel = 1.0 / max(eye_px, eps)
    return eye_px, depth_rel


def _orthonormal_frame_from_landmarks(face_landmarks):
    lms = face_landmarks

    chin = _lm_to_np(lms[IDX_CHIN])
    forehead = _lm_to_np(lms[IDX_FOREHEAD])
    left_cheek = _lm_to_np(lms[IDX_LEFT_CHEEK])
    right_cheek = _lm_to_np(lms[IDX_RIGHT_CHEEK])
    nose = _lm_to_np(lms[IDX_NOSE_TIP])

    origin = 0.25 * (chin + forehead + left_cheek + right_cheek)

    y_raw = forehead - chin
    x_raw = right_cheek - left_cheek

    Y = _safe_norm(y_raw)
    X = _safe_norm(x_raw)

    Z = _safe_norm(np.cross(X, Y))

    nose_dir = nose - origin
    if np.dot(Z, nose_dir) < 0:
        Z = -Z

    X = _safe_norm(np.cross(Y, Z))
    return origin, X, Y, Z


def _iris_center(face_landmarks, iris_indices):
    pts = np.stack([_lm_to_np(face_landmarks[i]) for i in iris_indices], axis=0)
    return pts.mean(axis=0)


def _eye_center_from_corners(face_landmarks, idx_outer, idx_inner):
    outer = _lm_to_np(face_landmarks[idx_outer])
    inner = _lm_to_np(face_landmarks[idx_inner])
    return 0.5 * (outer + inner)


def _estimate_gaze_in_head_frame(face_landmarks, X, Y, Z, gain_xy=2.5, z_bias=1.0):
    l_iris = _iris_center(face_landmarks, LEFT_IRIS_IDXS)
    r_iris = _iris_center(face_landmarks, RIGHT_IRIS_IDXS)

    l_eye_c = _eye_center_from_corners(face_landmarks, IDX_L_EYE_OUTER, IDX_L_EYE_INNER)
    r_eye_c = _eye_center_from_corners(face_landmarks, IDX_R_EYE_OUTER, IDX_R_EYE_INNER)

    l_off = l_iris - l_eye_c
    r_off = r_iris - r_eye_c
    off = 0.5 * (l_off + r_off)

    dx = float(np.dot(off, X))
    dy = float(np.dot(off, Y))

    gaze = Z * z_bias + X * (gain_xy * dx) + Y * (gain_xy * dy)
    gaze = _safe_norm(gaze)
    return gaze, dx, dy