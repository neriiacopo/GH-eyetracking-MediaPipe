import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# =============================
# Viz setup
# =============================
AXIS_LEN_PX = 120           
GAZE_LEN_PX = 160          
DEPTH_MIN = 5            
DEPTH_MAX = 10          

MANROPE_TTF_PATH = "Manrope-Regular.ttf"
FONT_SIZE = 16             


def _clamp01(t):
    return 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)

def _depth_to_bgr(depth_value, dmin=DEPTH_MIN, dmax=DEPTH_MAX):
    """
    Map depth -> color from RED (near) to BLUE (far).
    OpenCV uses BGR.
    """
    t = (depth_value - dmin) / max(dmax - dmin, 1e-6)
    t = _clamp01(t)

    # near = red (0,0,255), far = blue (255,0,0)
    b = int(255 * t)
    g = 0
    r = int(255 * (1.0 - t))
    return (b, g, r)


def _draw_arrow(img_bgr, p0, p1, color_bgr, thickness=2, tip=0.25):
    p0 = (int(p0[0]), int(p0[1]))
    p1 = (int(p1[0]), int(p1[1]))
    cv2.arrowedLine(img_bgr, p0, p1, color_bgr, thickness, line_type=cv2.LINE_AA, tipLength=tip)


def _project_axis_2d(origin_lm_xyz, axis_xyz, w, h, length_px):
    """
    Simple 2D overlay projection:
      - anchor point uses origin landmark x,y
      - direction uses axis x,y components
    """
    ox, oy = origin_lm_xyz[0] * w, origin_lm_xyz[1] * h
    dir2 = np.array([axis_xyz[0], axis_xyz[1]], dtype=np.float32)
    n = np.linalg.norm(dir2)
    if n < 1e-8:
        dir2 = np.array([1.0, 0.0], dtype=np.float32)
        n = 1.0
    dir2 = dir2 / n

    p0 = np.array([ox, oy], dtype=np.float32)
    p1 = p0 + dir2 * float(length_px)
    return p0, p1


def _get_font():
    try:
        return ImageFont.truetype(MANROPE_TTF_PATH, FONT_SIZE)
    except:
        return ImageFont.load_default()


def _draw_text_pil(img_bgr, lines, xy=(20, 20), color=(255, 255, 255)):
    """
    Draw Manrope text (via PIL) onto an OpenCV BGR image.
    """
    font = _get_font()
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil)

    x, y = xy
    line_h = FONT_SIZE + 6
    for i, line in enumerate(lines):
        draw.text((x, y + i * line_h), line, font=font, fill=color)

    out_rgb = np.array(pil)
    out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
    return out_bgr

def _draw_text_cv2(img_bgr, lines, xy=(20, 50), color=(255, 255, 255), size=0.6):

    x, y = xy
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    line_h = int(30 * size) 

    for i, line in enumerate(lines):
        cv2.putText(img_bgr, line, (x, y + (i +1)* line_h), 
                    font, size, color, thickness, cv2.LINE_AA)
    return img_bgr


def _make_frame(canvas_bgr, face_lms, gaze_s, origin, X, Y, Z, depth_out, eye_px, dx, dy, w, h, LEFT_IRIS_IDXS, RIGHT_IRIS_IDXS, PLOT_LMS=True, DRAW_AXES=True, DRAW_GAZE=True):
    scale_factor = w / 960.0

    # ----- Draw axes & gaze -----
    origin_px = np.array([origin[0] * w, origin[1] * h], dtype=np.float32)
    axes_color = _depth_to_bgr(depth_out)

    if PLOT_LMS:
        for lm in face_lms:
            px, py = int(lm.x * w), int(lm.y * h)
            cv2.circle(canvas_bgr, (px, py), int(1 * scale_factor), (200, 200, 200), -1)

        # iris points emphasized
        for idx in LEFT_IRIS_IDXS + RIGHT_IRIS_IDXS:
            lm = face_lms[idx]
            px, py = int(lm.x * w), int(lm.y * h)
            cv2.circle(canvas_bgr, (px, py), int(2 * scale_factor), (255, 255, 255), -1)

    if DRAW_AXES:
        # X axis arrow (depth-colored)
        p0, p1 = _project_axis_2d(origin, X, w, h, AXIS_LEN_PX)
        _draw_arrow(canvas_bgr, p0, p1, axes_color, thickness=int(2 * scale_factor))
        cv2.putText(canvas_bgr, "X", (int(p1[0]) + 6, int(p1[1]) + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, axes_color, 1, cv2.LINE_AA)

        # Y axis arrow
        p0, p1 = _project_axis_2d(origin, Y, w, h, AXIS_LEN_PX)
        _draw_arrow(canvas_bgr, p0, p1, axes_color, thickness=int(2 * scale_factor))
        cv2.putText(canvas_bgr, "Y", (int(p1[0]) + 6, int(p1[1]) + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, axes_color, 1, cv2.LINE_AA)

        # Z axis arrow
        p0, p1 = _project_axis_2d(origin, Z, w, h, AXIS_LEN_PX)
        _draw_arrow(canvas_bgr, p0, p1, axes_color, thickness=int(2 * scale_factor))
        cv2.putText(canvas_bgr, "Z", (int(p1[0]) + 6, int(p1[1]) + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, axes_color, 1, cv2.LINE_AA)

        # origin marker
        cv2.circle(canvas_bgr, (int(origin_px[0]), int(origin_px[1])), int(4 * scale_factor), axes_color, -1)

    if DRAW_GAZE:
        # Draw gaze in white (using gaze vector's x,y as direction)
        p0, p1 = _project_axis_2d(origin, gaze_s, w, h, GAZE_LEN_PX)
        _draw_arrow(canvas_bgr, p0, p1, (255, 255, 255), thickness=int(2 * scale_factor))
        cv2.putText(canvas_bgr, "gaze", (int(p1[0]) + 6, int(p1[1]) + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # ----- Manrope overlay text -----
    lines = [
        f"gaze: {gaze_s[0]:+.3f}, {gaze_s[1]:+.3f}, {gaze_s[2]:+.3f}",
        f"depth: {depth_out:.2f}   eye_px: {eye_px:.1f}",
        f"dx: {dx:+.4f}   dy: {dy:+.4f}",
    ]
    canvas_bgr = _draw_text_cv2(canvas_bgr, lines, xy=(16, 16), color=(255, 255, 255))

    return canvas_bgr