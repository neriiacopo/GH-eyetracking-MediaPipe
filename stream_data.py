import time
import socket
import numpy as np
import cv2
import os

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import utils.plotting as utils_plt
import utils.detecting as utils_det

# =============================
# UDP setup
# =============================
UDP_IP = "127.0.0.1"
UDP_PORT_OUT = 5005 # OUT --> gh
UDP_PORT_IN = 5006  # IN <-- gh
sock_out = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_in = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_in.bind((UDP_IP, UDP_PORT_IN))
sock_in.setblocking(False)

# =============================
# Video setup
# =============================
SHOW_VIDEO = False          
PLOT_LMS = True            
DRAW_AXES = True            
DRAW_GAZE = True           
WINDOW_NAME = "Gaze UDP (MediaPipe)"
SAVE_DIR = "frames"
RECORDING = False
EXPORT_FPS = 10  

export_interval = 1.0 / EXPORT_FPS
last_export_t = 0
os.makedirs(SAVE_DIR, exist_ok=True)

# =============================
# MediaPipe model
# =============================
MODEL_PATH = "face_landmarker.task"

class EMA:
    def __init__(self, alpha=0.25):
        self.alpha = float(alpha)
        self.v = None

    def update(self, x):
        x = np.asarray(x, dtype=np.float32)
        if self.v is None:
            self.v = x
        else:
            self.v = (1.0 - self.alpha) * self.v + self.alpha * x
        return self.v

def main():
    global last_export_t, RECORDING

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False
    )
    landmarker = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam (index 0).")

    ema = EMA(alpha=0.25)

    print(f"Streaming gaze over UDP to {UDP_IP}:{UDP_PORT_OUT} ... Press ESC to quit.")
    t0 = time.time()

    while True:
        current_t = time.time()

        try:
            data, addr = sock_in.recvfrom(1024)
            msg_in = data.decode("utf-8").strip().upper()
            if msg_in == "START":
                RECORDING = True
            elif msg_in == "STOP":
                RECORDING = False
        except BlockingIOError:
            pass
        
        ok, frame_bgr = cap.read()
        if not ok:
            break

        h, w = frame_bgr.shape[:2]
        canvas_bgr = frame_bgr.copy() if SHOW_VIDEO else np.zeros_like(frame_bgr)

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        timestamp_ms = int((time.time() - t0) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.face_landmarks:
            face_lms = result.face_landmarks[0]
            origin, X, Y, Z = utils_det._orthonormal_frame_from_landmarks(face_lms)

            gaze, dx, dy = utils_det._estimate_gaze_in_head_frame(face_lms, X, Y, Z, gain_xy=2.5, z_bias=1.0)
            gaze_s = ema.update(gaze)

            eye_px, depth_rel = utils_det._estimate_depth_from_eyes_px(face_lms, w, h)
            depth_out = depth_rel * 1000.0

            # UDP: gx,gy,gz,depth,eye_px
            msg = f"{gaze_s[0]:.5f},{gaze_s[1]:.5f},{gaze_s[2]:.5f},{depth_out:.5f},{eye_px:.2f}"
            sock_out.sendto(msg.encode("utf-8"), (UDP_IP, UDP_PORT_OUT))


            # ----- Draw landmarks -----
            canvas_bgr = utils_plt._make_frame(
                canvas_bgr, face_lms, gaze_s, origin, X, Y, Z,
                depth_out, eye_px, dx, dy, w, h,
                LEFT_IRIS_IDXS=utils_det.LEFT_IRIS_IDXS,
                RIGHT_IRIS_IDXS=utils_det.RIGHT_IRIS_IDXS,
                PLOT_LMS=PLOT_LMS,
                DRAW_AXES=DRAW_AXES,
                DRAW_GAZE=DRAW_GAZE
            )

        else:
            # No face detected
            canvas_bgr = utils_plt._draw_text_pil(canvas_bgr, ["No face detected"], xy=(16, 16), color=(255, 100, 100))

        if RECORDING:
            if (current_t - last_export_t) >= export_interval:
                ts = int(current_t * 1000)
                cv2.imwrite(os.path.join(SAVE_DIR, f"frame_{ts}.jpg"), canvas_bgr)
                last_export_t = current_t

        cv2.imshow(WINDOW_NAME, canvas_bgr)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()