import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pickle
import threading

os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')

# === CONFIGURATION ===
MODEL_PATH          = 'bisindo_model.h5'
LABELS_PATH         = 'class_names.pkl'
SCALER_PATH         = 'scaler.pkl'
SEQUENCE_LENGTH     = 30
PREDICTION_THRESHOLD = 0.80
COOLDOWN_FRAMES     = 45
EMA_ALPHA           = 0.2

POSE_LANDMARKS_IDX = [11, 12, 13, 14, 15, 16]

mp_holistic = mp.solutions.holistic  # type: ignore
mp_drawing  = mp.solutions.drawing_utils  # type: ignore

# === STATE MACHINE ===
STATE_IDLE       = 'IDLE'
STATE_RECORDING  = 'RECORDING'
STATE_PREDICTING = 'PREDICTING'
STATE_COOLDOWN   = 'COOLDOWN'

# ── Normalisasi (EMA & Scale Identical to Script 1) ───────────────────────

class Normalizer:
    def __init__(self):
        self.ref_ema = None
        self.scale_ema = None

    def process(self, results, image_shape):
        ref, scale = self.get_ref_and_scale(results, image_shape)
        if self.ref_ema is None:
            self.ref_ema = ref
            self.scale_ema = scale
        else:
            if self.ref_ema is not None and self.scale_ema is not None:
                self.ref_ema   = EMA_ALPHA * ref + (1 - EMA_ALPHA) * self.ref_ema
                self.scale_ema = EMA_ALPHA * scale + (1 - EMA_ALPHA) * self.scale_ema
        return self.ref_ema, self.scale_ema

    def get_ref_and_scale(self, results, image_shape):
        if results.pose_landmarks:
            left  = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER]
            right = results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER]
            lx, ly = left.x * image_shape[1], left.y * image_shape[0]
            rx, ry = right.x * image_shape[1], right.y * image_shape[0]
            mid_x = (lx + rx) / 2
            mid_y = (ly + ry) / 2
            dist = np.sqrt((lx - rx)**2 + (ly - ry)**2)
            if dist < 1.0:
                dist = 1.0
            return np.array([mid_x, mid_y]), dist
        return np.array([image_shape[1] / 2, image_shape[0] / 2]), 100.0

def extract_keypoints(results, image_shape, normalizer):
    ref, scale = normalizer.process(results, image_shape)

    # Pose: 6 landmarks (12 features)
    pose = np.zeros(len(POSE_LANDMARKS_IDX) * 2)
    if results.pose_landmarks:
        pts = []
        for idx in POSE_LANDMARKS_IDX:
            lm = results.pose_landmarks.landmark[idx]
            pts.append([lm.x * image_shape[1], lm.y * image_shape[0]])
        pts = np.array(pts)
        pose = ((pts - ref) / scale).flatten()

    # Left Hand: 21 landmarks (42 features)
    lh = np.zeros(21 * 2)
    if results.left_hand_landmarks:
        w_lm = results.left_hand_landmarks.landmark[0]
        wrist = np.array([w_lm.x * image_shape[1], w_lm.y * image_shape[0]])
        m_lm = results.left_hand_landmarks.landmark[9]
        mcp = np.array([m_lm.x * image_shape[1], m_lm.y * image_shape[0]])
        hand_scale = np.linalg.norm(wrist - mcp)
        if hand_scale < 1.0:
            hand_scale = 1.0

        lh_wrist_global = (wrist - ref) / scale
        lh_local = []
        for lm in results.left_hand_landmarks.landmark[1:]:
            pt = np.array([lm.x * image_shape[1], lm.y * image_shape[0]])
            lh_local.append((pt - wrist) / hand_scale)
        lh_local = np.array(lh_local).flatten()
        lh = np.concatenate([lh_wrist_global, lh_local])

    # Right Hand: 21 landmarks (42 features)
    rh = np.zeros(21 * 2)
    if results.right_hand_landmarks:
        w_lm = results.right_hand_landmarks.landmark[0]
        wrist = np.array([w_lm.x * image_shape[1], w_lm.y * image_shape[0]])
        m_lm = results.right_hand_landmarks.landmark[9]
        mcp = np.array([m_lm.x * image_shape[1], m_lm.y * image_shape[0]])
        hand_scale = np.linalg.norm(wrist - mcp)
        if hand_scale < 1.0:
            hand_scale = 1.0

        rh_wrist_global = (wrist - ref) / scale
        rh_local = []
        for lm in results.right_hand_landmarks.landmark[1:]:
            pt = np.array([lm.x * image_shape[1], lm.y * image_shape[0]])
            rh_local.append((pt - wrist) / hand_scale)
        rh_local = np.array(rh_local).flatten()
        rh = np.concatenate([rh_wrist_global, rh_local])

    return np.concatenate([pose, lh, rh]), ref

# ── Threaded Webcam ───────────────────────────────────────────────────────

class WebcamStream:
    def __init__(self, src=0, width=1280, height=720):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.lock    = threading.Lock()
        threading.Thread(target=self._update, daemon=True).start()

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def release(self):
        self.stopped = True
        self.cap.release()

# ── UI Drawing ────────────────────────────────────────────────────────────

def draw_confidence_panel(frame, class_names, preds):
    if preds is None:
        return

    h, w      = frame.shape[:2]
    panel_w   = 280
    panel_x   = w - panel_w - 10
    bar_h     = 22
    bar_gap   = 8
    start_y   = 60
    max_bar_w = panel_w - 120

    order = np.argsort(preds)[::-1]

    for rank, idx in enumerate(order):
        conf      = float(preds[idx])
        label     = class_names[idx]
        y         = start_y + rank * (bar_h + bar_gap)

        bar_color = (0, 220, 80) if rank == 0 else (100, 100, 100)
        filled_w  = int(max_bar_w * conf)

        cv2.rectangle(frame, (panel_x, y), (panel_x + max_bar_w, y + bar_h), (30, 30, 30), -1)
        if filled_w > 0:
            cv2.rectangle(frame, (panel_x, y), (panel_x + filled_w, y + bar_h), bar_color, -1)

        cv2.putText(frame, label, (panel_x - 5, y + bar_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

        cv2.putText(frame, f"{conf:.0%}", (panel_x + max_bar_w + 5, y + bar_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 80) if rank == 0 else (160, 160, 160), 1, cv2.LINE_AA)

    cv2.putText(frame, "Confidence", (panel_x, start_y - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

def draw_recording_bar(frame, sequence_buffer):
    h, w     = frame.shape[:2]
    progress = len(sequence_buffer) / SEQUENCE_LENGTH
    bar_w    = int(w * 0.6)
    bar_x    = int(w * 0.2)
    bar_y    = h - 50

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 20), (50, 50, 50), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + 20), (0, 200, 255), -1)
    cv2.putText(frame, f"Merekam: {len(sequence_buffer)}/{SEQUENCE_LENGTH} frame", (bar_x, bar_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)

def draw_status_bar(frame, state, last_pred, last_conf):
    if state == STATE_IDLE:
        label = "Idle — Tunjukkan tangan untuk mulai"
        color = (160, 160, 160)
    elif state == STATE_RECORDING:
        label = "Merekam gesture..."
        color = (0, 200, 255)
    elif state == STATE_PREDICTING:
        label = "Memproses..."
        color = (255, 200, 0)
    elif state == STATE_COOLDOWN:
        verdict = last_pred if last_pred else "Tidak Dikenali"
        label   = f"{verdict}  ({last_conf:.0%})"
        color   = (0, 255, 80) if last_pred else (0, 80, 255)
    else:
        return

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    cv2.rectangle(frame, (15, 12), (25 + tw, 30 + th), (0, 0, 0), -1)
    cv2.putText(frame, label, (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)

def draw_result_center(frame, last_pred, last_conf):
    if not last_pred:
        return
    h, w = frame.shape[:2]

    (tw, th), _ = cv2.getTextSize(last_pred, cv2.FONT_HERSHEY_SIMPLEX, 3.5, 7)
    tx = (w - tw) // 2
    ty = h // 2

    cv2.rectangle(frame, (tx - 15, ty - th - 15), (tx + tw + 15, ty + 20), (0, 0, 0), -1)
    cv2.putText(frame, last_pred, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (0, 255, 80), 7, cv2.LINE_AA)

    conf_label = f"{last_conf:.1%} confidence"
    (cw, ch), _ = cv2.getTextSize(conf_label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cx = (w - cw) // 2
    cv2.putText(frame, conf_label, (cx, ty + ch + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 80), 2, cv2.LINE_AA)

# ── Main ──────────────────────────────────────────────────────────────────

def main():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, 'rb') as f:
        class_names = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)

    print(f"Model loaded. Classes: {class_names}")

    state            = STATE_IDLE
    sequence_buffer  = []
    last_pred        = ""
    last_conf        = 0.0
    last_preds_raw   = None
    cooldown_counter = 0
    
    hand_seen_streak = 0

    normalizer = Normalizer()
    stream = WebcamStream(width=1280, height=720)

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,           # Menggunakan model paling akurat (Heavy)
        min_detection_confidence=0.5, # Mengambil logika threshold dari skrip 2
        min_tracking_confidence=0.5,  # Mengambil logika threshold dari skrip 2
        enable_segmentation=False
    ) as holistic:
        
        while True:
            ret, frame = stream.read()
            if not ret or frame is None:
                continue

            frame     = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results   = holistic.process(image_rgb)

            hands_present = (
                results.left_hand_landmarks  is not None or
                results.right_hand_landmarks is not None
            )

            if hands_present:
                hand_seen_streak += 1
            else:
                hand_seen_streak = 0

            # ── STATE MACHINE ─────────────────────────────────────────────

            if state == STATE_IDLE:
                if hand_seen_streak >= 5:
                    sequence_buffer = []
                    state = STATE_RECORDING

            elif state == STATE_RECORDING:
                keypoints, ref = extract_keypoints(results, frame.shape, normalizer)
                sequence_buffer.append(keypoints)
                if len(sequence_buffer) == SEQUENCE_LENGTH:
                    state = STATE_PREDICTING

            elif state == STATE_PREDICTING:
                seq_arr = np.array(sequence_buffer)
                seq_scaled = scaler.transform(seq_arr)
                
                input_data = np.expand_dims(seq_scaled, axis=0)
                preds = model.predict(input_data, verbose=0)[0]
                conf = float(np.max(preds))
                pred_class = class_names[int(np.argmax(preds))]
                
                last_preds_raw = preds
                last_conf = conf
                last_pred = pred_class if conf >= PREDICTION_THRESHOLD else ""

                cooldown_counter = COOLDOWN_FRAMES
                state = STATE_COOLDOWN

            elif state == STATE_COOLDOWN:
                cooldown_counter -= 1
                if cooldown_counter <= 0:
                    last_pred      = ""
                    last_conf      = 0.0
                    last_preds_raw = None
                    hand_seen_streak = 0
                    state = STATE_IDLE

            # ── DRAW LANDMARKS ────────────────────────────────────────────

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            _, ref = extract_keypoints(results, frame.shape, normalizer)
            cv2.circle(frame, (int(ref[0]), int(ref[1])), 6, (0, 0, 255), -1)

            # ── DRAW UI ───────────────────────────────────────────────────

            if state == STATE_RECORDING:
                draw_recording_bar(frame, sequence_buffer)

            draw_status_bar(frame, state, last_pred, last_conf)
            draw_confidence_panel(frame, class_names, last_preds_raw)

            if state == STATE_COOLDOWN:
                draw_result_center(frame, last_pred, last_conf)

            cv2.imshow('BISINDO Real-Time Recognition', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

    stream.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
