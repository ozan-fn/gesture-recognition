import os
import sys

import cv2
import numpy as np

# trik cerdas: sembunyikan tensorflow dari deteksi internal mediapipe
sys.modules['tensorflow'] = None
import mediapipe as mp

# kembalikan atau hapus trik setelah mediapipe sukses diimpor
del sys.modules['tensorflow']

from tensorflow.keras.models import load_model

# ==========================================
# 1. INISIALISASI MODEL DAN MEDIAPIPE
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, '..', 'dataset')
MODEL_NAME = 'model_lstm_gerakan-all-60fps.keras'
model_path = os.path.join(BASE_DIR, MODEL_NAME)

if not os.path.exists(model_path):
    print(f"❌ Model tidak ditemukan di: {model_path}")
    sys.exit(1)

def list_gesture_folders(dataset_dir):
    if not os.path.isdir(dataset_dir):
        return []

    folders = []
    for name in sorted(os.listdir(dataset_dir)):
        path = os.path.join(dataset_dir, name)
        if not os.path.isdir(path):
            continue
        if name in {"Logs", "__pycache__"} or name.startswith("."):
            continue
        folders.append(name)
    return folders

DAFTAR_GERAKAN = np.array(list_gesture_folders(DATASET_DIR))
if DAFTAR_GERAKAN.size == 0:
    print(f"❌ Tidak ada folder gerakan yang valid di {DATASET_DIR}.")
    sys.exit(1)

print(f"=== Memuat Model: {os.path.basename(model_path)} ===")
model = load_model(model_path)

# Inisialisasi MediaPipe Holistic & Drawing Utilities
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ==========================================
# 2. FUNGSI PENDUKUNG (EXTRACT KEYPOINTS)
# ==========================================

def extract_keypoints(results):
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([face, pose, lh, rh])

# ==========================================
# 3. KONFIGURASI & STATE MACHINE
# ==========================================

SEQUENCE_LENGTH = 60
PREDICTION_THRESHOLD = 0.85
COOLDOWN_FRAMES = 45
HAND_STREAK_TRIGGER = 5

STATE_IDLE = 'IDLE'
STATE_RECORDING = 'RECORDING'
STATE_PREDICTING = 'PREDICTING'
STATE_COOLDOWN = 'COOLDOWN'


def draw_confidence_panel(frame, class_names, preds):
    if preds is None:
        return

    h, w = frame.shape[:2]
    panel_w = 280
    panel_x = w - panel_w - 10
    bar_h = 22
    bar_gap = 8
    start_y = 60
    max_bar_w = panel_w - 120

    order = np.argsort(preds)[::-1]

    for rank, idx in enumerate(order):
        conf = float(preds[idx])
        label = class_names[idx]
        y = start_y + rank * (bar_h + bar_gap)

        bar_color = (0, 220, 80) if rank == 0 else (100, 100, 100)
        filled_w = int(max_bar_w * conf)

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
    h, w = frame.shape[:2]
    progress = len(sequence_buffer) / SEQUENCE_LENGTH
    bar_w = int(w * 0.6)
    bar_x = int(w * 0.2)
    bar_y = h - 50

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
        label = f"{verdict}  ({last_conf:.0%})"
        color = (0, 255, 80) if last_pred else (0, 80, 255)
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


state = STATE_IDLE
sequence_buffer = []
last_pred = ""
last_conf = 0.0
last_preds_raw = None
cooldown_counter = 0
hand_seen_streak = 0

cap = cv2.VideoCapture(0)
print("\n📹 Kamera Aktif! Tunjukkan tangan untuk mulai merekam gesture...")
print("Tekan tombol 'q' atau ESC untuk keluar.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb_frame)

    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS,
            mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
        )
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    hands_present = (
        results.left_hand_landmarks is not None or
        results.right_hand_landmarks is not None
    )

    if hands_present:
        hand_seen_streak += 1
    else:
        hand_seen_streak = 0

    if state == STATE_IDLE:
        if hand_seen_streak >= HAND_STREAK_TRIGGER:
            sequence_buffer = []
            state = STATE_RECORDING

    elif state == STATE_RECORDING:
        keypoints = extract_keypoints(results)
        sequence_buffer.append(keypoints)
        if len(sequence_buffer) == SEQUENCE_LENGTH:
            state = STATE_PREDICTING

    elif state == STATE_PREDICTING:
        res = model.predict(np.expand_dims(sequence_buffer, axis=0), verbose=0)[0]
        conf = float(np.max(res))
        pred_class = DAFTAR_GERAKAN[int(np.argmax(res))]

        last_preds_raw = res
        last_conf = conf
        last_pred = pred_class if conf >= PREDICTION_THRESHOLD else ""

        cooldown_counter = COOLDOWN_FRAMES
        state = STATE_COOLDOWN

    elif state == STATE_COOLDOWN:
        cooldown_counter -= 1
        if cooldown_counter <= 0:
            last_pred = ""
            last_conf = 0.0
            last_preds_raw = None
            hand_seen_streak = 0
            state = STATE_IDLE

    if state == STATE_RECORDING:
        draw_recording_bar(frame, sequence_buffer)

    draw_status_bar(frame, state, last_pred, last_conf)
    draw_confidence_panel(frame, DAFTAR_GERAKAN, last_preds_raw)

    if state == STATE_COOLDOWN:
        draw_result_center(frame, last_pred, last_conf)

    cv2.imshow('BISINDO Real-Time Recognition', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
holistic.close()
print("\n🏁 Uji coba real-time selesai.")
