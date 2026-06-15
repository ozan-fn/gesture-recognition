import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pickle
import streamlit as st
import tempfile
import subprocess

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

# ── Normalisasi (EMA & Scale Identical to Script 3) ───────────────────────

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

# ── UI Drawing ────────────────────────────────────────────────────────────

def draw_confidence_panel(frame, class_names, preds):
    if preds is None:
        return

    h, w      = frame.shape[:2]
    panel_w   = 200
    panel_x   = w - panel_w - 10
    bar_h     = 18
    bar_gap   = 6
    start_y   = 50
    max_bar_w = panel_w - 80

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

        cv2.putText(frame, label, (panel_x - 5, y + bar_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (220, 220, 220), 1, cv2.LINE_AA)

        cv2.putText(frame, f"{conf:.0%}", (panel_x + max_bar_w + 5, y + bar_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 80) if rank == 0 else (160, 160, 160), 1, cv2.LINE_AA)

    cv2.putText(frame, "Confidence", (panel_x, start_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

def draw_recording_bar(frame, sequence_buffer):
    h, w     = frame.shape[:2]
    progress = len(sequence_buffer) / SEQUENCE_LENGTH
    bar_w    = int(w * 0.5)
    bar_x    = int(w * 0.25)
    bar_y    = h - 40

    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 15), (50, 50, 50), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_w * progress), bar_y + 15), (0, 200, 255), -1)
    cv2.putText(frame, f"Merekam: {len(sequence_buffer)}/{SEQUENCE_LENGTH}", (bar_x, bar_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

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

    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (10, 8), (20 + tw, 20 + th), (0, 0, 0), -1)
    cv2.putText(frame, label, (15, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

def draw_result_center(frame, last_pred, last_conf):
    if not last_pred:
        return
    h, w = frame.shape[:2]

    (tw, th), _ = cv2.getTextSize(last_pred, cv2.FONT_HERSHEY_SIMPLEX, 2.0, 5)
    tx = (w - tw) // 2
    ty = h // 2

    cv2.rectangle(frame, (tx - 10, ty - th - 10), (tx + tw + 10, ty + 15), (0, 0, 0), -1)
    cv2.putText(frame, last_pred, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 80), 5, cv2.LINE_AA)

    conf_label = f"{last_conf:.1%} confidence"
    (cw, ch), _ = cv2.getTextSize(conf_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cx = (w - cw) // 2
    cv2.putText(frame, conf_label, (cx, ty + ch + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 80), 2, cv2.LINE_AA)

# ── Load Model ────────────────────────────────────────────────────────────

@st.cache_resource
def load_model_and_artifacts():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, 'rb') as f:
        class_names = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    return model, class_names, scaler

# ── Process Frame ─────────────────────────────────────────────────────────

def process_frame(frame, holistic, normalizer, model, class_names, scaler, state_dict):
    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)

    hands_present = (
        results.left_hand_landmarks is not None or
        results.right_hand_landmarks is not None
    )

    if hands_present:
        state_dict['hand_seen_streak'] += 1
    else:
        state_dict['hand_seen_streak'] = 0

    # ── STATE MACHINE ─────────────────────────────────────────────

    state = state_dict['state']

    if state == STATE_IDLE:
        if state_dict['hand_seen_streak'] >= 5:
            state_dict['sequence_buffer'] = []
            state_dict['state'] = STATE_RECORDING

    elif state == STATE_RECORDING:
        keypoints, ref = extract_keypoints(results, frame.shape, normalizer)
        state_dict['sequence_buffer'].append(keypoints)
        if len(state_dict['sequence_buffer']) == SEQUENCE_LENGTH:
            state_dict['state'] = STATE_PREDICTING

    elif state == STATE_PREDICTING:
        seq_arr = np.array(state_dict['sequence_buffer'])
        seq_scaled = scaler.transform(seq_arr)
        
        input_data = np.expand_dims(seq_scaled, axis=0)
        preds = model.predict(input_data, verbose=0)[0]
        conf = float(np.max(preds))
        pred_class = class_names[int(np.argmax(preds))]
        
        state_dict['last_preds_raw'] = preds
        state_dict['last_conf'] = conf
        state_dict['last_pred'] = pred_class if conf >= PREDICTION_THRESHOLD else ""

        state_dict['cooldown_counter'] = COOLDOWN_FRAMES
        state_dict['state'] = STATE_COOLDOWN

    elif state == STATE_COOLDOWN:
        state_dict['cooldown_counter'] -= 1
        if state_dict['cooldown_counter'] <= 0:
            state_dict['last_pred'] = ""
            state_dict['last_conf'] = 0.0
            state_dict['last_preds_raw'] = None
            state_dict['hand_seen_streak'] = 0
            state_dict['state'] = STATE_IDLE

    # ── DRAW LANDMARKS ────────────────────────────────────────────

    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    _, ref = extract_keypoints(results, frame.shape, normalizer)
    cv2.circle(frame, (int(ref[0]), int(ref[1])), 6, (0, 0, 255), -1)

    # ── DRAW UI ───────────────────────────────────────────────────

    if state_dict['state'] == STATE_RECORDING:
        draw_recording_bar(frame, state_dict['sequence_buffer'])

    draw_status_bar(frame, state_dict['state'], state_dict['last_pred'], state_dict['last_conf'])
    draw_confidence_panel(frame, class_names, state_dict['last_preds_raw'])

    if state_dict['state'] == STATE_COOLDOWN:
        draw_result_center(frame, state_dict['last_pred'], state_dict['last_conf'])

    return frame

# ── Streamlit App ─────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="BISINDO Recognition", layout="centered")
    st.title("🤟 BISINDO Real-Time Recognition")

    model, class_names, scaler = load_model_and_artifacts()

    tab1, tab2 = st.tabs(["📹 Webcam", "🔗 Rangkai Kalimat"])

    # ── Tab 1: Webcam ─────────────────────────────────────────────

    with tab1:
        st.subheader("Webcam Real-Time Recognition")
        
        run_webcam = st.checkbox("Aktifkan Webcam", value=False)
        
        if run_webcam:
            stframe = st.empty()
            
            # Initialize state
            if 'webcam_state' not in st.session_state:
                st.session_state.webcam_state = {
                    'state': STATE_IDLE,
                    'sequence_buffer': [],
                    'last_pred': "",
                    'last_conf': 0.0,
                    'last_preds_raw': None,
                    'cooldown_counter': 0,
                    'hand_seen_streak': 0
                }
            
            normalizer = Normalizer()
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            with mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                enable_segmentation=False
            ) as holistic:
                
                while run_webcam:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Tidak dapat membaca dari webcam")
                        break

                    processed_frame = process_frame(
                        frame, holistic, normalizer, model, class_names, scaler,
                        st.session_state.webcam_state
                    )

                    stframe.image(processed_frame, channels="BGR", width=800)

            cap.release()
        else:
            st.info("Centang checkbox di atas untuk mengaktifkan webcam")

    # ── Tab 2: Rangkai Kalimat ────────────────────────────────────

    with tab2:
        st.subheader("Rangkai Kalimat BISINDO")
        
        st.info("Pilih kata-kata untuk membuat kalimat, lalu proses secara berurutan")
        
        # Get available words from dataset
        dataset_path = "raw_video"
        available_words = []
        
        if os.path.exists(dataset_path):
            for item in os.listdir(dataset_path):
                item_path = os.path.join(dataset_path, item)
                if os.path.isdir(item_path):
                    # Check if folder has video files
                    videos = [f for f in os.listdir(item_path) if f.endswith(('.mp4', '.avi', '.mov'))]
                    if videos:
                        available_words.append(item)
            available_words.sort()
        
        if available_words:
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Multi-select for words
                selected_words = st.multiselect(
                    "Pilih kata-kata untuk membuat kalimat:",
                    options=available_words,
                    help="Pilih kata dalam urutan yang diinginkan"
                )
            
            with col2:
                st.write("")
                st.write("")
                if st.button("Reset", use_container_width=True):
                    st.rerun()
            
            # Display selected sentence
            if selected_words:
                st.success(f"Kalimat: **{' '.join(selected_words)}**")
                st.write(f"Total kata: {len(selected_words)}")
                
                # Process button
                process_sentence = st.button("🎬 Proses Kalimat", type="primary", use_container_width=True)
                
                if process_sentence:
                    stframe = st.empty()
                    st_status = st.empty()
                    st_progress = st.progress(0)
                    
                    # Initialize state for continuous processing
                    sentence_state = {
                        'state': STATE_IDLE,
                        'sequence_buffer': [],
                        'last_pred': "",
                        'last_conf': 0.0,
                        'last_preds_raw': None,
                        'cooldown_counter': 0,
                        'hand_seen_streak': 0
                    }
                    
                    normalizer = Normalizer()
                    
                    with mp_holistic.Holistic(
                        static_image_mode=False,
                        model_complexity=2,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5,
                        enable_segmentation=False
                    ) as holistic:
                        
                        for word_idx, word in enumerate(selected_words):
                            st_status.info(f"Memproses kata {word_idx + 1}/{len(selected_words)}: **{word}**")
                            st_progress.progress((word_idx) / len(selected_words))
                            
                            # Get video for this word
                            word_folder = os.path.join(dataset_path, word)
                            videos = [f for f in os.listdir(word_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
                            
                            if videos:
                                video_path = os.path.join(word_folder, videos[0])  # Use first video
                                
                                # Extract and play audio
                                audio_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                                try:
                                    # Extract audio from video using ffmpeg
                                    subprocess.run([
                                        'ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a',
                                        audio_file.name, '-y'
                                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                                    
                                    # Play audio in Streamlit
                                    with open(audio_file.name, 'rb') as f:
                                        st.audio(f.read(), format='audio/mp3')
                                except Exception:
                                    # If audio extraction fails, continue without audio
                                    pass
                                finally:
                                    # Clean up temp file
                                    try:
                                        os.unlink(audio_file.name)
                                    except OSError:
                                        pass
                                
                                # Reset state for new word
                                sentence_state['state'] = STATE_IDLE
                                sentence_state['sequence_buffer'] = []
                                sentence_state['hand_seen_streak'] = 0
                                
                                cap = cv2.VideoCapture(video_path)
                                
                                while cap.isOpened():
                                    ret, frame = cap.read()
                                    if not ret:
                                        break
                                    
                                    processed_frame = process_frame(
                                        frame, holistic, normalizer, model, class_names, scaler,
                                        sentence_state
                                    )
                                    
                                    stframe.image(processed_frame, channels="BGR", width=800)
                                
                                cap.release()
                            
                            else:
                                st.warning(f"Tidak ada video untuk kata: {word}")
                        
                        st_progress.progress(1.0)
                        st.success("✅ Semua kata dalam kalimat selesai diproses!")
            
            else:
                st.info("👆 Pilih kata-kata di atas untuk membuat kalimat")
        
        else:
            st.warning(f"Folder {dataset_path} tidak ditemukan atau kosong. Silakan extract video terlebih dahulu.")

if __name__ == "__main__":
    main()
