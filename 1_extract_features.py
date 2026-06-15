import os
import cv2
import numpy as np
from tqdm import tqdm
import mediapipe as mp

os.environ.setdefault('QT_QPA_PLATFORM', 'xcb')

# === CONFIGURATION ===
RAW_VIDEO_DIR = 'Dataset/raw_video'
OUTPUT_DIR    = 'MP_Data'
SEQUENCE_LENGTH = 30
MP_MIN_WIDTH = 640
SHOW_PREVIEW = False  # Disable preview for headless environments

# We only track critical pose points: shoulders (11, 12), elbows (13, 14), wrists (15, 16)
# This reduces dimensionality and avoids face/leg noise.
POSE_LANDMARKS_IDX = [11, 12, 13, 14, 15, 16]

mp_holistic  = mp.solutions.holistic  # type: ignore
mp_drawing   = mp.solutions.drawing_utils  # type: ignore

def get_ref_and_scale(results, image_shape):
    """Calculates shoulder midpoint and shoulder distance for scale-invariant normalization."""
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

def extract_keypoints(results, image_shape, prev_ref=None, prev_scale=None, prev_keypoints=None):
    """
    Extracts high-quality features:
    1. Critical pose landmarks normalized relative to shoulders.
    2. Hand shape normalized locally relative to hand's own wrist and hand scale.
    3. Global hand movement relative to shoulders.
    4. NEW: Temporal features (velocity, acceleration)
    5. NEW: Relative distances (hand-to-hand, hand-to-face)
    6. NEW: Angle features (elbow angles)
    """
    ref, scale = get_ref_and_scale(results, image_shape)
    if not results.pose_landmarks and prev_ref is not None:
        ref   = prev_ref
        scale = prev_scale

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
    left_wrist_pos = None
    if results.left_hand_landmarks:
        # Wrist coordinate
        w_lm = results.left_hand_landmarks.landmark[0]
        wrist = np.array([w_lm.x * image_shape[1], w_lm.y * image_shape[0]])
        left_wrist_pos = wrist.copy()
        
        # Hand scale (distance between Wrist and Middle Finger root)
        m_lm = results.left_hand_landmarks.landmark[9]
        mcp = np.array([m_lm.x * image_shape[1], m_lm.y * image_shape[0]])
        hand_scale = np.linalg.norm(wrist - mcp)
        if hand_scale < 1.0:
            hand_scale = 1.0

        # Global position of wrist relative to shoulders
        lh_wrist_global = (wrist - ref) / scale
        
        # Local shape of other joints relative to wrist
        lh_local = []
        for lm in results.left_hand_landmarks.landmark[1:]:
            pt = np.array([lm.x * image_shape[1], lm.y * image_shape[0]])
            lh_local.append((pt - wrist) / hand_scale)
        lh_local = np.array(lh_local).flatten()
        lh = np.concatenate([lh_wrist_global, lh_local])

    # Right Hand: 21 landmarks (42 features)
    rh = np.zeros(21 * 2)
    right_wrist_pos = None
    if results.right_hand_landmarks:
        w_lm = results.right_hand_landmarks.landmark[0]
        wrist = np.array([w_lm.x * image_shape[1], w_lm.y * image_shape[0]])
        right_wrist_pos = wrist.copy()
        
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

    # === NEW FEATURES ===
    
    # 1. Relative Distances (4 features)
    distances = np.zeros(4)
    
    # Safe scale value
    safe_scale = scale if scale is not None and scale > 0 else 1.0
    
    # Hand-to-hand distance
    if left_wrist_pos is not None and right_wrist_pos is not None:
        distances[0] = np.linalg.norm(left_wrist_pos - right_wrist_pos) / safe_scale
    
    # Hand-to-face distances (using shoulder midpoint as face proxy)
    if left_wrist_pos is not None:
        distances[1] = np.linalg.norm(left_wrist_pos - ref) / safe_scale
    if right_wrist_pos is not None:
        distances[2] = np.linalg.norm(right_wrist_pos - ref) / safe_scale
    
    # Hand width ratio (left to right)
    if left_wrist_pos is not None and right_wrist_pos is not None:
        distances[3] = abs(left_wrist_pos[0] - right_wrist_pos[0]) / safe_scale
    
    # 2. Elbow Angles (2 features)
    angles = np.zeros(2)
    
    if results.pose_landmarks:
        # Left elbow angle
        l_shoulder = results.pose_landmarks.landmark[11]
        l_elbow = results.pose_landmarks.landmark[13]
        l_wrist = results.pose_landmarks.landmark[15]
        
        ls = np.array([l_shoulder.x * image_shape[1], l_shoulder.y * image_shape[0]])
        le = np.array([l_elbow.x * image_shape[1], l_elbow.y * image_shape[0]])
        lw = np.array([l_wrist.x * image_shape[1], l_wrist.y * image_shape[0]])
        
        v1 = ls - le
        v2 = lw - le
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angles[0] = np.arccos(np.clip(cos_angle, -1.0, 1.0)) / np.pi  # Normalize to [0, 1]
        
        # Right elbow angle
        r_shoulder = results.pose_landmarks.landmark[12]
        r_elbow = results.pose_landmarks.landmark[14]
        r_wrist = results.pose_landmarks.landmark[16]
        
        rs = np.array([r_shoulder.x * image_shape[1], r_shoulder.y * image_shape[0]])
        re = np.array([r_elbow.x * image_shape[1], r_elbow.y * image_shape[0]])
        rw = np.array([r_wrist.x * image_shape[1], r_wrist.y * image_shape[0]])
        
        v1 = rs - re
        v2 = rw - re
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angles[1] = np.arccos(np.clip(cos_angle, -1.0, 1.0)) / np.pi
    
    # Combine base features
    base_features = np.concatenate([pose, lh, rh, distances, angles])
    
    # 3. Temporal Features (velocity) - will be computed later in sequence
    # For now, return base features and positions for velocity calculation
    
    return base_features, ref, scale, left_wrist_pos, right_wrist_pos

def draw_preview(frame, results, class_name, vid_name, frame_idx, total_frames, sampled_indices, ref):
    """Draws tracking overlay for validation."""
    h, w = frame.shape[:2]
    
    # Draw landmarks
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    # Reference point marker
    cv2.circle(frame, (int(ref[0]), int(ref[1])), 8, (0, 0, 255), -1)

    # UI details
    info = [
        f"Class: {class_name}",
        f"Video: {vid_name}",
        f"Frame: {frame_idx + 1}/{total_frames}",
        f"L Hand: {'OK' if results.left_hand_landmarks else 'MISSING'}",
        f"R Hand: {'OK' if results.right_hand_landmarks else 'MISSING'}"
    ]
    for i, text in enumerate(info):
        color = (0, 255, 0) if "OK" in text or "Class" in text or "Video" in text or "Frame" in text else (0, 0, 255)
        cv2.putText(frame, text, (10, 25 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Progress bar
    bar_w = w - 40
    progress = (frame_idx + 1) / total_frames
    cv2.rectangle(frame, (20, h - 30), (20 + bar_w, h - 15), (50, 50, 50), -1)
    cv2.rectangle(frame, (20, h - 30), (20 + int(bar_w * progress), h - 15), (0, 200, 255), -1)

    for si in sampled_indices:
        sx = 20 + int(bar_w * si / total_frames)
        cv2.circle(frame, (sx, h - 22), 4, (0, 255, 255), -1)

    cv2.putText(frame, "SPACE=skip  Q=quit", (20, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

def sample_frames(total_frames, desired_count):
    if total_frames < desired_count:
        return np.linspace(0, total_frames - 1, desired_count).astype(int)
    return np.round(np.linspace(0, total_frames - 1, desired_count)).astype(int)

def process_video(video_path, holistic, class_name):
    cap = cv2.VideoCapture(video_path)
    vid_name = os.path.basename(video_path)

    frames_data = []
    results_data = []
    keypoints_data = []
    ref_points = []
    wrist_positions = []  # Store wrist positions for velocity
    
    prev_ref, prev_scale = None, None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        if w < MP_MIN_WIDTH:
            scale_up = MP_MIN_WIDTH / w
            mp_frame = cv2.resize(frame, (MP_MIN_WIDTH, int(h * scale_up)))
        else:
            mp_frame = frame

        rgb = cv2.cvtColor(mp_frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        
        kp, prev_ref, prev_scale, left_wrist, right_wrist = extract_keypoints(results, mp_frame.shape, prev_ref, prev_scale)
        
        keypoints_data.append(kp)
        results_data.append(results)
        ref_points.append(prev_ref)
        wrist_positions.append((left_wrist, right_wrist))
        
        if SHOW_PREVIEW:
            small_frame = cv2.resize(mp_frame, (640, int(mp_frame.shape[0] * (640 / mp_frame.shape[1]))))
            frames_data.append(small_frame)
            
    cap.release()

    total_frames = len(keypoints_data)
    if total_frames == 0:
        return None

    # Filter/crop sequence to where hands are active
    hand_indices = [
        i for i, r in enumerate(results_data)
        if r.left_hand_landmarks is not None or r.right_hand_landmarks is not None
    ]

    if len(hand_indices) >= 5:
        start_idx = hand_indices[0]
        end_idx   = hand_indices[-1]
        
        keypoints_data = keypoints_data[start_idx:end_idx + 1]
        results_data   = results_data[start_idx:end_idx + 1]
        ref_points     = ref_points[start_idx:end_idx + 1]
        wrist_positions = wrist_positions[start_idx:end_idx + 1]
        if SHOW_PREVIEW:
            frames_data = frames_data[start_idx:end_idx + 1]
        
        total_frames = len(keypoints_data)
        print(f"  [Crop] {vid_name}: frames {start_idx}-{end_idx} (active: {len(hand_indices)})")
    else:
        print(f"  [Warn] {vid_name}: hands missing. No crop.")

    sampled_indices = sample_frames(total_frames, SEQUENCE_LENGTH)

    if SHOW_PREVIEW and len(frames_data) > 0:
        for i, (frame, results, ref) in enumerate(zip(frames_data, results_data, ref_points)):
            display = frame.copy()
            draw_preview(display, results, class_name, vid_name, i, total_frames, sampled_indices, ref)
            cv2.imshow('Preview Tracking', display)
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                return 'QUIT'
            elif key == ord(' '):
                break

    # Sample keypoints
    sequence = np.array([keypoints_data[i] for i in sampled_indices])
    
    # Compute velocity features (temporal)
    velocities = np.zeros((SEQUENCE_LENGTH, 4))  # left_x, left_y, right_x, right_y
    for i in range(1, SEQUENCE_LENGTH):
        prev_idx = sampled_indices[i-1]
        curr_idx = sampled_indices[i]
        
        prev_left, prev_right = wrist_positions[prev_idx]
        curr_left, curr_right = wrist_positions[curr_idx]
        
        # Left hand velocity
        if prev_left is not None and curr_left is not None:
            velocities[i, 0:2] = (curr_left - prev_left) / (curr_idx - prev_idx + 1)
        
        # Right hand velocity
        if prev_right is not None and curr_right is not None:
            velocities[i, 2:4] = (curr_right - prev_right) / (curr_idx - prev_idx + 1)
    
    # Normalize velocities
    velocities = velocities / (ref_points[0][1] if ref_points[0][1] > 0 else 1.0)  # Scale by shoulder distance
    
    # Concatenate spatial features with temporal features
    sequence_with_velocity = np.concatenate([sequence, velocities], axis=1)
    
    return sequence_with_velocity

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    classes = sorted([d for d in os.listdir(RAW_VIDEO_DIR) if os.path.isdir(os.path.join(RAW_VIDEO_DIR, d))])
    print(f"Classes: {classes}")

    quit_all = False
    with mp_holistic.Holistic(static_image_mode=False, model_complexity=1, enable_segmentation=False) as holistic:
        for class_name in classes:
            if quit_all:
                break
            class_dir = os.path.join(RAW_VIDEO_DIR, class_name)
            out_class_dir = os.path.join(OUTPUT_DIR, class_name)
            os.makedirs(out_class_dir, exist_ok=True)

            videos = sorted([f for f in os.listdir(class_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))])
            for vid in tqdm(videos, desc=f"Class {class_name}"):
                video_path = os.path.join(class_dir, vid)
                try:
                    result = process_video(video_path, holistic, class_name)
                    if result is None:
                        continue
                    if isinstance(result, str) and result == 'QUIT':
                        quit_all = True
                        break
                    np.save(os.path.join(out_class_dir, os.path.splitext(vid)[0] + '.npy'), result)
                except Exception as e:
                    print(f"  [Error] {vid}: {e}")

    if SHOW_PREVIEW:
        cv2.destroyAllWindows()
    print("Done.")

if __name__ == "__main__":
    main()
