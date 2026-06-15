import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pickle
import json
from datetime import datetime

# === ANSI COLOR CODES ===
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# === CONFIGURATION ===
MODEL_PATH          = 'bisindo_model.h5'
LABELS_PATH         = 'class_names.pkl'
SCALER_PATH         = 'scaler.pkl'
SEQUENCE_LENGTH     = 30
PREDICTION_THRESHOLD = 0.80
EMA_ALPHA           = 0.2

POSE_LANDMARKS_IDX = [11, 12, 13, 14, 15, 16]

mp_holistic = mp.solutions.holistic  # type: ignore

# ── Normalisasi ───────────────────────────────────────────────────────────

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

    return np.concatenate([pose, lh, rh])

# ── Test Video ────────────────────────────────────────────────────────────

def test_video(video_path, expected_label, model, scaler, class_names, holistic):
    """Test a single video and return prediction result"""
    normalizer = Normalizer()
    cap = cv2.VideoCapture(video_path)
    
    sequence_buffer = []
    hand_seen_streak = 0
    recording_started = False
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)
        
        # Check if hands are present
        hands_present = (
            results.left_hand_landmarks is not None or
            results.right_hand_landmarks is not None
        )
        
        if hands_present:
            hand_seen_streak += 1
        else:
            hand_seen_streak = 0
        
        # Start recording after hands detected for 5 frames
        if not recording_started and hand_seen_streak >= 5:
            recording_started = True
        
        # Record keypoints when recording started
        if recording_started:
            keypoints = extract_keypoints(results, frame.shape, normalizer)
            sequence_buffer.append(keypoints)
            
            # Stop when we have enough frames
            if len(sequence_buffer) >= SEQUENCE_LENGTH:
                break
    
    cap.release()

    if len(sequence_buffer) < SEQUENCE_LENGTH:
        return None, 0.0, "Insufficient frames"

    # Predict
    seq_arr = np.array(sequence_buffer[:SEQUENCE_LENGTH])
    seq_scaled = scaler.transform(seq_arr)
    input_data = np.expand_dims(seq_scaled, axis=0)
    preds = model.predict(input_data, verbose=0)[0]

    conf = float(np.max(preds))
    pred_class = class_names[int(np.argmax(preds))]

    # Check if prediction is correct
    is_correct = (pred_class == expected_label and conf >= PREDICTION_THRESHOLD)

    return pred_class, conf, "Correct" if is_correct else "Wrong"

# ── Main Testing ──────────────────────────────────────────────────────────

def main():
    print(Colors.BOLD + "="*70 + Colors.ENDC)
    print(Colors.HEADER + Colors.BOLD + "BISINDO VIDEO TESTING - Find Best Videos for Each Word" + Colors.ENDC)
    print(Colors.BOLD + "="*70 + Colors.ENDC)
    print()

    # Load model
    print(Colors.CYAN + "[1/4] Loading model and artifacts..." + Colors.ENDC)
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, 'rb') as f:
        class_names = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    print(Colors.GREEN + f"✓ Model loaded. Classes: {class_names}" + Colors.ENDC)
    print()

    # Scan raw_videos folder
    print(Colors.CYAN + "[2/4] Scanning raw_videos folder..." + Colors.ENDC)
    raw_video_path = 'raw_videos/raw_video'

    if not os.path.exists(raw_video_path):
        print(Colors.RED + f"✗ Error: Folder '{raw_video_path}' not found!" + Colors.ENDC)
        return

    word_folders = [f for f in os.listdir(raw_video_path)
                    if os.path.isdir(os.path.join(raw_video_path, f))]
    word_folders.sort()
    print(Colors.GREEN + f"✓ Found {len(word_folders)} word folders" + Colors.ENDC)
    print()

    # Prepare results storage
    results = {
        'timestamp': datetime.now().isoformat(),
        'total_words': len(word_folders),
        'words': {}
    }

    correct_videos = {}
    incorrect_videos = {}

    # Test each word folder
    print(Colors.CYAN + "[3/4] Testing videos..." + Colors.ENDC)
    print(Colors.BOLD + "-"*70 + Colors.ENDC)

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        enable_segmentation=False
    ) as holistic:

        for word_idx, word in enumerate(word_folders, 1):
            word_path = os.path.join(raw_video_path, word)
            videos = [f for f in os.listdir(word_path)
                     if f.endswith(('.mp4', '.avi', '.mov'))]

            if not videos:
                print(Colors.YELLOW + f"[{word_idx}/{len(word_folders)}] {word}: No videos found - SKIP" + Colors.ENDC)
                continue

            print(Colors.BLUE + Colors.BOLD + f"[{word_idx}/{len(word_folders)}] {word}: Testing {len(videos)} video(s)..." + Colors.ENDC)

            word_results = {
                'total_videos': len(videos),
                'correct_videos': [],
                'incorrect_videos': []
            }

            correct_count = 0

            for video_file in videos:
                video_path = os.path.join(word_path, video_file)
                pred_class, conf, status = test_video(
                    video_path, word, model, scaler, class_names, holistic
                )

                video_info = {
                    'filename': video_file,
                    'predicted': pred_class,
                    'confidence': float(conf) if conf else 0.0,
                    'status': status
                }

                if status == "Correct":
                    correct_count += 1
                    word_results['correct_videos'].append(video_info)
                    if word not in correct_videos:
                        correct_videos[word] = []
                    correct_videos[word].append(video_file)
                    print(Colors.GREEN + f"  ✓ {video_file}: {pred_class} ({conf:.2%}) - CORRECT" + Colors.ENDC)
                elif status == "Wrong":
                    word_results['incorrect_videos'].append(video_info)
                    if word not in incorrect_videos:
                        incorrect_videos[word] = []
                    incorrect_videos[word].append({
                        'file': video_file,
                        'predicted': pred_class,
                        'confidence': conf
                    })
                    print(Colors.RED + f"  ✗ {video_file}: {pred_class} ({conf:.2%}) - WRONG" + Colors.ENDC)
                else:
                    print(Colors.YELLOW + f"  - {video_file}: {status}" + Colors.ENDC)

            # Summary for this word
            accuracy = (correct_count / len(videos) * 100) if videos else 0
            if accuracy == 100:
                accuracy_color = Colors.GREEN
            elif accuracy >= 70:
                accuracy_color = Colors.CYAN
            elif accuracy >= 30:
                accuracy_color = Colors.YELLOW
            else:
                accuracy_color = Colors.RED
            print(accuracy_color + f"  → Accuracy: {correct_count}/{len(videos)} ({accuracy:.1f}%)" + Colors.ENDC)

            results['words'][word] = word_results
            print()

    print(Colors.BOLD + "-"*70 + Colors.ENDC)
    print()

    # Save results
    print(Colors.CYAN + "[4/4] Saving results..." + Colors.ENDC)

    output_file = 'video_test_results.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(Colors.GREEN + f"✓ Detailed results saved to: {output_file}" + Colors.ENDC)

    # Create summary
    summary_file = 'correct_videos_summary.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("CORRECT VIDEOS SUMMARY\n")
        f.write("="*70 + "\n\n")

        if correct_videos:
            for word in sorted(correct_videos.keys()):
                f.write(f"{word}:\n")
                for video in correct_videos[word]:
                    f.write(f"  - {video}\n")
                f.write("\n")
        else:
            f.write("No correct videos found.\n")

        f.write("\n" + "="*70 + "\n")
        f.write("INCORRECT VIDEOS SUMMARY\n")
        f.write("="*70 + "\n\n")

        if incorrect_videos:
            for word in sorted(incorrect_videos.keys()):
                f.write(f"{word}:\n")
                for item in incorrect_videos[word]:
                    f.write(f"  - {item['file']}: predicted as '{item['predicted']}' "
                           f"({item['confidence']:.2%})\n")
                f.write("\n")
        else:
            f.write("All videos predicted correctly!\n")

    print(Colors.GREEN + f"✓ Summary saved to: {summary_file}" + Colors.ENDC)
    print()

    # Print summary statistics
    print(Colors.BOLD + "="*70 + Colors.ENDC)
    print(Colors.HEADER + Colors.BOLD + "SUMMARY STATISTICS" + Colors.ENDC)
    print(Colors.BOLD + "="*70 + Colors.ENDC)

    total_correct = sum(len(v) for v in correct_videos.values())
    total_incorrect = sum(len(v) for v in incorrect_videos.values())
    total_tested = total_correct + total_incorrect

    print(f"Total words tested: {len(word_folders)}")
    print(f"Total videos tested: {total_tested}")
    if total_tested > 0:
        print(Colors.GREEN + f"Correct predictions: {total_correct} ({total_correct/total_tested*100:.1f}%)" + Colors.ENDC)
        print(Colors.RED + f"Incorrect predictions: {total_incorrect} ({total_incorrect/total_tested*100:.1f}%)" + Colors.ENDC)
    print()

    print(f"Words with all correct videos: {sum(1 for w in word_folders if w in correct_videos and w not in incorrect_videos)}")
    print(f"Words with some incorrect videos: {len(incorrect_videos)}")
    print()

    # Best video recommendations
    print(Colors.BOLD + "="*70 + Colors.ENDC)
    print(Colors.HEADER + Colors.BOLD + "BEST VIDEO RECOMMENDATIONS (Highest Confidence)" + Colors.ENDC)
    print(Colors.BOLD + "="*70 + Colors.ENDC)
    print()

    best_videos_file = 'best_videos_recommendation.txt'
    with open(best_videos_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("BEST VIDEO RECOMMENDATIONS (Keep These Videos)\n")
        f.write("="*70 + "\n\n")

        words_with_best = 0
        words_without_video = []

        for word in sorted(word_folders):
            if word in results['words']:
                word_data = results['words'][word]
                correct_vids = word_data['correct_videos']

                if correct_vids:
                    # Sort by confidence, get highest
                    best = max(correct_vids, key=lambda x: x['confidence'])
                    words_with_best += 1

                    video_path = os.path.join(raw_video_path, word, best['filename'])
                    recommendation = f"{word}: {video_path} (confidence: {best['confidence']:.2%})"
                    print(Colors.GREEN + f"  {recommendation}" + Colors.ENDC)
                    f.write(recommendation + "\n")
                else:
                    # No correct video found
                    words_without_video.append(word)
                    print(Colors.RED + f"  {word}: ⚠️  NO CORRECT VIDEO FOUND" + Colors.ENDC)
                    f.write(f"{word}: NO CORRECT VIDEO FOUND\n")

        f.write("\n" + "="*70 + "\n")
        f.write("SUMMARY\n")
        f.write("="*70 + "\n")
        f.write(f"Words with best video: {words_with_best}/{len(word_folders)}\n")
        f.write(f"Words without correct video: {len(words_without_video)}\n")

        if words_without_video:
            f.write("\nWords needing attention:\n")
            for w in words_without_video:
                f.write(f"  - {w}\n")

        f.write("\n" + "="*70 + "\n")
        f.write("DELETE COMMANDS (Windows)\n")
        f.write("="*70 + "\n\n")
        f.write("REM Delete all videos except the best ones\n")

        # Generate delete commands for non-best videos
        for word in sorted(word_folders):
            if word in results['words']:
                word_data = results['words'][word]
                all_vids = word_data['correct_videos'] + word_data['incorrect_videos']
                correct_vids = word_data['correct_videos']

                if correct_vids:
                    best = max(correct_vids, key=lambda x: x['confidence'])
                    best_filename = best['filename']

                    # List all other videos to delete
                    for vid in all_vids:
                        if vid['filename'] != best_filename:
                            delete_path = os.path.join(raw_video_path, word, vid['filename'])
                            f.write(f'del "{delete_path}"\n')

    print()
    print(Colors.GREEN + f"✓ Best video recommendations saved to: {best_videos_file}" + Colors.ENDC)
    print()

    if words_without_video:
        print(Colors.YELLOW + "⚠️  WARNING: The following words have NO correct predictions:" + Colors.ENDC)
        for w in words_without_video:
            print(Colors.RED + f"  - {w}" + Colors.ENDC)
        print()

    print(Colors.BOLD + "="*70 + Colors.ENDC)
    print(Colors.CYAN + "RECOMMENDATION: Keep only the videos listed in best_videos_recommendation.txt" + Colors.ENDC)
    print(Colors.CYAN + "Delete other videos to keep only the most accurate one per word." + Colors.ENDC)
    print(Colors.BOLD + "="*70 + Colors.ENDC)

    print(Colors.BOLD + "="*70 + Colors.ENDC)
    print(Colors.GREEN + Colors.BOLD + "Testing complete!" + Colors.ENDC)
    print(Colors.BOLD + "="*70 + Colors.ENDC)

if __name__ == "__main__":
    main()
