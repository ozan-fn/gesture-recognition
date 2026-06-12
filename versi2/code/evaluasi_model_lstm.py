import os
import sys

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# ==========================================
# 1. LOAD DATASET & MODEL
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, '..', 'dataset')
# MODEL_NAME = 'bisindo_model.h5'
# model_path = os.path.join(BASE_DIR, MODEL_NAME)
model_path = "/mnt/data/Code/gesture-recognition/bisindo_model.h5"

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

label_map = {label: num for num, label in enumerate(DAFTAR_GERAKAN)}

sequences, labels = [], []

print(f"=== Memuat File .npy untuk {len(DAFTAR_GERAKAN)} Gerakan ===")
for gerakan in DAFTAR_GERAKAN:
    folder_gerakan_path = os.path.join(DATASET_DIR, gerakan)

    if not os.path.exists(folder_gerakan_path):
        print(f"⚠️ Folder '{gerakan}' tidak ditemukan. Dilewati...")
        continue

    for video_idx in range(50):
        window = []
        target_folder = os.path.join(folder_gerakan_path, str(video_idx))
        if not os.path.exists(target_folder):
            continue

        # for frame_idx in range(30):
        #     npy_file = os.path.join(target_folder, f"{frame_idx}.npy")
        #     if os.path.exists(npy_file):
        #         window.append(np.load(npy_file))
        #     else:
        #         # window.append(np.zeros(1662))
        #         data = np.load(npy_file)
        #         window.append(data[:96])
        # 
        for frame_idx in range(30):
                    npy_file = os.path.join(target_folder, f"{frame_idx}.npy")
                    if os.path.exists(npy_file):
                        res = np.load(npy_file)
                        # Ambil hanya 96 fitur pertama agar cocok dengan model
                        window.append(res[:96]) 
                    else:
                        window.append(np.zeros(96))

        sequences.append(window)
        labels.append(label_map[gerakan])

if len(sequences) == 0:
    print("❌ Tidak ada data .npy yang ditemukan untuk evaluasi.")
    sys.exit(1)

X = np.array(sequences)
y = to_categorical(labels).astype(int)

# Split Dataset: 70% Training dan 30% Testing (sama seperti training)
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.3, stratify=labels, random_state=42
)

print(f"=== Memuat Model: {os.path.basename(model_path)} ===")
model = load_model(model_path)

# ==========================================
# 2. EVALUASI
# ==========================================

print("\n=== Evaluasi Model pada Data Uji ===")
res_eval = model.evaluate(X_test, y_test, verbose=0)
print(f"Loss pada Data Uji    : {res_eval[0]:.4f}")
print(f"Akurasi pada Data Uji : {res_eval[1]*100:.2f}%")

predictions = model.predict(X_test, verbose=0)
y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(predictions, axis=1)

print("\n=== Laporan Klasifikasi ===")
print(classification_report(y_true, y_pred, target_names=DAFTAR_GERAKAN))

print("\n=== Confusion Matrix ===")
cm = confusion_matrix(y_true, y_pred)
print(cm)
