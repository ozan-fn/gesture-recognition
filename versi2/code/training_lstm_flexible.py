import os
import sys

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

# ==========================================
# 1. TAHAP INPUT GERAKAN
# ==========================================

# Deteksi lokasi folder skrip berada
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, '..', 'dataset')

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

# Cek apakah user memberikan argument gerakan
if len(sys.argv) < 2:
    print("Penggunaan: python training_lstm_flexible.py <gerakan1> <gerakan2> ... <gerakanN>")
    print("Atau      : python training_lstm_flexible.py --all")
    print("Contoh 1  : python training_lstm_flexible.py Apa")
    print("Contoh 2  : python training_lstm_flexible.py Apa 'Apa Kabar' Bagaimana")
    print("Contoh 3  : python training_lstm_flexible.py Apa 'Apa Kabar' Bagaimana Baik Belajar")
    sys.exit(1)

# Ambil gerakan dari argument command line
if sys.argv[1] == "--all":
    DAFTAR_GERAKAN = np.array(list_gesture_folders(DATASET_DIR))
    if DAFTAR_GERAKAN.size == 0:
        print(f"⚠️ Tidak ada folder gerakan yang valid di {DATASET_DIR}.")
        sys.exit(1)
else:
    DAFTAR_GERAKAN = np.array(sys.argv[1:])

# Membuat label otomatis (Label Map) berdasarkan daftar gerakan
label_map = {label: num for num, label in enumerate(DAFTAR_GERAKAN)}
print("=== Kamus Pelabelan Otomatis ===")
print(label_map, "\n")

sequences, labels = [], []

print(f"=== Memuat File .npy untuk {len(DAFTAR_GERAKAN)} Gerakan ===")
# Loop untuk membaca seluruh folder gerakan dan file .npy di dalamnya
for gerakan in DAFTAR_GERAKAN:
    folder_gerakan_path = os.path.join(DATASET_DIR, gerakan)

    # Validasi folder gerakan
    if not os.path.exists(folder_gerakan_path):
        print(f"⚠️ Folder '{gerakan}' tidak ditemukan. Dilewati...")
        continue

    print(f"📂 Memuat gerakan: {gerakan}")

    # Looping 50 video sekuensial per gerakan
    for video_idx in range(50):
        window = []
        target_folder = os.path.join(folder_gerakan_path, str(video_idx))

        # Pastikan folder video tersebut ada sebelum dibaca
        if not os.path.exists(target_folder):
            continue

        # Membaca 30 frame sekuensial (.npy) di dalam folder video
        for frame_idx in range(60):
            npy_file = os.path.join(target_folder, f"{frame_idx}.npy")
            if os.path.exists(npy_file):
                res = np.load(npy_file)
                window.append(res)
            else:
                # Jika ada frame yang corrupt/hilang, diisi array kosong (1662 fitur)
                window.append(np.zeros(1662))

        sequences.append(window)
        labels.append(label_map[gerakan])

# Konversi hasil load menjadi Array NumPy
X = np.array(sequences) # Dimensi Fitur: (Total_Video, 30_Frame, 1662_Fitur)
y = to_categorical(labels).astype(int) # Mengubah label ke One-Hot Encoding

print(f"\n✅ Data berhasil dimuat!")
print(f"Bentuk Matriks Fitur (X): {X.shape}")
print(f"Bentuk Matriks Label  (y): {y.shape}\n")

# Split Dataset: 70% Training dan 30% Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=labels, random_state=42)

print(f"Data Latih (X_train): {X_train.shape}")
print(f"Data Uji   (X_test) : {X_test.shape}\n")


# ==========================================
# 2. TAHAP PEMODELAN DATA (ARSITEKTUR LSTM)
# ==========================================

print("=== Membangun Arsitektur LSTM ===")
model = Sequential()

# Layer 1 & 2 LSTM (return_sequences=True karena output dikirim ke layer LSTM berikutnya)
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(60, 1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))

# Layer 3 LSTM (return_sequences=False karena setelah ini masuk ke Dense Layer biasa)
model.add(LSTM(64, return_sequences=False, activation='relu'))

# Fully Connected Layers (Dense Layers)
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))

# Output Layer: Jumlah unit otomatis mengikuti panjang DAFTAR_GERAKAN
model.add(Dense(DAFTAR_GERAKAN.shape[0], activation='softmax'))

# Kompilasi Model menggunakan Optimizer Adam dan Categorical Crossentropy
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Menampilkan Summary struktur parameter model di terminal
model.summary()


# ==========================================
# 3. TAHAP PROSES PELATIHAN (TRAINING)
# ==========================================

print("\n=== Memulai Proses Training (250 Epoch) ===")
# Setup TensorBoard untuk memantau grafik loss dan akurasi nantinya
log_dir = os.path.join(BASE_DIR, 'Logs')
tb_callback = TensorBoard(log_dir=log_dir)

# Jalankan proses training model
model.fit(X_train, y_train, epochs=250, callbacks=[tb_callback], validation_data=(X_test, y_test))


# ==========================================
# 4. MENYIMPAN MODEL HASIL TRAINING
# ==========================================

# Buat nama file model berdasarkan gerakan yang di-train
# gerakan_str = '_'.join(DAFTAR_GERAKAN).replace(' ', '_')
gerakan_str = 'gerakan-all-60fps'
model_save_path = os.path.join(BASE_DIR, f'model_lstm_{gerakan_str}.keras')
model.save(model_save_path)
print(f"\n🎉 TRAINING SELESAI! Model pintar kamu telah disimpan di: {model_save_path}")
