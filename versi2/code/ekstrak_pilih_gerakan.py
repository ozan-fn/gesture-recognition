import os

import cv2
import mediapipe as mp
import numpy as np

# 1. Inisialisasi MediaPipe Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 2. Definisikan daftar gerakan sesuai folder yang kamu miliki
# DAFTAR_GERAKAN = np.array(['Berdiri', 'Bingung', 'Dia', 'Dimana', 'Duduk'])
DAFTAR_GERAKAN = np.array(['Apa', 'Apa Kabar', 'Bagaimana', 'Baik', 'Belajar', 'Berapa', 'Berdiri', 'Bingung']) 

# Deteksi lokasi folder 'code' tempat skrip ini berada
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Naik satu level ke folder 'psc_project', lalu masuk ke 'dataset'
DATASET_DIR = os.path.join(BASE_DIR, '..', 'dataset')

# Fungsi untuk mengekstrak dan meratakan (flatten) koordinat menjadi 1 array tunggal
def extract_keypoints(results):
    # Wajah: 468 titik x 3 (x, y, z) = 1404 angka (Sesuai spesifikasi jurnal)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)

    # Pose/Tubuh: 33 titik x 4 (x, y, z, visibility) = 132 angka
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)

    # Tangan Kiri: 21 titik x 3 (x, y, z) = 63 angka
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)

    # Tangan Kanan: 21 titik x 3 (x, y, z) = 63 angka
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)

    # Gabungkan semua menjadi satu baris array (Total: 1662 angka per frame)
    return np.concatenate([face, pose, lh, rh])

print("=======================================================")
print(f"Memulai Ekstraksi Fitur Massal untuk {len(DAFTAR_GERAKAN)} Gerakan")
print("=======================================================")

# Loop bersarang (Nested Loop): Mengitari setiap gerakan, lalu mengitari ke-50 videonya
for gerakan in DAFTAR_GERAKAN:
    print(f"\n📂 [GERAKAN] Menjelajahi folder gerakan: {gerakan.upper()}")
    folder_gerakan_path = os.path.join(DATASET_DIR, gerakan)

    # Validasi apakah folder gerakan tersebut memang ada secara fisik
    if not os.path.exists(folder_gerakan_path):
        print(f"⚠️ Folder '{gerakan}' tidak ditemukan di {DATASET_DIR}. Dilewati...")
        continue

    for video_idx in range(50):
        # Format penamaan file video (0 -> "001", 49 -> "050")
        video_num_padded = f"{video_idx + 1:03d}"
        video_name = f"BISINDO_{gerakan}_{video_num_padded}.mp4"
        video_file_path = os.path.join(folder_gerakan_path, video_name)

        # Tentukan folder angka tujuan penyimpanannya (0 sampai 49)
        target_folder_path = os.path.join(folder_gerakan_path, str(video_idx))

        # Cek apakah file video fisiknya tersedia
        if not os.path.exists(video_file_path):
            continue

        print(f"   📹 Memproses: {video_name} -> {gerakan}/{video_idx}/")

        # Buat folder target otomatis jika belum ada
        os.makedirs(target_folder_path, exist_ok=True)

        # Membaca video via OpenCV
        cap = cv2.VideoCapture(video_file_path)
        frame_idx = 0

        # Batasi pengambilan data maksimal hingga 30 frame per video sekuensial
        while cap.isOpened() and frame_idx < 60:
            ret, frame = cap.read()
            if not ret:
                break

            # Konversi BGR ke RGB untuk konsumsi MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Ekstraksi fitur koordinat utama
            results = holistic.process(rgb_frame)
            keypoints = extract_keypoints(results)

            # Buat file path .npy (0.npy sampai 29.npy)
            npy_file_path = os.path.join(target_folder_path, f"{frame_idx}.npy")

            # Simpan koordinat biner secara aman ke disk
            np.save(npy_file_path, keypoints)

            frame_idx += 1

        cap.release()

        # Zero Padding: Jika video selesai tapi frame kurang dari 30
        while frame_idx < 60:
            print(f"   -> Menambal frame kosong (Padding) di frame ke-{frame_idx}")
            # Buat array berisi angka 0 sebanyak 1662 (ukuran output MediaPipe Anda)
            keypoints = np.zeros(1662)
            npy_file_path = os.path.join(target_folder_path, f"{frame_idx}.npy")
            np.save(npy_file_path, keypoints)
            frame_idx += 1

# Tutup resource MediaPipe setelah semua loop tuntas
holistic.close()
print("\n=======================================================")
print(f"🎉 SUKSES! Seluruh {len(DAFTAR_GERAKAN)} gerakan telah berhasil diekstrak menjadi file .npy")
print("=======================================================")
