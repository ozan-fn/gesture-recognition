import os
import shutil
import random

# Video recommendations with their paths
video_recommendations = {
    "Apa": "raw_videos/raw_video/Apa/BISINDO_Apa_003.mp4",
    "Apa Kabar": None,  # No correct video found
    "Bagaimana": "raw_videos/raw_video/Bagaimana/BISINDO_Bagaimana_003.mp4",
    "Baik": "raw_videos/raw_video/Baik/BISINDO_Baik_004.mp4",
    "Belajar": "raw_videos/raw_video/Belajar/BISINDO_Belajar_002.mp4",
    "Berapa": "raw_videos/raw_video/Berapa/BISINDO_Berapa_003.mp4",
    "Berdiri": "raw_videos/raw_video/Berdiri/BISINDO_Berdiri_005.mp4",
    "Bingung": "raw_videos/raw_video/Bingung/BISINDO_Bingung_002.mp4",
    "Dia": "raw_videos/raw_video/Dia/BISINDO_Dia_004.mp4",
    "Dimana": "raw_videos/raw_video/Dimana/BISINDO_Dimana_003.mp4",
    "Duduk": "raw_videos/raw_video/Duduk/BISINDO_Duduk_003.mp4",
    "Halo": "raw_videos/raw_video/Halo/BISINDO_Halo_004.mp4",
    "Kalian": "raw_videos/raw_video/Kalian/BISINDO_Kalian_001.mp4",
    "Kami": "raw_videos/raw_video/Kami/BISINDO_Kami_002.mp4",
    "Kamu": "raw_videos/raw_video/Kamu/BISINDO_Kamu_002.mp4",
    "Kapan": "raw_videos/raw_video/Kapan/BISINDO_Kapan_005.mp4",
    "Kemana": "raw_videos/raw_video/Kemana/BISINDO_Kemana_005.mp4",
    "Kita": "raw_videos/raw_video/Kita/BISINDO_Kita_005.mp4",
    "Makan": "raw_videos/raw_video/Makan/BISINDO_Makan_002.mp4",
    "Mandi": "raw_videos/raw_video/Mandi/BISINDO_Mandi_005.mp4",
    "Marah": "raw_videos/raw_video/Marah/BISINDO_Marah_001.mp4",
    "Melihat": "raw_videos/raw_video/Melihat/BISINDO_Melihat_003.mp4",
    "Membaca": "raw_videos/raw_video/Membaca/BISINDO_Membaca_001.mp4",
    "Menulis": "raw_videos/raw_video/Menulis/BISINDO_Menulis_005.mp4",
    "Mereka": "raw_videos/raw_video/Mereka/BISINDO_Mereka_004.mp4",
    "Minum": "raw_videos/raw_video/Minum/BISINDO_Minum_004.mp4",
    "Pendek": "raw_videos/raw_video/Pendek/BISINDO_Pendek_002.mp4",
    "Ramah": "raw_videos/raw_video/Ramah/BISINDO_Ramah_002.mp4",
    "Sabar": "raw_videos/raw_video/Sabar/BISINDO_Sabar_003.mp4",
    "Saya": "raw_videos/raw_video/Saya/BISINDO_Saya_001.mp4",
    "Sedih": "raw_videos/raw_video/Sedih/BISINDO_Sedih_005.mp4",
    "Selamat Malam": None,  # No correct video found
    "Selamat Pagi": None,  # No correct video found
    "Selamat Siang": None,  # No correct video found
    "Selamat Sore": None,  # No correct video found
    "Senang": "raw_videos/raw_video/Senang/BISINDO_Senang_001.mp4",
    "Siapa": "raw_videos/raw_video/Siapa/BISINDO_Siapa_001.mp4",
    "Terima Kasih": "raw_videos/raw_video/Terima Kasih/BISINDO_Terima Kasih_004.mp4",
    "Tidur": "raw_videos/raw_video/Tidur/BISINDO_Tidur_004.mp4",
    "Tinggi": "raw_videos/raw_video/Tinggi/BISINDO_Tinggi_004.mp4",
}

# Output directory for Streamlit
OUTPUT_DIR = "streamlit_videos"
SOURCE_DIR = "raw_videos/raw_video"

def get_random_video(class_name, source_dir):
    """Get a random video from the class folder."""
    class_dir = os.path.join(source_dir, class_name)
    if not os.path.exists(class_dir):
        print(f"  ⚠️  Directory not found: {class_dir}")
        return None
    
    videos = [f for f in os.listdir(class_dir) if f.endswith(('.mp4', '.avi', '.mov'))]
    if not videos:
        print(f"  ⚠️  No videos found in: {class_dir}")
        return None
    
    random_video = random.choice(videos)
    return os.path.join(class_dir, random_video)

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📁 Creating directory: {OUTPUT_DIR}\n")
    
    success_count = 0
    random_count = 0
    not_found_count = 0
    
    for class_name, video_path in video_recommendations.items():
        # Clean filename for output
        safe_filename = class_name.replace(" ", "_") + ".mp4"
        output_path = os.path.join(OUTPUT_DIR, safe_filename)
        
        # If recommended video exists, copy it
        if video_path and os.path.exists(video_path):
            shutil.copy2(video_path, output_path)
            print(f"✅ {class_name:<20} → {os.path.basename(video_path)}")
            success_count += 1
        
        # If no recommendation, get random video from class folder
        else:
            random_video = get_random_video(class_name, SOURCE_DIR)
            if random_video and os.path.exists(random_video):
                shutil.copy2(random_video, output_path)
                print(f"🎲 {class_name:<20} → {os.path.basename(random_video)} (random)")
                random_count += 1
            else:
                print(f"❌ {class_name:<20} → NOT FOUND")
                not_found_count += 1
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"✅ Recommended videos copied: {success_count}")
    print(f"🎲 Random videos selected: {random_count}")
    print(f"❌ Videos not found: {not_found_count}")
    print(f"📁 Output directory: {OUTPUT_DIR}/")
    print("="*60)

if __name__ == "__main__":
    main()
