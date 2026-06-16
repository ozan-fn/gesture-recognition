"""
Text-to-Speech Helper using Edge-TTS with caching
"""
import os
import asyncio
import hashlib

try:
    import edge_tts  # type: ignore
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False
    print("Warning: edge-tts not installed. Install with: pip install edge-tts")

# Configuration
TTS_CACHE_DIR = "tts_cache"
VOICE = "id-ID-ArdiNeural"  # Indonesian male voice
# Alternative: "id-ID-GadisNeural"  # Indonesian female voice

def get_cache_path(text):
    """Generate cache file path based on text hash."""
    text_hash = hashlib.md5(text.encode()).hexdigest()
    return os.path.join(TTS_CACHE_DIR, f"{text_hash}.mp3")

def text_to_speech(text, output_path=None):
    """
    Convert text to speech and cache the result.
    
    Args:
        text: Text to convert to speech
        output_path: Optional custom output path
        
    Returns:
        Path to the audio file
    """
    if not EDGE_TTS_AVAILABLE:
        return None
    
    # Create cache directory
    os.makedirs(TTS_CACHE_DIR, exist_ok=True)
    
    # Use cache path if no custom output specified
    if output_path is None:
        output_path = get_cache_path(text)
    
    # Check if already cached
    if os.path.exists(output_path):
        return output_path
    
    # Generate TTS
    try:
        asyncio.run(_generate_tts(text, output_path))
        return output_path
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

async def _generate_tts(text, output_path):
    """Async function to generate TTS."""
    if EDGE_TTS_AVAILABLE:
        import edge_tts  # type: ignore
        communicate = edge_tts.Communicate(text, VOICE)
        await communicate.save(output_path)

def preload_common_phrases():
    """Preload TTS for common BISINDO words."""
    common_words = [
        "Apa", "Apa Kabar", "Bagaimana", "Baik", "Belajar",
        "Berapa", "Berdiri", "Bingung", "Dia", "Dimana",
        "Duduk", "Halo", "Kalian", "Kami", "Kamu",
        "Kapan", "Kemana", "Kita", "Makan", "Mandi",
        "Marah", "Melihat", "Membaca", "Menulis", "Mereka",
        "Minum", "Pendek", "Ramah", "Sabar", "Saya",
        "Sedih", "Selamat Malam", "Selamat Pagi", "Selamat Siang",
        "Selamat Sore", "Senang", "Siapa", "Terima Kasih",
        "Tidur", "Tinggi"
    ]
    
    print("Preloading TTS cache...")
    for word in common_words:
        text_to_speech(word)
    print(f"TTS cache ready: {len(common_words)} words cached")

if __name__ == "__main__":
    # Test and preload
    print("Testing edge-tts...")
    test_audio = text_to_speech("Halo, ini adalah tes")
    if test_audio:
        print(f"Test successful: {test_audio}")
        preload_common_phrases()
    else:
        print("TTS not available")
