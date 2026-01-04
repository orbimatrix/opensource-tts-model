# Local AI Voice Cloner (XTTS-v2 Pipeline)

This project is a local, offline Python pipeline for generating high-quality AI voiceovers using the **XTTS-v2** model. It includes custom engineering for realistic pauses, emotional mapping, and post-processing (deep voice effects + room tone) to avoid the "AI Slop" sound.

## 🌟 Features
* **Offline Generation:** Runs entirely on your machine (no API costs).
* **Smart Pauses:** Uses SSML-style `<break>` tags to create dramatic silences.
* **Audio Post-Processing:** Automatically adds "Room Tone" (noise floor) and applies compression so the audio doesn't sound like "digital zero."
* **Deep Voice Filter:** Optional pitch-shifting for cinematic/horror narration.

## 🛠️ Prerequisites

* **Python 3.9+**
* **FFmpeg:** Required for audio processing.
    * *Windows:* `winget install ffmpeg`
    * *Mac:* `brew install ffmpeg`
    * *Linux:* `sudo apt install ffmpeg`
* **NVIDIA GPU (Optional but Recommended):** Running on CPU is possible but slow.

## 📦 Installation

1.  **Clone/Download this repository.**
2.  **Install dependencies:**
    ```bash
    pip install torch TTS pydub
    ```
    *(Note: If you have a GPU, ensure you install the CUDA version of PyTorch first).*

## 🎤 Setup: The Reference Audio

The quality of the voice depends **100%** on your reference file.

1.  Record a **6-10 second** audio clip of a voice (or yourself).
2.  **Crucial:** The emotion in the recording must match the emotion you want in the result.
    * *For Horror/Space:* Whisper intensely, breathe heavily.
    * *For Gaming:* Speak loudly and energetically.
3.  Save this file as `sample.mp3` in the project folder.

## 🚀 Usage

1.  Open `main.py` and paste your script into the `script` variable.
2.  Use `<break time="0.5" />` tags to control pacing.
3.  Run the script:
    ```bash
    python main.py
    ```
4.  The final audio will be saved as `upload_ready_audio.wav`.

## 📄 The Code (`main.py`)

Save the following code as `main.py` in the same folder:

```python
import torch
from TTS.api import TTS
from pydub import AudioSegment
from pydub.generators import WhiteNoise
from pydub.effects import compress_dynamic_range
import re
import os

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REFERENCE_AUDIO = "sample.mp3"  # Ensure this file exists!
OUTPUT_FILE = "upload_ready_audio.wav"
DEEP_VOICE_FACTOR = -2.0  # -2.0 semitones for deeper voice. Set to 0 for normal.

# --- INIT MODEL ---
print(f"Loading XTTS on {DEVICE}...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE)

# --- YOUR SCRIPT ---
# Use <break time="0.5" /> for pauses.
script = """
You know what a diamond looks like. <break time="0.8" /> It is a crystal that repeats a pattern in space.

But imagine a crystal... that repeats its pattern in time.

This is a Time Crystal. <break time="1.0" /> And until recently? Physics said they were impossible.

You see... normal objects need energy to keep moving. But a Time Crystal is like a bowl of jelly... that jiggles forever... without ever losing energy. <break time="0.5" /> It breaks the laws of thermodynamics.

Google’s Quantum Computer actually created one. <break time="0.8" /> And here is where it gets scary.

It is a glitch in reality. <break time="1.0" /> A perpetual motion machine... that actually works.
"""

def generate_with_pauses(text, ref_audio):
    parts = re.split(r'(<break time="[\d\.]+" />)', text)
    combined_audio = AudioSegment.empty()
    
    for part in parts:
        if not part.strip(): continue
            
        break_match = re.match(r'<break time="([\d\.]+)" />', part)
        
        if break_match:
            seconds = float(break_match.group(1))
            print(f"--> Adding {seconds}s silence...")
            silence = AudioSegment.silent(duration=seconds * 1000)
            combined_audio += silence
        else:
            print(f"Generating: '{part[:20]}...'")
            temp_file = "temp_chunk.wav"
            tts.tts_to_file(
                text=part.strip(),
                speaker_wav=ref_audio,
                language="en",
                file_path=temp_file,
                speed=0.9,         # Slightly slower for dramatic effect
                temperature=0.75,  # Higher = more emotion
                repetition_penalty=2.0
            )
            chunk = AudioSegment.from_wav(temp_file)
            combined_audio += chunk
            os.remove(temp_file)
            
    return combined_audio

# --- MAIN PIPELINE ---
if __name__ == "__main__":
    print("Starting Generation...")
    raw_audio = generate_with_pauses(script, REFERENCE_AUDIO)

    print("Applying Post-Processing (Deep Voice + Room Tone)...")
    
    # 1. Pitch Shift (Deep Voice)
    if DEEP_VOICE_FACTOR != 0:
        new_sample_rate = int(raw_audio.frame_rate * (2.0 ** (DEEP_VOICE_FACTOR / 12.0)))
        raw_audio = raw_audio._spawn(raw_audio.raw_data, overrides={'frame_rate': new_sample_rate})
        raw_audio = raw_audio.set_frame_rate(44100)

    # 2. Add Room Tone (Fixes "Dead Air")
    room_tone = WhiteNoise().to_audio_segment(duration=len(raw_audio), volume=-55.0)
    room_tone = room_tone.low_pass_filter(500) # Muffle the noise
    final_mix = raw_audio.overlay(room_tone)

    # 3. Compression (Even volume)
    final_mix = compress_dynamic_range(final_mix, threshold=-20.0, ratio=4.0)

    final_mix.export(OUTPUT_FILE, format="wav")
    print(f"✅ Success! Saved to {OUTPUT_FILE}")
