import torch
from TTS.api import TTS
from pydub import AudioSegment
import re
import os

# --- CONFIGURATION ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REFERENCE_AUDIO = "sample.mp3" # Your 6s sample (CRITICAL: Use an emotional sample!)
OUTPUT_FILE = "final_deep_output.wav"
DEEP_VOICE_FACTOR = -2.0  # Lower pitch by 2 semitones (makes it deeper)

# --- INIT MODEL ---
print(f"Loading XTTS on {DEVICE}...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(DEVICE)

# --- THE SCRIPT (With functional break tags) ---
script = """
You know what a diamond looks like. <break time="0.8" /> It is a crystal that repeats a pattern in space.

But imagine a crystal... that repeats its pattern in time.

This is a Time Crystal. <break time="1.0" /> And until recently? Physics said they were impossible.

You see... normal objects need energy to keep moving. But a Time Crystal is like a bowl of jelly... that jiggles forever... without ever losing energy. <break time="0.5" /> It breaks the laws of thermodynamics.

Google’s Quantum Computer actually created one. <break time="0.8" /> And here is where it gets scary.

Scientists believe these crystals are the key to stable Quantum Computing. <break time="0.4" /> They could allow us to simulate phases of matter that don't exist in our universe... or even model dimensions beyond our own.

It is a glitch in reality. <break time="1.0" /> A perpetual motion machine... that actually works.

So... <break time="0.5" /> You know what a diamond looks like...
"""

def generate_with_pauses(text, ref_audio):
    # 1. Parse the text into chunks based on <break> tags
    # Regex finds: <break time="0.5" />
    parts = re.split(r'(<break time="[\d\.]+" />)', text)
    
    combined_audio = AudioSegment.empty()
    
    for part in parts:
        if not part.strip():
            continue
            
        # Check if this part is a break tag
        break_match = re.match(r'<break time="([\d\.]+)" />', part)
        
        if break_match:
            # Create silence
            seconds = float(break_match.group(1))
            print(f"--> Adding {seconds}s silence...")
            silence = AudioSegment.silent(duration=seconds * 1000)
            combined_audio += silence
        else:
            # Generate Speech for text chunk
            print(f"Generating: '{part[:30]}...'")
            temp_file = "temp_chunk.wav"
            
            # TWEAK THESE PARAMETERS FOR EMOTION
            tts.tts_to_file(
                text=part.strip(),
                speaker_wav=ref_audio,
                language="en",
                file_path=temp_file,
                speed=1.0,         # Slow down slightly for dramatic effect? Try 0.9
                temperature=0.75,  # Higher = more emotional variance (0.7-0.8 is sweet spot)
                repetition_penalty=2.0
            )
            
            chunk = AudioSegment.from_wav(temp_file)
            combined_audio += chunk
            os.remove(temp_file)
            
    return combined_audio

# --- RUN THE PIPELINE ---
print("Starting Generation...")
final_audio = generate_with_pauses(script, REFERENCE_AUDIO)

# --- POST PROCESSING (The "Deep Voice" Effect) ---
print("Applying Deep Voice Filter...")
# Decrease pitch (sample rate trick)
new_sample_rate = int(final_audio.frame_rate * (2.0 ** (DEEP_VOICE_FACTOR / 12.0)))
low_pitch_audio = final_audio._spawn(final_audio.raw_data, overrides={'frame_rate': new_sample_rate})
low_pitch_audio = low_pitch_audio.set_frame_rate(final_audio.frame_rate)

# Export
low_pitch_audio.export(OUTPUT_FILE, format="wav")
print(f"Success! Saved to {OUTPUT_FILE}")
