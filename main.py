!pip install coqui-tts
!pip install torchcodec

import torch
from TTS.api import TTS

# 1. Select Device (GPU is King, CPU is Peasant)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")

# 2. Init Model
# This will download the XTTS-v2 model (~2GB) automatically on first run
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# 3. The Script you want to read
text_to_speak = """
You know what a diamond looks like. <break time="0.5s" /> It is a crystal that repeats a pattern in space.

But imagine a crystal... that repeats its pattern in time.

This is a Time Crystal. <break time="0.6s" /> And until recently? Physics said they were impossible.

You see... normal objects need energy to keep moving. But a Time Crystal is like a bowl of jelly... that jiggles forever... without ever losing energy. <break time="0.4s" /> It breaks the laws of thermodynamics.

Google’s Quantum Computer actually created one. <break time="0.5s" /> And here is where it gets scary.

Scientists believe these crystals are the key to stable Quantum Computing. <break time="0.3s" /> They could allow us to simulate phases of matter that don't exist in our universe... or even model dimensions beyond our own.

It is a glitch in reality. <break time="0.5s" /> A perpetual motion machine... that actually works.

So... <break time="0.3s" /> You know what a diamond looks like...
"""

# 4. Generate Audio (Cloning "brian_reference.wav")
# 'split_sentences=True' helps with long pauses and better flow
tts.tts_to_file(
    text=text_to_speak,
    speaker_wav="sample.mp3", # <--- Your 6s sample of Brian
    language="en",
    file_path="output_cloned_voice.wav",
    split_sentences=True
)

print("Done! Audio saved as output_cloned_voice.wav")
