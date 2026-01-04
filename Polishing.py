from pydub import AudioSegment
from pydub.generators import WhiteNoise

# Load your generated voice
voice = AudioSegment.from_wav("final_deep_output.wav")

# 1. GENERATE ROOM TONE (Brown Noise is better than White Noise for "warmth")
# We generate a faint hum that matches the length of your voice clip
# 'sample_rate=16000' makes it sound lo-fi/radio-like
room_tone = WhiteNoise().to_audio_segment(duration=len(voice), volume=-50.0)
room_tone = room_tone.low_pass_filter(500) # Muffle it so it sounds like empty room air

# 2. OVERLAY THE TONE
# This fills the "Digital Silence" gaps with the room tone
final_mix = voice.overlay(room_tone)

# 3. COMPRESS IT (The "YouTuber" Effect)
# This boosts the quiet parts (the end of sentences) so they don't disappear
from pydub.effects import compress_dynamic_range
final_mix = compress_dynamic_range(final_mix, threshold=-20.0, ratio=4.0)

# Export
final_mix.export("upload_ready_audio.wav", format="wav")
print("Saved upload_ready_audio.wav with Room Tone added.")
