from faster_whisper import WhisperModel
from pathlib import Path
import os

project_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(
    project_dir, "files", "i'm tired (markiplier)", "i'm tired.mp3"
)

print("pathing:", file_path)

model_size = "small"

# Run on GPU with FP16
# model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
model = WhisperModel(model_size, device="cpu", compute_type="int8")

print(file_path)
segments, info = model.transcribe(file_path, beam_size=5)

print(
    "Detected language '%s' with probability %f"
    % (info.language, info.language_probability)
)

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
