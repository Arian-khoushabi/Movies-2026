import whisper
from pathlib import Path
import json

base_path = Path('E:/myRepository/movie_audio_dataset')
model = whisper.load_model("medium", device='cpu')
print('model loaded successfully')

with open('E:/myRepository/movie_audio_dataset/data/metadata.json', 'r', encoding='utf-8') as f:
  metadata = json.load(f)
  print('metadata loaded successfully')

counter = 0
print('starting transcribtion')
for records in metadata.values():
  for record in records:
    if record["transcribe"] is None:
      audio_path = base_path / record["audio_path"]
      if audio_path.exists() and audio_path.stat().st_size > 0:
        transcribtion = model.transcribe(str(audio_path))["text"]
        record["transcribe"] = transcribtion
        counter += 1
        if counter % 25 == 0:
          print(f"progressed {counter}")
          with open('E:/myRepository/movie_audio_dataset/data/metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=4, ensure_ascii=False)
with open('E:/myRepository/movie_audio_dataset/data/metadata.json', 'w', encoding='utf-8') as f:
  json.dump(metadata, f, indent=4, ensure_ascii=False)
  print('metadata saved successfully')