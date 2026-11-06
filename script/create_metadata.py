from pathlib import Path
import json

PROJECT_DIR = Path(__file__).parent.parent
AUDIO_DIR = PROJECT_DIR / "audio"
DATA_DIR = PROJECT_DIR / "data"
metadata = {}

for item in AUDIO_DIR.iterdir():
    if not item.is_dir():
        continue

    metadata[item.name] = []
    movie_name = item.name.replace("_"," ")
    counter = 1
    for file in item.iterdir():
        if not file.is_file():
            continue

        if file.suffix.lower() in ['.wav', '.mp3', '.flac']:
            audio_id = f"{item.name}_{counter}"
            audio_path = file.relative_to(PROJECT_DIR).as_posix()
            counter += 1
            record = {
                "audio_name": file.name,
                "audio_id": audio_id,
                "audio_path": audio_path,
                "movie_name": movie_name,
                "transcribe": None,
                "emotion_recognition": None
            }
            metadata[item.name].append(record)

DATA_DIR.mkdir(parents=True, exist_ok=True)
metadata_path = DATA_DIR/"metadata.json"
with open(metadata_path, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=4, ensure_ascii=False)