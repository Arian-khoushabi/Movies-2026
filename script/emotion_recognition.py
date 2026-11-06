from pathlib import Path
from transformers import pipeline
import json

model = pipeline(
    "audio-classification",
    model = "superb/wav2vec2-base-superb-er",
    device='cpu'
)
print('model loaded successfully')

BASE_PATH = Path("E:/myRepository/movie_audio_dataset")
PROJECT_DIR = Path(__file__).parent.parent
METADATA = PROJECT_DIR / 'data' / 'metadata.json'
with open(METADATA, 'r', encoding='utf-8') as f:
    metadata = json.load(f)

print('model loaded successfully')

counter = 0
print('starting recognition')
for records in metadata.values():
    for record in records:
        if record["emotion_recognition"] is None:
            audio_path = BASE_PATH / record["audio_path"]
            if audio_path.exists() and audio_path.stat().st_size > 0:
                emo_dict = {}
                emotion = model(str(audio_path))
                for emo in emotion:
                    emo_dict[emo["label"]] = emo["score"] * 100
                record["emotion_recognition"] = emo_dict
                counter += 1
                if counter % 50 == 0:
                    print(f"progressed {counter}")
                    with open(METADATA, 'w', encoding='utf-8') as f:
                        json.dump(metadata,f, indent=4, ensure_ascii=False)
with open(METADATA, 'w', encoding='utf-8') as f:
    json.dump(metadata,f, indent=4, ensure_ascii=False)
    print("metadata saved successfully")
