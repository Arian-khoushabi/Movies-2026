from langchain_core.documents import Document
from pathlib import Path
import json

def documentize():
    PROJECT_DIR = Path(__file__).parent.parent
    METADATA = PROJECT_DIR / "data" / "metadata.json"

    with open(METADATA, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    documents = []
    print("Start documentization")

    for movie_name, records in metadata.items():
        for record in records:
            if isinstance(record.get("transcribe"), str) and record["transcribe"].strip():
                doc = Document(
                    page_content=f"movie_name:{movie_name} comment:{record['transcribe']}",
                    metadata={
                        "movie_name": movie_name,
                        "emotion_recognition": json.dumps(record["emotion_recognition"]),
                        "path": record["audio_path"],
                        "audio":record["audio_id"]
                    },
                )
                documents.append(doc)

    print(f"Documentization successfully completed: {len(documents)} documents")
    return documents