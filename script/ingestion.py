from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from pathlib import Path
import json
from documentize import documentize

PROJECT_DIR = Path(__file__).parent.parent
CHROMA = PROJECT_DIR / 'data' / 'chroma_db'

documents =  documentize()

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",

    model_kwargs={
        "device": "cpu",
    },

    encode_kwargs={
        "normalize_embeddings": True,
    },
)

ids = [
    f"{document.metadata['movie_name']}_{document.metadata['audio']}"
    for document in documents
]

vector_db = Chroma(
    collection_name="movie_review",
    persist_directory=str(CHROMA),
    embedding_function=embeddings,
)

existing_ids = (
    set(vector_db.get(ids=ids, include=[])["ids"])
    if ids else set()
)

new_documents = []
new_ids = []

for document, document_id in zip(documents, ids):
    if document_id not in existing_ids:
        new_documents.append(document)
        new_ids.append(document_id)
        existing_ids.add(document_id)

if new_documents:
    vector_db.add_documents(
        documents=new_documents,
        ids=new_ids,
    )

print(f"Added {len(new_documents)} new documents.")