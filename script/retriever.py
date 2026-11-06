from documentize import documentize
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_cohere import CohereRerank
from pathlib import Path
import json

def retriever(api_key):

    PROJECT_DIR = Path(__file__).parent.parent
    CHROMA_DIR = PROJECT_DIR / 'data' /  'chroma_db'

    documents = documentize()

    print(f"Loaded {len(documents)} documents.")

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    vector_db = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings,
        collection_name="movie_review",
    )

    retriever = vector_db.as_retriever(
        search_type = "mmr",
        search_kwargs = {
            "fetch_k":100,
            "k":40,
            "lambda_mult":0.8,
        }
    )

    print("Creating BM25 retriever...")

    bm25 = BM25Retriever.from_documents(documents)
    bm25.k = 40

    ensemble = EnsembleRetriever(
        retrievers=[
            retriever,
            bm25
        ],
        weights=[
            0.8,
            0.2
        ]
    )

    reranker = CohereRerank(
        model="rerank-multilingual-v3.0",
        top_n=25,
        cohere_api_key = api_key
    )

    compressor_reranker = ContextualCompressionRetriever(
        base_compressor=reranker, base_retriever=ensemble
    )

    return compressor_reranker