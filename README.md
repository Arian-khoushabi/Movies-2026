# Movie Review Analysis

An LLM-powered movie-review analysis application built around a **hybrid Retrieval-Augmented Generation (RAG)** pipeline. Users can ask questions about movies and explore AI-generated summaries based on audience reviews in the dataset.

## Features

- **Movie-review chatbot:** answers questions using retrieved audience-review comments.
- **Hybrid retrieval:** combines dense vector retrieval from ChromaDB with BM25 keyword retrieval.
- **MMR search:** uses Maximal Marginal Relevance to retrieve a diverse set of candidate documents.
- **Reranking:** uses Cohere Rerank to refine the retrieved results before generation.
- **Conversation history:** uses LangChain message history to support follow-up questions.
- **Movie summaries:** displays pre-generated summaries for each movie.
- **Audio-derived metadata:** the data preparation pipeline transcribes review audio and extracts emotion predictions.

## Architecture

1. Collect and organize movie-review audio files by movie.
2. Create `metadata.json` records for the audio files.
3. Transcribe audio using OpenAI Whisper.
4. Extract emotion predictions using the Hugging Face model `superb/wav2vec2-base-superb-er`.
5. Convert valid transcriptions into LangChain documents.
6. Generate embeddings with `BAAI/bge-m3` and store them in ChromaDB.
7. At query time, combine Chroma vector retrieval and BM25 retrieval, then rerank results with Cohere.
8. Pass the retrieved context and conversation history to Cohere Command for answer generation.
9. Display chatbot responses and pre-generated movie summaries in Streamlit.

## Technology Stack

- Python
- Streamlit
- LangChain
- Cohere Command (LLM)
- Cohere Rerank
- ChromaDB
- Hugging Face Transformers
- BAAI/bge-m3 embeddings
- OpenAI Whisper
- BM25

## Project Structure

The paths below reflect the layout expected by the supplied code. Adjust them if your repository uses different names.

```text
movie-review-analysis/
├── app/
│   └── streamlit_app.py       # Streamlit entry point (example name)
├── src/
│   ├── main.py                 # LLM_chat and RAG chain
│   ├── retriever.py            # Hybrid retrieval and reranking
│   ├── documentize.py          # Metadata-to-Document conversion
│   └── ...                     # Data preparation scripts
├── audio/                      # Source audio, grouped by movie (not committed)
├── data/
│   ├── metadata.json           # Transcriptions and emotion metadata
│   ├── summaries.json          # Pre-generated movie summaries
│   └── chroma_db/              # Persistent Chroma collection (not committed)
├── requirements.txt
├── .gitignore
└── README.md
```

> **Important:** Your code uses `Path(__file__).parent.parent` in several modules. Keep the actual directory structure consistent with those relative paths, or update the paths before deployment. Also update the Streamlit entry-point path in the deployment settings to match the real location of your app file.

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

macOS / Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Whisper also requires **FFmpeg** to be installed on the operating system if you run the audio-transcription pipeline. Installing the Python package alone does not install the FFmpeg executable.

### 4. Configure the Cohere API key

Create a `.env` file in the project root:

```env
COHERE_API_KEY=your_cohere_api_key
```

Do not commit `.env` or publish your API key.

### 5. Prepare the data and vector database

Run your data-preparation scripts in the required order:

1. Generate `metadata.json` from the audio folders.
2. Run transcription.
3. Run emotion recognition.
4. Generate `summaries.json`.
5. Run the Chroma indexing script to create/populate `data/chroma_db/`.

The Streamlit app expects the summaries file and metadata file to exist. The chatbot also expects the Chroma collection named `movie_review` to have been populated.

### 6. Run Streamlit

```bash
streamlit run app/streamlit_app.py
```

Replace `app/streamlit_app.py` with the actual path to your Streamlit file.

## Deploy on Streamlit Community Cloud

1. Push the project to a GitHub repository.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/).
3. Create a new app and select your repository, branch, and Streamlit entry-point file.
4. Open the app's **Settings → Secrets**.
5. Add your Cohere API key in TOML format:

```toml
COHERE_API_KEY = "your_cohere_api_key"
```

6. Deploy the app.

The application reads the key from the environment through `os.getenv`, which is compatible with Streamlit secrets exposed as environment variables only if you explicitly map them. A safer and more direct approach is to update the app's key lookup to:

```python
api_key = st.text_input(
    "Cohere API Key",
    value=os.getenv("COHERE_API_KEY", st.secrets.get("COHERE_API_KEY", "")),
    type="password"
)
```

If you use this fallback, ensure the app can access `st.secrets` in the deployment environment.

## Deployment Notes

- **Do not commit credentials.** Keep `.env` and `.streamlit/secrets.toml` out of Git.
- **Large files:** audio files, model weights, and the Chroma database can make a Git repository very large. This template ignores the audio directory and local Chroma database.
- **Chroma database availability:** because `data/chroma_db/` is ignored, the deployed app will not have the local vector index unless you create it during deployment or use a hosted/persistent vector database. The current app does not automatically build the index at startup.
- **Required data files:** `summaries.json` and `metadata.json` must be made available to the deployed app. The current `.gitignore` ignores `metadata.json`; if you intend to commit a small, non-sensitive metadata file, remove that ignore rule. Otherwise, provide it through a suitable data-storage solution.
- **Model downloads:** the embedding and reranking components may download model files at runtime. First startup can therefore be slow and may require substantial memory and disk space.
- **Audio pipeline vs. web app:** Whisper transcription and emotion recognition are data-preparation steps. They generally should be run before deployment rather than every time a user opens the Streamlit app.

## Current Limitations

- The application can only answer questions supported by the reviews indexed in its vector database.
- Summary quality and coverage depend on the available transcriptions.
- Emotion predictions are auxiliary metadata and should not be treated as definitive evidence of a reviewer's opinion.
- The app currently uses a Cohere API key supplied by the user or configured in deployment secrets.

## Author

**Arian Khoushabi**
