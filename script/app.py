import os
import json
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from main import LLM_chat


PROJECT_DIR = Path(__file__).resolve().parent.parent

METADATA_PATH = PROJECT_DIR / "data" / "metadata.json"
SUMMARY_PATH = PROJECT_DIR / "data" / "summaries.json"

load_dotenv()

st.set_page_config(
    page_title="Movie Review Analysis",
    page_icon="🎬",
    layout="wide"
)

def load_summaries():

    if not SUMMARY_PATH.exists():
        raise FileNotFoundError("summaries.json was not found.")

    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        summaries = json.load(f)

    if not isinstance(summaries, dict):
        raise ValueError("summaries.json must contain a JSON object.")

    return summaries


def load_metadata():

    if not METADATA_PATH.exists():
        raise FileNotFoundError("metadata.json was not found.")

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return metadata


def stream_chat_answer(prompt):

    try:

        response = st.session_state.chatbot.process_input(prompt)

        if isinstance(response, str):
            yield response
            return

        for chunk in response:

            if isinstance(chunk, str):
                yield chunk

            elif isinstance(chunk, dict):

                answer = chunk.get("answer")

                if isinstance(answer, str):
                    yield answer

    except Exception as error:

        yield f"Error: {error}"


def display_chat_history():

    for message in st.session_state.chatbot.get_chat_history():

        if message.type == "human":
            role = "user"

        elif message.type == "ai":
            role = "assistant"

        else:
            continue

        content = message.content

        if isinstance(content, str) and content:

            with st.chat_message(role):
                st.markdown(content)


if "chatbot" not in st.session_state:
    st.session_state.chatbot = None

if "chatbot_key" not in st.session_state:
    st.session_state.chatbot_key = None


st.title("🎬 Movie Review Analysis")

st.caption(
    "Movie review analysis using Hybrid RAG "
    "and Natural Language Processing."
)


with st.sidebar:

    st.header("Settings")

    api_key = st.text_input(
        "Cohere API Key",
        value=os.getenv("COHERE_API_KEY", ""),
        type="password"
    )


chat_tab, summary_tab = st.tabs(
    [
        "💬 Chatbot",
        "📚 Movie Summaries",
    ]
)


with chat_tab:

    st.header("Movie Review Chatbot")

    st.caption(
        "Ask questions about movies based on the comments "
        "available in your dataset."
    )

    if not api_key:

        st.info("Enter your Cohere API key in the sidebar.")

    else:

        if (
            st.session_state.chatbot is None
            or st.session_state.chatbot_key != api_key
        ):

            with st.spinner("Initializing chatbot..."):

                try:

                    st.session_state.chatbot = LLM_chat(api_key)

                    st.session_state.chatbot.reset_chat()

                    st.session_state.chatbot_key = api_key

                except Exception as error:

                    st.session_state.chatbot = None

                    st.error(f"Initialization failed: {error}")

        if st.session_state.chatbot is not None:

            col1, col2 = st.columns([5, 1])

            with col2:

                if st.button("Clear Chat", use_container_width=True):

                    st.session_state.chatbot.reset_chat()

                    st.rerun()

            display_chat_history()

            prompt = st.chat_input(
                "Ask about a movie..."
            )

            if prompt:

                with st.chat_message("user"):

                    st.markdown(prompt)

                with st.chat_message("assistant"):

                    st.write_stream(
                        stream_chat_answer(prompt)
                    )


with summary_tab:

    st.header("Movie Summaries")

    st.caption(
        "Explore AI-generated summaries of movie reviews."
    )

    try:

        summaries = load_summaries()

        metadata = load_metadata()

    except Exception as error:

        st.error(str(error))

    else:

        available_movies = [
            movie
            for movie, summary in summaries.items()
            if isinstance(summary, str) and summary.strip()
        ]

        if not available_movies:

            st.warning("No movie summaries are available.")

        else:

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Movies Summarized",
                    len(available_movies)
                )

            with col2:

                st.metric(
                    "Movies in Dataset",
                    len(metadata)
                )

            st.divider()

            selected_movie = st.selectbox(
                "Select a movie",
                options=sorted(available_movies)
            )

            summary = summaries[selected_movie]

            st.subheader(selected_movie)

            st.markdown("### Review Summary")

            st.write(summary)

            records = metadata.get(selected_movie, [])

            valid_comments = sum(
                1
                for record in records
                if isinstance(record, dict)
                and isinstance(record.get("transcribe"), str)
                and record["transcribe"].strip()
            )

            st.divider()

            st.caption(
                f"Based on {valid_comments} valid transcriptions."
            )

st.divider()

st.caption(
    "Movie Review Analysis | Hybrid RAG & NLP | Created by Arian Khoushabi"
)