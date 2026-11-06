from documentize import documentize
from retriever import retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_cohere import CohereRerank, ChatCohere
from dotenv import load_dotenv
import os
from pathlib import Path
import json

class LLM_chat:
    def __init__(self, api_key):
        self.api_key = api_key
        self.model = ChatCohere(
        model="command-a-plus-05-2026",
        temperature=0.0,
        cohere_api_key=api_key,
        streaming=True
    )
        self.chat_history = StreamlitChatMessageHistory(key="special_app_key")
        self.retriever = retriever(self.api_key)

        qa_memory_massage = """Given a chat history and the latest user question 
        which might reference context in the chat history, formulate a standalone question 
        which can be understood without the chat history. Do NOT answer the question, 
        just reformulate it if needed and otherwise return it as is.
        Always formulate the standalone question in English, regardless of the language used by the user."""

        qa_memory_templete = ChatPromptTemplate.from_messages(
                [
            ("system", qa_memory_massage),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
                ]
            )

        history_aware_retriever = create_history_aware_retriever(
            self.model, self.retriever, qa_memory_templete
        )

        qa_system_prompt = """
        You are a movie review analysis assistant.

        Your task is to answer questions using the retrieved movie review comments.

        Language requirements:
        - Always respond in English.
        - If the user's question is in another language, understand it and answer in English.
        - Never mix English with other languages in your response.
        - Use natural, fluent, grammatically correct English.

        Answering requirements:
        - Answer the question directly.
        - Organize information in a logical order.
        - Write coherent, well-structured sentences.
        - Avoid repetitive introductions and unnecessary explanations.
        - Do not repeat the same information in different words.
        - Use natural phrasing instead of literal translations.
        - Distinguish explicitly stated information from assumptions.
        - Never present an inference as a confirmed fact.
        - If the retrieved context does not contain enough information, say so.
        - Do not invent information.

        Use the retrieved context as your primary source.

        Retrieved context:
        {context}
        """

        qa_prompt_tempelte = ChatPromptTemplate.from_messages(
            [
                ('system', qa_system_prompt),
                MessagesPlaceholder('chat_history'),
                ('human', '{input}')
            ]
        )

        qa_chain = create_stuff_documents_chain(self.model, qa_prompt_tempelte)
        self.rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)
        self.massage_history = RunnableWithMessageHistory(
            self.rag_chain,
            lambda session_id: self.chat_history,
            input_messages_key='input',
            history_messages_key='chat_history',
            output_messages_key='answer'
        )
    def reset_chat(self):
            self.chat_history.clear()
            self.chat_history.add_ai_message("Ready to find the answers to your query!")
    
    def get_chat_history(self):
        return self.chat_history.messages

    def process_input(self, prompt):
        config = {"configurable": {"session_id": "any"}}

        try:
            for chunk in self.massage_history.stream(
                {"input": prompt}, config
            ):
                if "answer" in chunk:
                    yield chunk["answer"]

        except Exception as error:
            yield str(error)