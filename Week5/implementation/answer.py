
import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings


# --------------------------------------------------
# 1. Environment configuration
# --------------------------------------------------

load_dotenv(override=True)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY is missing.")


# --------------------------------------------------
# 2. Model and database configuration
# --------------------------------------------------

MODEL = "nvidia/nemotron-3.5-lightning:free"

DB_NAME = str(Path(__file__).parent.parent / "vector_db")

RETRIEVAL_K = 10


# --------------------------------------------------
# 3. Embeddings and vector database
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = Chroma(
    persist_directory=DB_NAME,
    embedding_function=embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": RETRIEVAL_K}
)


# --------------------------------------------------
# 4. OpenRouter LLM configuration
# --------------------------------------------------

llm = ChatOpenAI(
    model=MODEL,
    temperature=0,
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)


# --------------------------------------------------
# 5. System prompt
# --------------------------------------------------

SYSTEM_PROMPT = """
You are a knowledgeable, friendly assistant representing
the company Insurellm.

You are chatting with a user about Insurellm.

Use the provided context to answer questions whenever relevant.

If the answer cannot be found in the context, clearly say so.
Do not invent information.

Context:
{context}
"""


# --------------------------------------------------
# 6. Extract text from Gradio messages
# --------------------------------------------------

def extract_text(content):

    if isinstance(content, str):
        return content

    if isinstance(content, list):

        parts = []

        for item in content:

            if isinstance(item, str):
                parts.append(item)

            elif isinstance(item, dict):

                text = item.get("text")

                if isinstance(text, str):
                    parts.append(text)

        return "\n".join(parts)

    return ""


# --------------------------------------------------
# 7. Retrieve relevant documents
# --------------------------------------------------

def fetch_context(question: str) -> list[Document]:

    question = extract_text(question)

    return retriever.invoke(question)


# --------------------------------------------------
# 8. Combine conversation history
# --------------------------------------------------

def combined_question(question, history=None):

    history = history or []

    prior = []

    for message in history:

        if not isinstance(message, dict):
            continue

        content = extract_text(
            message.get("content", "")
        )

        if content.strip():
            prior.append(content)

    question = extract_text(question)

    return "\n".join(prior + [question])


# --------------------------------------------------
# 9. Generate answer using RAG
# --------------------------------------------------

def answer_question(question, history=None):

    history = history or []

    question = extract_text(question)

    # Include conversation history for retrieval
    combined = combined_question(
        question,
        history
    )

    # Retrieve relevant documents
    docs = fetch_context(combined)

    # Build context
    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    system_prompt = SYSTEM_PROMPT.format(
        context=context
    )

    # Prepare LLM messages
    messages = [
        SystemMessage(content=system_prompt)
    ]

    for message in history:

        if not isinstance(message, dict):
            continue

        role = message.get("role")

        content = extract_text(
            message.get("content", "")
        )

        if not content.strip():
            continue

        if role == "user":

            messages.append(
                HumanMessage(content=content)
            )

        elif role == "assistant":

            messages.append(
                AIMessage(content=content)
            )

    # Add current question
    messages.append(
        HumanMessage(content=question)
    )

    # Generate response
    response = llm.invoke(messages)

    return response.content, docs
