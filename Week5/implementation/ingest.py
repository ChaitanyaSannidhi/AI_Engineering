
import os
import glob
from pathlib import Path

from dotenv import load_dotenv

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from langchain_huggingface import HuggingFaceEmbeddings


# --------------------------------------------------
# 1. Environment configuration
# --------------------------------------------------

load_dotenv(override=True)

if not os.getenv("OPENROUTER_API_KEY"):
    raise ValueError("OPENROUTER_API_KEY is missing.")


# --------------------------------------------------
# 2. Directory configuration
# --------------------------------------------------

DB_NAME = str(
    Path(__file__).parent.parent / "vector_db"
)

KNOWLEDGE_BASE = str(
    Path(__file__).parent.parent / "knowledge-base"
)


# --------------------------------------------------
# 3. Initialize Nemotron embeddings
# --------------------------------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# --------------------------------------------------
# 4. Load documents
# --------------------------------------------------

def fetch_documents():
    """
    Load all Markdown documents from the knowledge-base folders.

    Each document receives a doc_type metadata field
    based on its parent folder name.
    """

    folders = glob.glob(
        str(Path(KNOWLEDGE_BASE) / "*")
    )

    documents = []

    for folder in folders:

        if not os.path.isdir(folder):
            continue

        doc_type = os.path.basename(folder)

        loader = DirectoryLoader(
            folder,
            glob="**/*.md",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )

        folder_docs = loader.load()

        for doc in folder_docs:
            doc.metadata["doc_type"] = doc_type
            documents.append(doc)

    print(f"Loaded {len(documents)} documents.")

    return documents


# --------------------------------------------------
# 5. Split documents into chunks
# --------------------------------------------------

def create_chunks(documents):
    """
    Split documents into smaller overlapping chunks.
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    return chunks


# --------------------------------------------------
# 6. Generate embeddings and store in ChromaDB
# --------------------------------------------------

def create_embeddings(chunks):
    """
    Generate Nemotron embeddings and store them in ChromaDB.

    Existing vector database collection is deleted and rebuilt.
    """

    if not chunks:
        raise ValueError("No document chunks found to embed.")

    if os.path.exists(DB_NAME):
        existing_db = Chroma(
            persist_directory=DB_NAME,
            embedding_function=embeddings
        )

        existing_db.delete_collection()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_NAME
    )

    collection = vectorstore._collection

    count = collection.count()

    sample_embedding = collection.get(
        limit=1,
        include=["embeddings"]
    )["embeddings"][0]

    dimensions = len(sample_embedding)

    print(
        f"There are {count:,} vectors "
        f"with {dimensions:,} dimensions "
        f"in the vector store."
    )

    return vectorstore


# --------------------------------------------------
# 7. Main ingestion pipeline
# --------------------------------------------------

if __name__ == "__main__":

    documents = fetch_documents()

    chunks = create_chunks(documents)

    vectorstore = create_embeddings(chunks)

    print("Ingestion complete!")
