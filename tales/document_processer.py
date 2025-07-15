from typing import List
from tqdm import tqdm
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredExcelLoader,
)
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

from tales.config import (
    embeddings_model,
    DB_PATH,
    DATA_FOLDER,
    ACCEPTED_EXTENSIONS as AC,
)
from tales.utils import get_available_docs


def load_documents(docs_paths) -> List:
    documents = []

    # File extensions and their loaders, plus printing for bugfixing purposes
    print(f"Loading {len(docs_paths)} files...")
    pbar = tqdm(docs_paths, desc="Loading Documents")
    for file_path in pbar:
        try:
            pbar.set_postfix(file=file_path)
            file_extension = file_path.split(".")[-1].lower()

            if file_extension == "pdf":
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                documents.extend(docs)

            elif file_extension == "csv":
                loader = CSVLoader(file_path)
                docs = loader.load()
                documents.extend(docs)

            elif file_extension == "txt":
                loader = TextLoader(file_path)
                docs = loader.load()
                documents.extend(docs)

            elif file_extension in ["xlsx", "xls"]:
                loader = UnstructuredExcelLoader(file_path)
                docs = loader.load()
                documents.extend(docs)

        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    pbar.close()
    print(f"\nTotal loaded: {len(documents)} pages/documents")
    return documents


def process_documents(
    documents: List, chunk_size: int = 1536, chunk_overlap: int = 500
) -> List:
    """
    Process documents by splitting them into chunks for better handling by LLMs

    Args:
        documents: List of documents to process
        chunk_size: Size of each text chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of processed document chunks
    """

    total_length = sum(len(doc.page_content) for doc in documents)

    # Adjusting chunk size if its less then chunk_size: int = 500
    if total_length < chunk_size:
        chunk_size = max(100, total_length // 2)  # I set minimum chunk size to 100
        chunk_overlap = chunk_size // 4
        print(f"Adjusting chunk size to {chunk_size} due to small document")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )

    print("Processing and chunking documents...")
    chunks = text_splitter.split_documents(documents)

    # Using all document as a 1 chunk if its less then minimum chunk size
    if not chunks and documents:
        print("Document too small for chunking, using as is")
        chunks = documents

    print(f"Created {len(chunks)} document chunks")
    return chunks


def build_retriever():
    """
    Build a retriever for document chunks using embeddings and vector store

    Returns:
        A retriever object for querying document chunks
    """

    # Initialize the vector store
    vector_store = Chroma(
        collection_name="vector_store",
        embedding_function=embeddings_model,
        persist_directory=DB_PATH,
    )

    # Add new documents to the vector store if they are not already present
    docs = vector_store.get()["metadatas"]
    stored_docs = set([doc["source"] for doc in docs])
    available_docs = set(get_available_docs(folder_path=DATA_FOLDER, extensions=AC))
    docs_to_load = list(available_docs - stored_docs)

    if len(docs_to_load) > 0:
        print("Updating vector store with new documents...")
        documents = load_documents(docs_to_load)  # Load PDFs from paths
        chunks = process_documents(documents)  # Process documents into chunks
        vector_store.add_documents(chunks)  # Add documents to the vector store

    return vector_store
