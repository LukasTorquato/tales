import chromadb
from tales.utils import messages_to_json, json_to_messages, get_available_docs
from tales.document_processer import build_retriever
from tales.config import embeddings_model
from typing import Dict, List, Any, Optional
import chromadb
import os


class ChromaDBHandler:
    def __init__(self, persist_directory: str = "./database"):
        """Initialize the ChromaDB handler with persistent storage."""
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.conversation_collection = None
        self.documents_collection = None
        self.initialize_collections()

    def initialize_collections(
        self,
        conversation_collection_name: str = "conversations",
        docs_collection: str = "vector_store",
    ):
        """Initialize or get existing collections for conversations and vectorized documents."""
        try:
            self.conversation_collection = self.client.get_or_create_collection(
                name=conversation_collection_name,
                metadata={"description": "Collection for storing LLM conversations"},
            )

            self.documents_collection = self.client.get_or_create_collection(
                name=docs_collection,
                metadata={"description": "Collection for storing vectorized documents"},
            )

            print(f"Database Initialized Successfully...")
        except Exception as e:
            print(f"Error initializing collections: {e}")
            raise

    def add_conversation(
        self,
        thread_id: str,
        conversation: List[Any],
    ) -> str:
        """Add a conversation to the conversation collection."""
        if self.conversation_collection is None:
            raise ValueError(
                "Conversation collection not initialized. Run initialize_collections first."
            )
        msgs = messages_to_json(conversation)
        try:
            if self.conversation_collection.get(ids=[str(thread_id)])["documents"]:
                self.conversation_collection.update(
                    ids=[str(thread_id)], documents=msgs
                )
            else:
                self.conversation_collection.add(ids=[str(thread_id)], documents=msgs)
            print(f"Added conversation with Thread ID: {thread_id}")
            return thread_id
        except Exception as e:
            print(f"Error adding conversation: {e}")
            raise

    def get_conversation(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Get a conversation by ID."""
        if self.conversation_collection is None:
            raise ValueError(
                "Conversation collection not initialized. Call initialize_collections first."
            )

        try:
            result = self.conversation_collection.get(ids=[str(thread_id)])
            if result["documents"]:
                return json_to_messages(result["documents"][0])
            return []
        except Exception as e:
            print(f"Error retrieving conversation: {e}")
            raise

    def delete_conversation(self, thread_id: str) -> bool:
        """Delete a conversation by ID."""
        if self.conversation_collection is None:
            raise ValueError(
                "Conversation collection not initialized. Call initialize_collections first."
            )

        try:
            self.conversation_collection.delete(ids=[str(thread_id)])
            print(f"Deleted conversation with ID: {thread_id}")
            return True
        except Exception as e:
            print(f"Error deleting conversation: {e}")
            return False

    def get_stored_documents(self) -> List[str]:
        """Get all stored documents."""
        if self.documents_collection is None:
            raise ValueError(
                "Documents collection not initialized. Call initialize_collections first."
            )

        try:
            vectors = self.documents_collection.get()
            if vectors["documents"]:
                docs = []
                for doc in vectors["metadatas"]:
                    docs.append(doc["source"])
                return list(set(docs))
            else:
                return []
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            raise
            
    def add_document(self, file_path: str) -> bool:
        """Add a new document to the vector store.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            bool: True if document was added successfully
        """
        if self.documents_collection is None:
            raise ValueError(
                "Documents collection not initialized. Call initialize_collections first."
            )
            
        try:
            # Check if file exists
            import os
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                return False
                
            # Use the build_retriever to add the document
            # The build_retriever function will check if the document is already in the vector store
            from tales.document_processer import load_documents, process_documents
            documents = load_documents([file_path])
            if documents:
                chunks = process_documents(documents)
                vector_store = build_retriever()
                vector_store.add_documents(chunks)
            
            print(f"Added document to vector store: {file_path}")
            return True
            
        except Exception as e:
            print(f"Error adding document: {e}")
            return False
            
    def delete_document(self, file_path: str) -> bool:
        """Delete a document from the vector store.
        
        Args:
            file_path: Path to the document to delete
            
        Returns:
            bool: True if document was deleted successfully
        """
        if self.documents_collection is None:
            raise ValueError(
                "Documents collection not initialized. Call initialize_collections first."
            )
            
        try:
            # Check if document exists in the vector store
            stored_docs = self.get_stored_documents()
            if file_path not in stored_docs:
                print(f"Document not found in vector store: {file_path}")
                return False
                
            # Get all chunks for this document
            results = self.documents_collection.get()
            doc_ids = []
            
            for i, metadata in enumerate(results["metadatas"]):
                if metadata["source"] == file_path:
                    doc_ids.append(results["ids"][i])
            
            if doc_ids:
                # Delete the chunks
                self.documents_collection.delete(ids=doc_ids)
                print(f"Deleted document from vector store: {file_path}")
                return True
            else:
                return False
                
        except Exception as e:
            print(f"Error deleting document: {e}")
            return False
