import chromadb
from tales.utils import messages_to_json, json_to_messages
from typing import Dict, List, Any, Optional


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
                    docs.append(doc["file_path"])
                return list(set(docs))
            else:
                return []
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            raise
