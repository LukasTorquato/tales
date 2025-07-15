from typing import List, TypedDict, Annotated

# LangChain imports
from langchain.schema import Document
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AnyMessage,
    RemoveMessage,
)

# LangGraph imports
from langgraph.graph import StateGraph, add_messages, END
from langgraph.checkpoint.memory import MemorySaver

# Local imports
from tales.config import *
from tales.prompts import rag_prompt_template, analysis_prompt
from tales.document_processer import build_retriever


##################### Graph Compiling Script #####################
# This script compiles the LangGraph graph and the vector store for the RAG pipeline.
class GraphState(TypedDict):
    # Type for the state of the retrieval and query graph

    context: List[Document]
    summary: str  # Summary of the context
    query: AnyMessage  # Improved query for vector similarity search
    messages: Annotated[List[AnyMessage], add_messages]  # Built-in MessagesState


# Define the nodes in the graph
def analyze_query(state: GraphState) -> GraphState:
    # Analyze user query to improve vector similarity search
    print("Analyzing query...")

    context_str = "Previous Messages:\n" + "\n\n".join(
        m.content for m in state["messages"][:-1]
    )

    prev_msg = SystemMessage(content=context_str)
    user_query = HumanMessage(content="User query: " + state["messages"][-1].content)

    state["query"] = llm.invoke([analysis_prompt, prev_msg, user_query])

    return state


def retrieve_documents(state: GraphState) -> GraphState:
    # Retrieve relevant documents for the latest query
    print("Retrieving documents...")

    # print(f"Improved query: {state['query'].content}")
    # TODO: Check if similarity search can be done with HumanMessage instead of str
    # TODO: https://smith.langchain.com/hub/zulqarnain/multi-query-retriever-similarity
    try:
        if state["query"].content == "" or state["query"].content is None:
            raise Exception("No query returned from analysis.")

        retrieved_docs = vector_store.similarity_search(state["query"].content, k=DOCS_RETRIEVED)
        if retrieved_docs == [] or retrieved_docs is None:
            raise Exception("No documents found for the query.")

        state["context"] = retrieved_docs
    except Exception as e:
        state["context"] = "No existent context. " + f"ERROR: {e}"

    return state


def generate_answer(state: GraphState) -> GraphState:
    # Generate an answer using retrieved context
    print("Generating answer...")

    documents = state["context"]

    # Format context from documents
    context_str = "Context:\n" + "\n\n".join(doc.page_content for doc in documents)
    context = SystemMessage(content=context_str)

    # Get response from LLM
    if USE_REASONING:
        print("Thinking...")
        message = [
            llm_reasoning.invoke([rag_prompt_template] + [context] + state["messages"])
        ]
    else:
        message = [llm.invoke([rag_prompt_template] + [context] + state["messages"])]

    return {"context": documents, "messages": message}


def summarize_conversation(state: GraphState) -> GraphState:
    # Summarize the conversation
    print("Summarizing conversation...")

    summary = state.get("summary", "")
    if summary:
        # If a summary already exists, extend it with the latest message
        summary_message = (
            f"This is the summary of the conversation until now:\n{summary}\n\n"
            + "Extend the summary with the following messages:\n"
        )
        summary += "\n\n" + state["messages"][-1].content
    else:
        summary_message = (
            "Create a summary of the conversation with the following messages:\n"
        )

    messages = [HumanMessage(content=summary_message)] + state["messages"]
    state["summary"] = llm.invoke(messages).content

    # Remove all but the last two messages from the state
    # This is to keep the context manageable and avoid exceeding token limits
    state["messages"] = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]

    return state


# Determine whether to end or summarize the conversation
def should_continue(state: GraphState):
    """Return the next node to execute."""

    messages = state["messages"]

    # If there are more than six messages, then we summarize the conversation
    if len(messages) > 6:
        return "summarize_conversation"

    # Otherwise we can just end
    return END


# Build the vector store
vector_store = build_retriever()

# Create the graph
print("Compiling LangGraph Graph...")
workflow = StateGraph(GraphState)
memory = MemorySaver()

# Add nodes
workflow.add_node("analyze", analyze_query)
workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("response", generate_answer)

# Create edges
workflow.add_edge("analyze", "retrieve")
workflow.add_edge("retrieve", "response")
workflow.add_edge("response", END)

# Set the entry point
workflow.set_entry_point("analyze")

# Compile the graph
agent = workflow.compile(checkpointer=memory)
