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
from tales.prompts import rag_prompt_template, analysis_prompt, reflect_prompt
from tales.document_processer import build_retriever


##################### Graph Compiling Script #####################
# This script compiles the LangGraph graph and the vector store for the RAG pipeline.
class GraphState(TypedDict):
    # Type for the state of the retrieval and query graph

    context: List[Document]
    query: AnyMessage  # Improved query for vector similarity search
    more_research: bool  # Flag to indicate if more research is needed
    messages: Annotated[List[AnyMessage], add_messages]  # Built-in MessagesState


# Define the nodes in the graph
def analyze_query(state: GraphState) -> GraphState:
    # Analyze user query to improve vector similarity search
    print("Analyzing query...")
    context_str = "Previous Messages:\n" + "\n\n".join(
        m.content for m in state["messages"][:-1]
    )

    if state["more_research"]:
        context_str += "\n\nPrevious query was not sufficient for research, reformulate another query, previous query for reference: " + state["query"].content

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
        if state["query"].content == "" or state["query"] is None:
            raise Exception("No query returned from analysis.")

        retrieved_docs = vector_store.similarity_search(state["query"].content, k=DOCS_RETRIEVED)
        if retrieved_docs == [] or retrieved_docs is None:
            raise Exception("No documents found for the query.")

        state["context"] = retrieved_docs
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        state["context"] = None

    return state


def generate_answer(state: GraphState) -> GraphState:
    # Generate an answer using retrieved context
    print("Generating answer...")

    documents = state["context"]

    # Format context from documents
    if documents == [] or documents is None:
        context_str = "No relevant context found."
    else:
        context_str = "Context:\n" + "\n\n".join(doc.page_content for doc in documents)
    context = SystemMessage(content=context_str)

    state["messages"] = [llm.invoke([rag_prompt_template] + [context] + state["messages"])]

    return state


def reflect_on_answer(state: GraphState) -> GraphState:
    # Generate an answer using retrieved context
    print("Reflecting on answer...")

    msg = llm.invoke([reflect_prompt] + state["messages"])

    state["more_research"] = True if msg.content == "more research needed" else False

    return state

def need_rag(state: GraphState) -> bool:
    """Check if the RAG process is needed based on the query."""
    # If the query is empty or None, we don't need RAG
    if state["query"] is None or state["query"].content == "":
        return "response"

    # Otherwise, we can proceed with RAG
    return "retrieve"

def more_research(state: GraphState) -> GraphState:
    """do deeper research on the topic. Repeat the retrieval and answer generation steps."""

    # Retrieve more documents based on the current query
    if state["more_research"]:
        print("Deepening research...")
        return "analyze"
    
    return END


# Build the vector store
vector_store = build_retriever()

# Create the graph
print("Compiling RAG Agent...")
workflow = StateGraph(GraphState)
memory = MemorySaver()

# Add nodes
workflow.add_node("analyze", analyze_query)
workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("response", generate_answer)
workflow.add_node("reflect", reflect_on_answer)

# Create edges
workflow.add_edge("retrieve", "response")
workflow.add_edge("response", "reflect")
workflow.add_conditional_edges("analyze", need_rag, ["retrieve", "response"])
workflow.add_conditional_edges("reflect", more_research, ["analyze", END])
workflow.add_edge("response", END)

# Set the entry point
workflow.set_entry_point("analyze")

# Compile the graph
agent = workflow.compile(checkpointer=memory)
