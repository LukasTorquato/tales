from tales.utils import print_state_messages
from tales.agent import *
from tales.db_handler import ChromaDBHandler
from tales.ppt_agent import ppt_agent
import asyncio

def query_with_graph(agent, msgs: list[AnyMessage], thread_id: int):
    """
    Process a query through the LangGraph

    Args:
        graph: The compiled LangGraph
        query: The query to process
        thread_id: The thread ID for the conversation

    Returns:
        The result containing the answer and source documents
    """
    print(f"\nQuery: {msgs[-1].content}")
    print("Processing query through Agent's workflow...")

    config = {"configurable": {"thread_id": thread_id}}

    # Initialize state with the query
    initial_state = {"context": [], "messages": msgs, "summary": "", "query": None, "more_research": False}

    # Run the graph
    result = agent.invoke(initial_state, config)

    return result


def main():

    thread_id = 4
    # Initialize the ChromaDB handler
    db_handler = ChromaDBHandler(persist_directory=DB_PATH)
    msgs = db_handler.get_conversation(thread_id)
    while True:

        question = input("Enter your question (or 'exit' to quit): ")
        if question.lower() == "exit":
            break

        msgs.append(HumanMessage(content=question))
        result = query_with_graph(agent, msgs, thread_id)

        """
        db_handler.add_conversation(
            thread_id=thread_id,
            conversation=result["messages"],
        )"""

        print_state_messages(result)

        present = input("Would you like to make a presentation of this subject [y/n]: ")
        if present == "y":
            print("Creating PowerPoint presentation...")
            asyncio.run(ppt_agent(result["messages"]))
            print("PowerPoint presentation created successfully.")

if __name__ == "__main__":
    main()
