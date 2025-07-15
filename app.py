from tales.utils import print_state_messages
from tales.agent import *
from tales.db_handler import ChromaDBHandler

def query_with_graph(agent, query: HumanMessage, thread_id: int, pmsgs: list = None):
    """
    Process a query through the LangGraph

    Args:
        graph: The compiled LangGraph
        query: The query to process
        thread_id: The thread ID for the conversation

    Returns:
        The result containing the answer and source documents
    """
    print(f"\nQuery: {query.content}")
    print("Processing query through Agent's workflow...")

    config = {"configurable": {"thread_id": thread_id}}

    # Initialize state with the query
    initial_state = {"context": [], "messages": query}

    # Run the graph
    result = agent.invoke(initial_state, config)

    return result


def main():

    thread_id = 1
    # Initialize the ChromaDB handler
    db_handler = ChromaDBHandler(persist_directory=DB_PATH)
    while True:

        question = input("Enter your question (or 'exit' to quit): ")
        if question.lower() == "exit":
            break
        """
        pmsgs = db_handler.get_conversation(thread_id)
        if pmsgs:
            print("Previous messages found in the database. Loading...")
            for msg in pmsgs:
                print(f"Previous message: {msg.content}")"""
        result = query_with_graph(agent, HumanMessage(content=question), thread_id)

        db_handler.add_conversation(
            thread_id=thread_id,
            conversation=result["messages"],
        )

        print_state_messages(result)


if __name__ == "__main__":
    main()
