from langchain_core.messages import HumanMessage, AIMessage
import json
import os


def export_json(data: dict, filename: str) -> None:
    """
    Export data to a JSON file.

    Args:
        data: Data to export
        filename: Name of the file to save the data
    """
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def get_available_docs(folder_path, extensions) -> list:
    """
    Search for documents in the specified folder and return their names and paths.

    Args:
        folder_path: Path to the folder to search (defaults to 'data' in project root)
        extensions: List of file extensions to include (e.g., ['pdf', 'txt'])
                  If None, includes all files

    Returns:
        List of dictionaries containing document name and path
    """

    # Check if directory exists
    if not os.path.isdir(folder_path):
        print(f"Warning: Directory not found at {folder_path}")
        return []

    documents = []

    # Walk through directory and subdirectories
    for root, _, files in os.walk(folder_path):
        for file in files:
            # Check if file has one of the specified extensions
            if extensions is None or any(
                file.lower().endswith(f".{ext.lower()}") for ext in extensions
            ):
                documents.append(os.path.join(root, file))
                # documents.append({"name": file, "path": os.path.join(root, file)})

    return documents


def messages_to_json(messages: list) -> list:

    msgs = []
    for msg in messages:
        if not isinstance(msg, (HumanMessage, AIMessage)):
            raise ValueError("Invalid message type. Must be HumanMessage or AIMessage.")
        if isinstance(msg, HumanMessage):
            msgs.append([msg.content, {"role": "user"}])
        else:
            msgs.append([msg.content, {"role": "assistant"}])
    msgs_json = json.dumps(msgs)
    return msgs_json


def json_to_messages(json_str: str) -> list:
    msgs = json.loads(json_str)
    messages = []
    for msg in msgs:
        if msg[1]["role"] == "user":
            messages.append(
                HumanMessage(content=msg[0], additional_kwargs={"role": "user"})
            )
        else:
            messages.append(
                AIMessage(content=msg[0], additional_kwargs={"role": "assistant"})
            )
    return messages


def print_state_messages(state, context=False, metadata=False):
    if context:
        print("\n" + "=" * 25 + " CONTEXT " + "=" * 25)
        for c in state["context"]:
            print(f"Document: {c.metadata['source']}")
            print(c.page_content)
            print("=" * 50)

    print("\n" + "=" * 25 + " MESSAGES " + "=" * 25)
    for message in state["messages"]:
        message.pretty_print()

    if metadata:
        print("\n" + "=" * 25 + " USAGE " + "=" * 25)
        print(state["messages"][-1].usage_metadata)
