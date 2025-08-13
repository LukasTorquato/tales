from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from tales.agent import agent
from tales.db_handler import ChromaDBHandler
from tales.utils import print_state_messages, get_available_docs
from tales.config import DB_PATH, DATA_FOLDER, ACCEPTED_EXTENSIONS
from tales.ppt_agent import ppt_agent
import asyncio
import os
import shutil
import uuid
from werkzeug.utils import secure_filename
from langchain_core.messages import HumanMessage

app = Flask(__name__)
app.secret_key = str(uuid.uuid4())  # Set a secret key for flash messages
db_handler = ChromaDBHandler(persist_directory=DB_PATH)

# Ensure data directory exists
os.makedirs(DATA_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return redirect(url_for('chat', thread_id=1))

@app.route('/chat/<int:thread_id>')
def chat(thread_id):
    # Get list of all conversations/threads
    threads = get_all_thread_ids()
    # Get list of documents in vector store
    stored_docs = db_handler.get_stored_documents()
    return render_template('chat.html', 
                          thread_id=thread_id, 
                          threads=threads, 
                          stored_docs=stored_docs)

def get_all_thread_ids():
    # This is a placeholder - you'll need to implement a way to get all thread IDs
    # For now, let's return some sample thread IDs
    try:
        # Get all thread IDs from the conversation collection
        result = db_handler.conversation_collection.get()
        if result and result["ids"]:
            return [int(thread_id) for thread_id in result["ids"]]
        return [1]  # Default thread ID if none exists
    except Exception as e:
        print(f"Error retrieving thread IDs: {e}")
        return [1]  # Default thread ID if there's an error

@app.route('/api/query', methods=['POST'])
def query():
    data = request.json
    thread_id = data.get('thread_id', 1)
    question = data.get('question', '')
    
    if not question.strip():
        return jsonify({"error": "Question cannot be empty"}), 400

    try:
        # Get existing conversation
        msgs = db_handler.get_conversation(thread_id)
        
        # Add new question to messages
        msgs.append(HumanMessage(content=question))
        
        # Process query through agent
        result = query_with_graph(agent, msgs, thread_id)
        
        # Save updated conversation
        db_handler.add_conversation(
            thread_id=thread_id,
            conversation=result["messages"],
        )
        
        # Return the AI response
        return jsonify({
            "response": result["messages"][-1].content
        })
    except Exception as e:
        print(f"Error processing query: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/create_thread', methods=['POST'])
def create_thread():
    try:
        # Get all existing thread IDs
        existing_threads = get_all_thread_ids()
        
        # Create a new thread ID by incrementing the maximum existing thread ID
        new_thread_id = max(existing_threads) + 1 if existing_threads else 1
        
        # Initialize an empty conversation for the new thread
        db_handler.add_conversation(
            thread_id=new_thread_id,
            conversation=[]
        )
        
        return jsonify({"thread_id": new_thread_id})
    except Exception as e:
        print(f"Error creating thread: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/get_messages/<int:thread_id>')
def get_messages(thread_id):
    try:
        msgs = db_handler.get_conversation(thread_id)
        messages = []
        
        for msg in msgs:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            messages.append({
                "role": role,
                "content": msg.content
            })
            
        return jsonify({"messages": messages})
    except Exception as e:
        print(f"Error retrieving messages: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/generate_ppt/<int:thread_id>', methods=['POST'])
def generate_ppt(thread_id):
    try:
        # Get conversation for the thread
        msgs = db_handler.get_conversation(thread_id)
        
        if not msgs:
            return jsonify({"error": "No conversation found for this thread ID"}), 404
            
        # Generate PowerPoint presentation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(ppt_agent(msgs))
        loop.close()
        
        return jsonify({"success": True, "message": "PowerPoint presentation created successfully"})
    except Exception as e:
        print(f"Error generating PowerPoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/files', methods=['GET'])
def get_files():
    """Get all files in the vector database"""
    try:
        # Get stored documents
        stored_docs = db_handler.get_stored_documents()
        
        # Format the response
        files = []
        for doc_path in stored_docs:
            files.append({
                "path": doc_path,
                "name": os.path.basename(doc_path),
                "type": os.path.splitext(doc_path)[1][1:].upper()  # File extension without the dot
            })
            
        return jsonify({"files": files})
    except Exception as e:
        print(f"Error getting files: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/files/upload', methods=['POST'])
def upload_file():
    """Upload a file to the vector database"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
            
        file = request.files['file']
        
        # Check if file was selected
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
            
        # Check if file extension is allowed
        ext = os.path.splitext(file.filename)[1][1:].lower()  # Extension without the dot
        if ext not in ACCEPTED_EXTENSIONS:
            return jsonify({"error": f"File type not allowed. Accepted types: {', '.join(ACCEPTED_EXTENSIONS)}"}), 400
            
        # Save file to data folder
        filename = secure_filename(file.filename)
        file_path = os.path.join(DATA_FOLDER, filename)
        
        # Check if file already exists
        if os.path.exists(file_path):
            return jsonify({"error": f"File '{filename}' already exists"}), 400
            
        file.save(file_path)
        
        # Add file to vector database
        if db_handler.add_document(file_path):
            return jsonify({"success": True, "message": f"File '{filename}' uploaded and indexed successfully"})
        else:
            # Delete the file if it couldn't be added to the vector database
            os.remove(file_path)
            return jsonify({"error": "Error indexing file"}), 500
    except Exception as e:
        print(f"Error uploading file: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/files/delete', methods=['POST'])
def delete_file():
    """Delete a file from the vector database"""
    try:
        data = request.json
        file_path = data.get('file_path')
        
        if not file_path:
            return jsonify({"error": "No file path provided"}), 400
            
        # Delete from vector database
        if db_handler.delete_document(file_path):
            # Delete the file from disk if it exists
            if os.path.exists(file_path):
                os.remove(file_path)
            return jsonify({"success": True, "message": f"File deleted successfully"})
        else:
            return jsonify({"error": "Error deleting file from vector database"}), 500
    except Exception as e:
        print(f"Error deleting file: {e}")
        return jsonify({"error": str(e)}), 500

def query_with_graph(agent, msgs, thread_id):
    """
    Process a query through the LangGraph
    """
    print(f"\nQuery: {msgs[-1].content}")
    print("Processing query through Agent's workflow...")

    config = {"configurable": {"thread_id": thread_id}}

    # Initialize state with the query
    initial_state = {"context": [], "messages": msgs, "summary": "", "query": None, "more_research": False}

    # Run the graph
    result = agent.invoke(initial_state, config)

    return result

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    # Run Flask app
    app.run(debug=True, port=8000)
