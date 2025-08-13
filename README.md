# Tales - AI Document Assistant

### Installing Ollama and Pulling Models

- Go to https://ollama.com/download
- Download Ollama and install
- Run Ollama, open CMD and run:
  - ollama pull llama3.2:3b (Regular model)
  - ollama pull deepseek-r1 (Reasoning model)
  - ollama pull nomic-embed-text (Embbeding model)
- Get your Google Gemini API Key at https://aistudio.google.com/

### Installing dependencies

- Run pip install .

### Organizing Data

- Create a folder in the base directory named "data"
- Put in all wanted pdf, csv files in that folder

### Running

- CLI: Run app.py
- Web UI: Start the web server and open the browser
  - python web_app.py
  - Visit http://localhost:8000
  - The web interface provides:
    - A conversation selector on the left
    - A chat interface in the center
    - A file viewer on the right showing documents in your vector database
  - You can create new conversations, continue existing ones, and generate PowerPoint presentations.

### Evaluation System

The project includes a comprehensive evaluation system for both the RAG agent and PowerPoint generation capabilities.

#### Command-line Evaluation

```bash
# Evaluate a single query
python evaluate.py --query "What are the key concepts in information theory?"

# Generate and evaluate a PowerPoint presentation for a query
python evaluate.py --query "Explain the main principles of information theory" --ppt

# Run batch evaluation on predefined queries
python evaluate.py --batch

# Run batch evaluation on custom queries from a file
python evaluate.py --batch --input my_queries.txt --output my_results.json
```

#### Notebook Evaluation

For interactive evaluation and visualization:

1. Open the `evaluation_notebook.ipynb` file in Jupyter Notebook or VS Code
2. Follow the steps in the notebook to evaluate the agents and visualize the results

#### Evaluation Metrics

- **RAG Agent Metrics**:

  - Response Time: Time taken to generate the response
  - Context Relevance: How relevant the retrieved documents are to the query (0-10)
  - Answer Correctness: How factually correct the answer is based on the context (0-10)
  - Answer Completeness: How completely the answer addresses the query (0-10)
  - Hallucination Score: Degree of hallucination in the response (0-10, lower is better)
  - Number of Documents Retrieved: Count of documents used as context
  - Research Iterations: Number of research iterations performed

- **PowerPoint Metrics**:
  - Generation Time: Time taken to generate the presentation
  - Slides Count: Number of slides in the presentation
  - Content Coverage: How well the presentation covers the content from the conversation (0-10)
  - Design Quality: Quality of the presentation design (0-10)
  - Organization: How well-organized the presentation is (0-10)
