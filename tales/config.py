from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.runnables.config import RunnableConfig
from mcp import StdioServerParameters
from dotenv import load_dotenv

############### CONFIG FLAGS ############
USE_REASONING = False  # Set to True to use reasoning model
LOCAL_LLMS = False  # Set to True to use local LLMs (Ollama) instead of Gemini
DB_PATH = "database/tales"  # Path to the database
DATA_FOLDER = "data/"  # Folder containing data files
DOCS_RETRIEVED = 24  # Number of documents to retrieve for each query
RUNNABLE_CONFIG = RunnableConfig(recursion_limit=100) # Increase default recursion limit for agentic LLM tool recursion
ACCEPTED_EXTENSIONS = [
    "pdf",
    "txt",  # Uncommented to accept txt files
    # "docx",  # Added to accept docx files
    # "pptx",  # Added to accept pptx files
    "csv",  # Added to accept csv files
    "xlsx",  # Added to accept xlsx files
]

load_dotenv()

# Gemini LLM
if LOCAL_LLMS:
    # Ollama LLM
    llm = ChatOllama(
        model="gemma3n:e4b",
        temperature=0,
        max_tokens=4096,
        streaming=True,
        callbacks=[],
    )

    # Ollama Embeddings
    embeddings_model = OllamaEmbeddings(model="nomic-embed-text")

else:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        max_tokens=512000,
        timeout=None,
        max_retries=2,
    )

    # Gemini Embeddings
    embeddings_model = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        temperature=0,
        max_tokens=2048,
        max_retries=2,
        timeout=None,
    )

server_params = StdioServerParameters(
    command="C:\\Users\\lukas\\Documents\\Projects\\MCP Servers\\Office-PowerPoint-MCP-Server\\.venv\\Scripts\\python.exe",
    args=[
        "C:/Users/lukas/Documents/Projects/MCP Servers/Office-PowerPoint-MCP-Server/ppt_mcp_server.py"
    ],
    env={
        "PPT_TEMPLATE_PATH": "C:/Users/lukas/Documents/Projects/MCP Servers/Office-PowerPoint-MCP-Server/templates"
    },
)