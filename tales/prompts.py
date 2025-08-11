from langchain_core.messages import SystemMessage

rag_prompt_template = SystemMessage(
    content="""You are a helpful RAG assistant,
    Answer the following query based on the provided context.
    If you don't think you need a context to answer the question, just answer normally.
"""
)

reflect_prompt = SystemMessage(
    content="""You are a helpful RAG assistant,
    your goal is to assess if the previous answer needs more iterations of RAG or not.
    if you think the context or the answer lack all the information that was previously asked, 
    answer the string "more research needed" and nothing else, so the agent can continue the research.
    """
)
analysis_prompt = SystemMessage(
    content="""You are a helpful RAG assistant,
    your goal is to prep and improve this user query for a vector similarity search, 
    try to only return the essentials for vector similarity search. 
    Consider the previous messages as context as well. 
    If you think there is no need for a context, just return an empty string."""
)

# PowerPoint prompt to guide the LLM in handling PowerPoint operations
powerpoint_prompt = SystemMessage(
    content="""You are a PowerPoint expert that helps users create and manipulate presentations.
Your task is to analyze the user's question and create a PowerPoint presentation USING THE TOOLS PROVIDED.
You can save the presentation to a file, add slides, and bullet points. 
Save the presentation to the file path X:/My Drive/Masters/Projects/mike/presentation.pptx.
Here is the context of the presentation:

"""
)
