import time
from typing import  List, Any, Tuple
from dataclasses import dataclass, asdict

from langchain_core.messages import AnyMessage, HumanMessage, AIMessage

from tales.agent import agent
from tales.db_handler import ChromaDBHandler
from tales.config import DB_PATH, llm

# Define evaluation metrics for RAG agent
@dataclass
class RAGMetrics:
    """Metrics for evaluating RAG agent performance."""
    query: str
    response_time: float  # In seconds
    context_relevance_score: float  # 0-10 score
    answer_correctness_score: float  # 0-10 score
    answer_completeness_score: float  # 0-10 score
    hallucination_score: float  # 0-10 score (lower is better)
    num_documents_retrieved: int
    
    def to_dict(self):
        """Convert metrics to dictionary."""
        return asdict(self)
class RAGEvaluator:
    """Evaluator for RAG agent performance."""
    
    def __init__(self, db_handler: ChromaDBHandler = None):
        """Initialize RAG evaluator.
        
        Args:
            db_handler: ChromaDB handler for accessing conversations
        """
        self.db_handler = db_handler or ChromaDBHandler(persist_directory=DB_PATH)
        
        # Initialize evaluation LLM (using Gemini for evaluation)
        self.eval_llm = llm

    def evaluate_response(self, query: str, messages: List[AnyMessage], 
                          context_docs: List[Any]) -> RAGMetrics:
        """Evaluate the RAG agent's response.
        
        Args:
            query: Original user query
            messages: List of messages including the response
            context_docs: Documents retrieved as context
            iterations: Number of research iterations
            
        Returns:
            RAGMetrics object with evaluation scores
        """
        # Extract the response (last AI message)
        response = next((msg.content for msg in reversed(messages) 
                         if isinstance(msg, AIMessage)), "")

        # Get context text from documents
        context_texts = [doc.page_content for doc in context_docs] if context_docs else []
        
        print("Query: ", query.content)
        print("Response: ", response)
        print("Context: ", context_texts)

        # Evaluate context relevance
        context_relevance = self._evaluate_context_relevance(query, context_texts)
        
        # Evaluate answer quality
        answer_correctness = self._evaluate_answer_correctness(query, response, context_texts)
        
        # Evaluate answer completeness
        answer_completeness = self._evaluate_answer_completeness(query, response)
        
        # Evaluate for hallucinations
        hallucination_score = self._evaluate_hallucinations(response, context_texts)
        
        return RAGMetrics(
            query=query,
            response_time=0.0,  # Will be set by the evaluate_rag_query method
            context_relevance_score=context_relevance,
            answer_correctness_score=answer_correctness,
            answer_completeness_score=answer_completeness,
            hallucination_score=hallucination_score,
            num_documents_retrieved=len(context_docs)
        )
    
    def evaluate_rag_query(self, query: str, thread_id: int = None) -> Tuple[RAGMetrics, List[AnyMessage]]:
        """Run and evaluate a query through the RAG agent.
        
        Args:
            query: User query to evaluate
            thread_id: Optional thread ID for conversation context
            
        Returns:
            Tuple of (RAGMetrics, messages)
        """
        # Create a new thread ID if not provided
        thread_id = thread_id or int(time.time())
        
        # Get existing messages or start fresh
        messages = self.db_handler.get_conversation(thread_id) if thread_id else []
        
        # Add the query
        messages.append(HumanMessage(content=query))
        
        # Track iterations
        context_docs = []
        # Start timing
        start_time = time.time()
        
        # Initialize state with the query
        initial_state = {
            "context": [],
            "messages": messages,
            "summary": "",
            "query": None,
            "more_research": False
        }
        
        # Run the graph with tracking
        config = {"configurable": {"thread_id": thread_id}}
        result = agent.invoke(initial_state, config)
        # Calculate time
        end_time = time.time()
        response_time = end_time - start_time
        
        # Evaluate the results
        metrics = self.evaluate_response(
            query=result["query"],
            messages=result["messages"],
            context_docs=result["context"]
        )
        
        # Update the response time
        metrics.response_time = response_time
        
        return metrics, result["messages"]
    
    def _evaluate_context_relevance(self, query: str, context_texts: List[str]) -> float:
        """Evaluate the relevance of retrieved context to the query.
        
        Args:
            query: The user query
            context_texts: List of context document texts
            
        Returns:
            Score from 0-10 on context relevance
        """
        if not context_texts:
            return 0.0
            
        combined_context = "\n\n".join(context_texts)  
            
        eval_prompt = [
            HumanMessage(content=f"""You are an expert evaluator for RAG (Retrieval-Augmented Generation) systems.
                        
            For the following user query, evaluate how relevant the retrieved context is on a scale from 0 to 10, where:
            - 0: Completely irrelevant
            - 5: Somewhat relevant but missing key information
            - 10: Highly relevant and contains all needed information

            Query: "{query}"

            Retrieved Context:
            {combined_context}

            Provide your rating as a single number between 0-10 without explanation or other text.
            """)
        ]
        
        try:
            response = self.eval_llm.invoke(eval_prompt).content
            # Extract the numeric score
            score = float(response.strip())
            return min(max(score, 0.0), 10.0)  # Ensure it's between 0-10
        except:
            # Default score if evaluation fails
            return 5.0
    
    def _evaluate_answer_correctness(self, query: str, response: str, context_texts: List[str]) -> float:
        """Evaluate the correctness of the answer based on the context.
        
        Args:
            query: The user query
            response: The agent's response
            context_texts: List of context document texts
            
        Returns:
            Score from 0-10 on answer correctness
        """
        if not context_texts:
            print("No context provided for correctness evaluation.")
            return 5.0  # Middle score if no context
            
        # Limit context size for evaluation
        combined_context = "\n\n".join(context_texts)
            
        eval_prompt = [
            HumanMessage(content=f"""You are an expert evaluator for RAG (Retrieval-Augmented Generation) systems.
            
            For the following user query and context, evaluate how factually correct the response is on a scale from 0 to 10, where:
            - 0: Completely incorrect, contains false information
            - 5: Mixture of correct and incorrect information
            - 10: Completely factually correct based on the context
            - Remember that the context is ranked by similarity to the query
            Query: "{query}"

            Context (ground truth):
            {combined_context}

            Response to evaluate:
            {response}

            Provide your rating as a single number between 0-10 without explanation or other text.
            """)
        ]
        
        try:
            response = self.eval_llm.invoke(eval_prompt).content
            # Extract the numeric score
            score = float(response.strip())
            return min(max(score, 0.0), 10.0)  # Ensure it's between 0-10
        except:
            # Default score if evaluation fails
            return 5.0
    
    def _evaluate_answer_completeness(self, query: str, response: str) -> float:
        """Evaluate the completeness of the answer.
        
        Args:
            query: The user query
            response: The agent's response
            
        Returns:
            Score from 0-10 on answer completeness
        """
        eval_prompt = [
            HumanMessage(content=f"""You are an expert evaluator for question answering systems.
                    
            For the following user query, evaluate how completely the response answers the question on a scale from 0 to 10, where:
            - 0: Does not answer the question at all
            - 5: Partially answers the question but misses important aspects
            - 10: Fully and comprehensively answers the question

            Query: "{query}"

            Response:
            {response}

            Provide your rating as a single number between 0-10 without explanation or other text.
            """)
        ]
        
        try:
            response = self.eval_llm.invoke(eval_prompt).content
            # Extract the numeric score
            score = float(response.strip())
            return min(max(score, 0.0), 10.0)  # Ensure it's between 0-10
        except:
            # Default score if evaluation fails
            return 5.0
    
    def _evaluate_hallucinations(self, response: str, context_texts: List[str]) -> float:
        """Evaluate the response for hallucinations (information not in context).
        
        Args:
            response: The agent's response
            context_texts: List of context document texts
            
        Returns:
            Score from 0-10 on hallucination (lower is better)
        """
        if not context_texts:
            return 5.0  # Middle score if no context
            
        # Limit context size for evaluation
        combined_context = "\n\n".join(context_texts)  # All docs

        eval_prompt = [
            HumanMessage(content=f"""You are an expert evaluator for RAG (Retrieval-Augmented Generation) systems.
                            
                For the following response and context, evaluate the degree of hallucination on a scale from 0 to 10, where:
                - 0: No hallucination, response only contains information from the context
                - 5: Some hallucination, response includes some information not in the context
                - 10: Complete hallucination, response is unrelated to or contradicts the context

                Context (ground truth):
                {combined_context}

                Response to evaluate:
                {response}

                Provide your rating as a single number between 0-10 without explanation or other text.
                """)
        ]
        
        try:
            response = self.eval_llm.invoke(eval_prompt).content
            # Extract the numeric score
            score = float(response.strip())
            return min(max(score, 0.0), 10.0)  # Ensure it's between 0-10
        except:
            # Default score if evaluation fails
            return 5.0
