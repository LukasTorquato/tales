"""
Evaluation script for RAG and PowerPoint generation agents.

This script runs evaluation on the RAG agent and PowerPoint generation agent
using a set of predefined queries or custom queries provided by the user.
"""
import argparse
import json
import asyncio
from pathlib import Path

from tales.evaluation import RAGEvaluator, PowerPointEvaluator, run_batch_evaluation
from tales.agent import agent
from tales.db_handler import ChromaDBHandler
from tales.config import DB_PATH
from tales.utils import get_available_docs
from langchain_core.messages import HumanMessage


# Sample evaluation queries
DEFAULT_QUERIES = [
    "What are the key concepts in information theory?",
    "Explain the challenges of data overload in modern society",
    "What are the main differences between structured and unstructured data?",
    "How does language modeling work?",
    "What is the relationship between information theory and artificial intelligence?"
]


def evaluate_single_query(query: str, generate_ppt: bool = False):
    """Evaluate a single query and optionally generate a PowerPoint.
    
    Args:
        query: The query to evaluate
        generate_ppt: Whether to generate a PowerPoint presentation
    """
    print(f"\nEvaluating query: {query}")
    
    # Initialize evaluators
    rag_evaluator = RAGEvaluator()
    
    # Evaluate RAG
    rag_metrics, messages = rag_evaluator.evaluate_rag_query(query)
    
    # Print RAG metrics
    print("\n=== RAG Evaluation Results ===")
    print(f"Response Time: {rag_metrics.response_time:.2f} seconds")
    print(f"Context Relevance: {rag_metrics.context_relevance_score:.1f}/10")
    print(f"Answer Correctness: {rag_metrics.answer_correctness_score:.1f}/10")
    print(f"Answer Completeness: {rag_metrics.answer_completeness_score:.1f}/10")
    print(f"Hallucination Score: {rag_metrics.hallucination_score:.1f}/10 (lower is better)")
    print(f"Documents Retrieved: {rag_metrics.num_documents_retrieved}")
    print(f"Research Iterations: {rag_metrics.num_research_iterations}")
    
    # Generate and evaluate PowerPoint if requested
    if generate_ppt:
        print("\nGenerating PowerPoint presentation...")
        ppt_evaluator = PowerPointEvaluator()
        try:
            ppt_metrics = asyncio.run(ppt_evaluator.evaluate_ppt_generation(messages))
            
            # Print PowerPoint metrics
            print("\n=== PowerPoint Evaluation Results ===")
            print(f"Generation Time: {ppt_metrics.generation_time:.2f} seconds")
            print(f"Number of Slides: {ppt_metrics.slides_count}")
            print(f"Avg Content Per Slide: {ppt_metrics.avg_content_per_slide:.1f} characters")
            print(f"Content Coverage: {ppt_metrics.content_coverage_score:.1f}/10")
            print(f"Design Quality: {ppt_metrics.design_quality_score:.1f}/10")
            print(f"Organization: {ppt_metrics.organization_score:.1f}/10")
            
        except Exception as e:
            print(f"Error evaluating PowerPoint: {e}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG and PowerPoint generation agents")
    parser.add_argument(
        "--query", "-q", 
        help="Single query to evaluate"
    )
    parser.add_argument(
        "--batch", "-b", 
        action="store_true", 
        help="Run batch evaluation on predefined queries"
    )
    parser.add_argument(
        "--ppt", "-p", 
        action="store_true", 
        help="Generate and evaluate PowerPoint presentations"
    )
    parser.add_argument(
        "--output", "-o", 
        default="evaluation_results.json",
        help="Output file for evaluation results"
    )
    parser.add_argument(
        "--input", "-i", 
        help="Input file with custom evaluation queries (one per line)"
    )
    
    args = parser.parse_args()
    
    # Check available documents
    db_handler = ChromaDBHandler(persist_directory=DB_PATH)
    stored_docs = db_handler.get_stored_documents()
    print(f"Found {len(stored_docs)} documents in the vector store.")
    for doc in stored_docs:
        print(f" - {doc}")
    
    # Handle batch evaluation
    if args.batch:
        queries = DEFAULT_QUERIES
        
        if args.input:
            try:
                with open(args.input, 'r') as f:
                    custom_queries = [line.strip() for line in f if line.strip()]
                if custom_queries:
                    queries = custom_queries
                    print(f"Loaded {len(queries)} custom queries from {args.input}")
            except Exception as e:
                print(f"Error loading custom queries: {e}")
        
        print(f"Running batch evaluation on {len(queries)} queries...")
        results = run_batch_evaluation(queries, args.output)
        
        print(f"\nEvaluation complete. Results saved to {args.output}")
        
    # Handle single query evaluation
    elif args.query:
        evaluate_single_query(args.query, args.ppt)
    
    # No query specified
    else:
        print("Please provide a query with --query or run batch evaluation with --batch")
        print("For custom queries, provide a file with --input")


if __name__ == "__main__":
    main()
