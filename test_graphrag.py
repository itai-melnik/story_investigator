import sys
import os

# Ensure src is visible
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.graph_rag_adapter import MicrosoftGraphRagStrategy

def test_ms_graphrag():
    print("--- Testing Microsoft GraphRAG Strategy ---")
    
    # Check for API Key
    if not os.environ.get("GRAPHRAG_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: Please set GRAPHRAG_API_KEY or OPENAI_API_KEY environment variable.")
        return

    try:
        # 1. Initialize
        # This will load the parquet files from ./ms_graphrag/output/...
        investigator = MicrosoftGraphRagStrategy(root_dir="./ms_graphrag")
    except Exception as e:
        print(f"Initialization Failed: {e}")
        print("Did you run 'graphrag index --root ./ms_graphrag'?")
        return

    # 2. Ask Question
    question = "Who sees a fire boat?"
    print(f"\nQuestion: {question}")
    print("-" * 40)
    print("Thinking (this involves Local Search and LLM generation)...")

    try:
        answer = investigator.ask(question)
        print(f"\nAnswer:\n{answer}")
    except Exception as e:
        print(f"Query Failed: {e}")

if __name__ == "__main__":
    test_ms_graphrag()