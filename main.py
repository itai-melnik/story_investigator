import argparse
import sys
import os
from dotenv import load_dotenv
from src.naive_rag import NaiveRagStrategy
from src.neo4j_rag_adapter import Neo4jRagStrategy
from src.graph_rag_adapter import MicrosoftGraphRagStrategy
from src.llm_client import SafeLLMClient

load_dotenv()

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))



def main():

    llm_client = SafeLLMClient(os.getenv("OPENAI_API_KEY"))

    parser = argparse.ArgumentParser(description="AI Story Investigator")
    parser.add_argument(
        '--strategy', 
        choices=['naive', 'neo4j', 'graphrag'], 
        default='naive',
        help="Choose the RAG strategy: 'naive' (default), 'neo4j', or 'graphrag'"
    )
    parser.add_argument(
        '--file', 
        type=str, 
        default='data/story.xml',
        help="Path to the story file (default: data/story.xml). Note: Not used for 'graphrag' strategy."
    )
    parser.add_argument(
        '--graphrag-root',
        type=str,
        default='./ms_graphrag',
        help="Root directory for GraphRAG artifacts (default: ./ms_graphrag). Only used with 'graphrag' strategy."
    )
    
    args = parser.parse_args()

    print(f"--- AI Investigator 1.0 (Strategy: {args.strategy.upper()}) ---")
    if args.strategy != 'graphrag':
        print(f"Loading story from: {args.file}")
    else:
        print(f"Loading GraphRAG artifacts from: {args.graphrag_root}")

    investigator = None

    try:
        # Strategy Selection Pattern
        if args.strategy == 'naive':
            investigator = NaiveRagStrategy(args.file, llm_client)
        elif args.strategy == 'neo4j':
            print("Connecting to Neo4j Container...")
            investigator = Neo4jRagStrategy(args.file, llm_client)
        elif args.strategy == 'graphrag':
            print("Initializing Microsoft GraphRAG...")
            investigator = MicrosoftGraphRagStrategy(root_dir=args.graphrag_root)
        
        print("\nSystem Ready. Ask me any question about the story (or type 'quit' to exit).")
        print("-" * 50)

      
        while True:
            try:
                user_input = input("\nYour Question >> ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Shutting down Investigator.")
                    break
                
                if not user_input:
                    continue

                print("Thinking...")
                answer = investigator.ask(user_input)
                
                print(f"\n{answer}")
                print("-" * 50)

            except KeyboardInterrupt:
                print("\nInterrupted. Exiting.")
                break
            except Exception as e:
                print(f"\n[Error] An error occurred during query: {e}")

    except Exception as e:
        print(f"\n[Fatal Error] Could not initialize the investigator: {e}")
        if args.strategy == 'neo4j':
            print("Tip: Make sure your Docker container is running: 'docker run ... neo4j:5.26.0'")
        elif args.strategy == 'graphrag':
            print("Tip: Make sure you have run 'graphrag index --root ./ms_graphrag' to create the knowledge graph.")
            print("     Also ensure OPENAI_API_KEY or GRAPHRAG_API_KEY is set in your environment.")
    finally:
    
        if investigator and hasattr(investigator, 'close'):
            investigator.close()

if __name__ == "__main__":
    main()