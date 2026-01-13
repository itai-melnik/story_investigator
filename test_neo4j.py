from src.neo4j_rag_adapter import Neo4jRagStrategy
import sys
from src.llm_client import SafeLLMClient
from dotenv import load_dotenv
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

load_dotenv()

# Ensure you have the OPENAI_API_KEY set for the LLM part
try:
    print("Initializing Neo4j Strategy...")
    llm_client = SafeLLMClient(os.getenv("OPENAI_API_KEY"))
    graph_rag = Neo4jRagStrategy("data/story.xml", llm_client)
    
    question = "Who sees a fire boat?"
    print(f"\nAsking: {question}")
    
    answer = graph_rag.ask(question)
    print(f"\nAnswer:\n{answer}")
    
    graph_rag.close()
    
except Exception as e:
    print(f"Error: {e}")
    print("Did you start Docker? 'docker run ...'")