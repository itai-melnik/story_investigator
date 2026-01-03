# test_run.py
from src.naive_rag import NaiveRagStrategy
from src.llm_client import SafeLLMClient
from dotenv import load_dotenv
import os

load_dotenv()

llm_client = SafeLLMClient(os.getenv("OPENAI_API_KEY"))
# Initialize (downloads model on first run)
investigator = NaiveRagStrategy("data/story.xml", llm_client)

question = "Who sees a fire boat?"
print(f"Question: {question}")

answer = investigator.ask(question)
print(f"Answer: {answer}")