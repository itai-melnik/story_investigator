import os
from typing import List
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.types import RetrieverResultItem

from src.llm_client import SafeLLMClient, PromptTooLongError
from src.utils import parse_story_xml

class Neo4jRagStrategy:
    def __init__(self, story_file_path: str, llm_client: SafeLLMClient):
        self.story_file_path = story_file_path
        self.llm_client = llm_client
        # 1. Connect to Neo4j
        self.uri = "neo4j://localhost:7687"
        self.auth = ("neo4j", "password")
        self.driver = GraphDatabase.driver(self.uri, auth=self.auth)
        
        # 2. Setup Embedder
        self.embedder = SentenceTransformerEmbeddings(model="all-MiniLM-L6-v2")
        
        # 3. Initialize DB (Ingest if empty)
        self._initialize_graph_data()
        
        # 4. Setup the Retriever
        retrieval_query = """
        MATCH (node)-[:SENT_BY]->(sender:Person)
        MATCH (node)-[:RECEIVED_BY]->(receiver:Person)
        RETURN 
            "From: " + sender.name + "\nTo: " + receiver.name + "\nMessage: " + node.body as text,
            score
        """
        
        self.retriever = VectorCypherRetriever(
            driver=self.driver,
            index_name="message-embeddings",
            embedder=self.embedder,
            retrieval_query=retrieval_query,
        )
        

    def close(self):
        self.driver.close()

    def _initialize_graph_data(self):
        """
        Checks if the graph is empty. If so, parses the XML and ingests data.
        """
        with self.driver.session() as session:
            # Check count
            result = session.run("MATCH (n:Message) RETURN count(n) as c")
            count = result.single()["c"]
            
            if count > 0:
                print(f"Neo4j already has {count} messages. Skipping ingestion.")
                return

            print("Graph is empty. Ingesting story data...")
            chunks = parse_story_xml(self.story_file_path)
            
            # 1. Create Vector Index
            # Dimensions for all-MiniLM-L6-v2 is 384
            index_query = """
            CREATE VECTOR INDEX `message-embeddings` IF NOT EXISTS
            FOR (m:Message) ON (m.embedding)
            OPTIONS {indexConfig: {
                `vector.dimensions`: 384,
                `vector.similarity_function`: 'cosine'
            }}
            """
            session.run(index_query)
            
            # 2. Prepare Data for Bulk Insert
            print("Computing embeddings for graph nodes...")
            texts = [c['body'] for c in chunks]
            embeddings = self.embedder.embed_query(texts)
            
            node_data = []
            for i, chunk in enumerate(chunks):
                node_data.append({
                    "sender": chunk['sender'],
                    "receiver": chunk['receiver'],
                    "body": chunk['body'],
                    "embedding": embeddings[i]
                })

            # 3. Run Bulk Cypher Insert
            ingest_query = """
            UNWIND $data AS row
            MERGE (s:Person {name: row.sender})
            MERGE (r:Person {name: row.receiver})
            CREATE (m:Message {body: row.body})
            SET m.embedding = row.embedding
            CREATE (m)-[:SENT_BY]->(s)
            CREATE (m)-[:RECEIVED_BY]->(r)
            """
            session.run(ingest_query, data=node_data)
            print("Ingestion complete.")

    def ask(self, question: str) -> str:
        # 1. Retrieve using GraphRAG
        # This will run Vector Search -> find 'node' -> run Cypher to find sender/receiver
        results: List[RetrieverResultItem] = self.retriever.search(
            query_text=question, 
            top_k=5
        ).items

        # 2. Format Results into String Context
        formatted_chunks = []
        for res in results:
            # VectorCypherRetriever puts the 'text' column from our query into content
            formatted_chunks.append(str(res.content))

        full_context = "\n\n".join(formatted_chunks)
        
        # Simple safety truncate
        if len(full_context) > 2500:
            full_context = full_context[:2500] + "...(truncated)"

        try:
            return self.llm_client.generate_answer(full_context, question)
        except PromptTooLongError:
            return "System Error: Retrieved context was too long."