import time
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import SentenceTransformerEmbeddings
from src.llm_client import SafeLLMClient, PromptTooLongError
from src.utils import parse_story_xml

class Neo4jRagStrategy:
    def __init__(self, story_file_path: str, llm_client: SafeLLMClient = None):
        self.story_file_path = story_file_path
        self.llm_client = llm_client if llm_client else SafeLLMClient()
        
        self.uri = "neo4j://localhost:7687"
        self.auth = ("neo4j", "password")
        self.driver = GraphDatabase.driver(self.uri, auth=self.auth)
        
        self.embedder = SentenceTransformerEmbeddings(model="all-MiniLM-L6-v2")
        
        self._initialize_graph_data()

    def close(self):
        self.driver.close()

    def _initialize_graph_data(self):
        with self.driver.session() as session:
            # Check if data exists
            result = session.run("MATCH (n:Message) RETURN count(n) as c")
            count = result.single()["c"]
            
            if count > 0:
                print(f"Neo4j already has {count} messages. Checking index...")
                session.run("CALL db.awaitIndexes()")
                return

            print("Graph is empty. Ingesting story data...")
            chunks = parse_story_xml(self.story_file_path)
            
            # 1. Create Vector Index
            index_query = """
            CREATE VECTOR INDEX `message-embeddings` IF NOT EXISTS
            FOR (m:Message) ON (m.embedding)
            OPTIONS {indexConfig: {
                `vector.dimensions`: 384,
                `vector.similarity_function`: 'cosine'
            }}
            """
            session.run(index_query)
            
            # 2. Embed & Ingest
            print("Computing embeddings...")
            texts = [c['body'] for c in chunks]
            
            # FIX: Use the underlying model directly for batch encoding to be safe
            # The wrapper API can be inconsistent with lists.
            embeddings = self.embedder.model.encode(texts)
            
            node_data = []
            for i, chunk in enumerate(chunks):
                node_data.append({
                    "sender": chunk['sender'],
                    "receiver": chunk['receiver'],
                    "body": chunk['body'],
                    # Ensure it is a standard python list of floats
                    "embedding": embeddings[i].tolist() 
                })

            print("Inserting Graph Data...")
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
            
            print("Waiting for Index...")
            session.run("CALL db.awaitIndexes()")
            print("Ingestion complete.")

    def ask(self, question: str) -> str:
        # 1. Embed the Question
        question_embedding = self.embedder.embed_query(question)
        
        # 2. Run Raw Cypher Vector Search
        search_query = """
        CALL db.index.vector.queryNodes('message-embeddings', 5, $embedding)
        YIELD node, score
        
        OPTIONAL MATCH (node)-[:SENT_BY]->(sender:Person)
        OPTIONAL MATCH (node)-[:RECEIVED_BY]->(receiver:Person)
        
        RETURN 
            "From: " + coalesce(sender.name, "Unknown") + 
            "\nTo: " + coalesce(receiver.name, "Unknown") + 
            "\nMessage: " + coalesce(node.body, "") as text,
            score
        """
        
        with self.driver.session() as session:
            result = session.run(search_query, embedding=question_embedding)
            records = [record for record in result]

        if not records:
            return "No relevant info found in story."

        formatted_chunks = []
        for record in records:
            formatted_chunks.append(record["text"])

        full_context = "\n\n".join(formatted_chunks)
        
        if len(full_context) > 2500:
            full_context = full_context[:2500] + "...(truncated)"

        try:
            return self.llm_client.generate_answer(full_context, question)
        except PromptTooLongError:
            return "System Error: Retrieved context was too long."