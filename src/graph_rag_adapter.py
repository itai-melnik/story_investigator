import os
import pandas as pd
import asyncio
from graphrag.query.structured_search.local_search.local_search import LocalSearch
from graphrag.query.structured_search.local_search.mixed_search import MixedSearch
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import (
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
    read_indexer_covariates,
)
from graphrag.query.input.loaders.dfs import (
    store_entity_semantic_embeddings,
)
from graphrag.query.llm.oai.chat_openai import ChatOpenAI
from graphrag.query.llm.oai.embedding import OpenAIEmbedding
from graphrag.vector_stores.lancedb import LanceDBVectorStore

class MicrosoftGraphRagStrategy:
    def __init__(self, root_dir="./ms_graphrag"):
        print(f"Loading Microsoft GraphRAG artifacts from {root_dir}...")
        
        # 1. Determine Output Folder (timestamps)
        # We grab the most recent output folder
        output_dir = os.path.join(root_dir, "output")
        if not os.path.exists(output_dir):
            raise FileNotFoundError(f"GraphRAG output not found in {output_dir}. Did you run 'python -m graphrag.index'?")
            
        subdirs = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
        latest_output = max(subdirs, key=os.path.getmtime)
        artifacts_path = os.path.join(latest_output, "artifacts")
        
        print(f"Using artifacts from: {latest_output}")

        # 2. Load Parquet Tables
        self.entities = pd.read_parquet(os.path.join(artifacts_path, "create_final_entities.parquet"))
        self.relationships = pd.read_parquet(os.path.join(artifacts_path, "create_final_relationships.parquet"))
        self.reports = pd.read_parquet(os.path.join(artifacts_path, "create_final_community_reports.parquet"))
        self.text_units = pd.read_parquet(os.path.join(artifacts_path, "create_final_text_units.parquet"))
        
        # 3. Setup LLM & Embedder (Using standard OpenAI)
        api_key = os.environ.get("OPENAI_API_KEY")
        self.llm = ChatOpenAI(
            api_key=api_key,
            model="gpt-4o-mini",
            api_base=None, 
            api_type=None,
        )
        self.text_embedder = OpenAIEmbedding(
            api_key=api_key,
            api_base=None,
            api_type=None,
            model="text_embedding_3_small",
            deployment_name="text_embedding_3_small",
            max_retries=20,
        )

        # 4. Setup Vector Store (LanceDB is default for GraphRAG)
    
        description_embedding_store = LanceDBVectorStore(
            collection_name="entity_description_embeddings",
        )
        # Connect to the on-disk lanceDB (created during indexing)
        description_embedding_store.connect(db_uri=os.path.join(latest_output, "lancedb"))

        # 5. Initialize Local Search Engine
        # "Local Search" is best for "Who/What/Where" specific questions.
        self.engine = LocalSearch(
            llm=self.llm,
            context_builder_params={
                "text_unit_prop": 0.5,
                "community_prop": 0.1,
                "conversation_history_max_turns": 5,
                "conversation_history_user_turns_only": True,
                "top_k_mapped_entities": 10,
                "top_k_relationships": 10,
                "include_entity_rank": True,
            },
            completion_llm_params={
                "max_tokens": 1000,
                "temperature": 0.0,
            },
            search_engine_params={
                "max_tokens": 3000, # Trying to respect your 3000 limit concept
            },
            # Data Connectors
            entities=self.entities,
            relationships=self.relationships,
            reports=self.reports,
            text_units=self.text_units,
            entities_embedding_store=description_embedding_store, # Pass the vector store
            text_embedder=self.text_embedder,
            response_type="multiple paragraphs", # Format of the answer
        )

    def ask(self, question: str) -> str:
        # The engine is async, so we run it in a loop
        result = asyncio.run(self.engine.asearch(question))
        
        # GraphRAG returns a result object with .response
        return result.response