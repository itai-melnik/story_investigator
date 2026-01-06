import os
import pandas as pd
import asyncio
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.structured_search.local_search.mixed_context import LocalSearchMixedContext
from graphrag.query.indexer_adapters import (
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.language_model.providers.litellm.chat_model import LitellmChatModel
from graphrag.language_model.providers.litellm.embedding_model import LitellmEmbeddingModel
from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.config.models.vector_store_schema_config import VectorStoreSchemaConfig
from graphrag.vector_stores.lancedb import LanceDBVectorStore
from graphrag.tokenizer.tiktoken_tokenizer import TiktokenTokenizer
from graphrag.config.enums import ModelType


class MicrosoftGraphRagStrategy:
    def __init__(self, root_dir="./ms_graphrag"):
        print(f"Loading Microsoft GraphRAG artifacts from {root_dir}...")
        
        # 1. Determine Output Folder
        output_dir = os.path.join(root_dir, "output")
        if not os.path.exists(output_dir):
            raise FileNotFoundError(f"GraphRAG output not found in {output_dir}.")
        
        # Check if parquet files are directly in output_dir or in a subdirectory
        if os.path.exists(os.path.join(output_dir, "entities.parquet")):
            # New format: files directly in output
            artifacts_path = output_dir
            lancedb_path = os.path.join(output_dir, "lancedb")
        else:
            # Old format: files in timestamped subdirectory/artifacts
            subdirs = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
            if not subdirs:
                raise FileNotFoundError(f"No run folders found in {output_dir}")
            latest_output = max(subdirs, key=os.path.getmtime)
            artifacts_path = os.path.join(latest_output, "artifacts")
            lancedb_path = os.path.join(latest_output, "lancedb")
        
        print(f"Using artifacts from: {artifacts_path}")

        # 2. Load Parquet Tables
        # Try new naming convention first, fall back to old naming
        try:
            entity_df = pd.read_parquet(os.path.join(artifacts_path, "entities.parquet"))
            report_df = pd.read_parquet(os.path.join(artifacts_path, "community_reports.parquet"))
            relationship_df = pd.read_parquet(os.path.join(artifacts_path, "relationships.parquet"))
            text_unit_df = pd.read_parquet(os.path.join(artifacts_path, "text_units.parquet"))
            nodes_df = pd.read_parquet(os.path.join(artifacts_path, "communities.parquet"))
        except FileNotFoundError:
            # Fall back to old naming convention
            entity_df = pd.read_parquet(os.path.join(artifacts_path, "create_final_entities.parquet"))
            report_df = pd.read_parquet(os.path.join(artifacts_path, "create_final_community_reports.parquet"))
            relationship_df = pd.read_parquet(os.path.join(artifacts_path, "create_final_relationships.parquet"))
            text_unit_df = pd.read_parquet(os.path.join(artifacts_path, "create_final_text_units.parquet"))
            nodes_df = pd.read_parquet(os.path.join(artifacts_path, "create_final_nodes.parquet"))
        
        # 3. Process Dataframes for ContextBuilder
        community_level = 2
        self.entities = read_indexer_entities(entity_df, nodes_df, community_level)
        self.reports = read_indexer_reports(report_df, nodes_df, community_level)
        self.relationships = read_indexer_relationships(relationship_df)
        self.text_units = read_indexer_text_units(text_unit_df)

        # 4. Setup LLM & Embedder using new LiteLLM-based models
        api_key = os.environ.get("GRAPHRAG_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not api_key: 
            raise ValueError("API Key not found (GRAPHRAG_API_KEY or OPENAI_API_KEY)")

        # Create language model config for chat
        chat_config = LanguageModelConfig(
            type=ModelType.Chat,
            model="gpt-4o-mini",
            model_provider="openai",
            api_key=api_key,
            max_tokens=2000,
            temperature=0.0,
            max_retries=20,
        )
        
        # Create language model config for embeddings
        embedding_config = LanguageModelConfig(
            type=ModelType.Embedding,
            model="text-embedding-3-small",
            model_provider="openai",
            api_key=api_key,
            max_retries=20,
        )

        self.llm = LitellmChatModel(name="chat", config=chat_config)
        self.text_embedder = LitellmEmbeddingModel(name="embedding", config=embedding_config)
        self.tokenizer = TiktokenTokenizer(encoding_name="cl100k_base")

        # 5. Connect to LanceDB (Vector Store)
        vector_store_config = VectorStoreSchemaConfig(
            index_name="default-entity-description"
        )
        description_embedding_store = LanceDBVectorStore(
            vector_store_schema_config=vector_store_config,
        )
        description_embedding_store.connect(db_uri=lancedb_path)

        # 6. Build the Context Builder (The "Brain" of local search)
        self.context_builder = LocalSearchMixedContext(
            entities=self.entities,
            entity_text_embeddings=description_embedding_store,
            text_embedder=self.text_embedder,
            community_reports=self.reports,
            text_units=self.text_units,
            relationships=self.relationships,
            embedding_vectorstore_key=EntityVectorStoreKey.ID,
            tokenizer=self.tokenizer,
        )

        # 7. Initialize Local Search Engine with custom system prompt
        system_prompt = (
            "Answer ONLY from context below. Format: [Answer]. Here is why: [Quote or reasoning]. "
            "If unknown, say so.\n\n{context_data}\n\nResponse: {response_type}"
        )
        
        self.engine = LocalSearch(
            model=self.llm,
            context_builder=self.context_builder,
            tokenizer=self.tokenizer,
            system_prompt=system_prompt,
            model_params={
                "max_tokens": 2000, 
                "temperature": 0.0,
            },
            context_builder_params={
                "text_unit_prop": 0.6,       
                "community_prop": 0.1,      
                "conversation_history_max_turns": 5,
                "conversation_history_user_turns_only": True,
                "top_k_mapped_entities": 15,  
                "top_k_relationships": 15,    
                "include_entity_rank": True,
                "max_tokens": 8000,           
            },
            response_type="concise answer",   
        )

    def ask(self, question: str) -> str:
        # Run async search
        result = asyncio.run(self.engine.search(question))
        return result.response
