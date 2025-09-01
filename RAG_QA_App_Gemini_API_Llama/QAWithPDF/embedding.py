from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.gemini import GeminiEmbedding

import sys
from exception import CustomException
from logger import logging


def download_gemini_embedding(model, document):
    """
    Initializes a Gemini Embedding model and builds a query engine for PDF QA.

    Args:
        model: The LLM (e.g., Gemini) used for query answering.
        document: A list of documents loaded (usually from PDF).

    Returns:
        query_engine: A query engine built from the vector store index.
    """
    try:
        logging.info("Initializing Gemini embedding model...")

        # Create Gemini embedding model
        gemini_embed_model = GeminiEmbedding(model_name="models/embedding-001")

        logging.info("Configuring global LlamaIndex settings...")

        # Configure global settings (LLM + embedding model + chunking)
        Settings.llm = model
        Settings.embed_model = gemini_embed_model
        Settings.chunk_size = 800
        Settings.chunk_overlap = 20

        logging.info("Building vector index from documents...")

        # Build vector index from docs (uses global Settings)
        index = VectorStoreIndex.from_documents(document)

        # # Save FAISS index to disk
        # index.storage_context.persist("storage")

        logging.info("Creating query engine from index...")

        # Build query engine (answers questions)
        query_engine = index.as_query_engine()

        logging.info("✅ Query engine successfully created and ready to use.")
        return query_engine

    except Exception as e:
        logging.error("❌ Error occurred while initializing Gemini embedding or creating index.")
        raise CustomException(e, sys)
