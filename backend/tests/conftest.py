"""Keep unit tests deterministic and independent of external OpenAI services."""

import os

os.environ["EMBEDDING_PROVIDER"] = "hashing"
os.environ["EMBEDDING_DIMENSION"] = "384"
os.environ["QDRANT_COLLECTION"] = "document_chunks_test"
os.environ["LLM_PROVIDER"] = "extractive"
os.environ["RERANKER_PROVIDER"] = "lexical"
os.environ["QUESTION_CACHE_ENABLED"] = "false"
os.environ["QUERY_CONCURRENCY_DISTRIBUTED_ENABLED"] = "false"
