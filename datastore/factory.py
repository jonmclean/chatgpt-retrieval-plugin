from datastore.datastore import DataStore
import os
from loguru import logger


async def get_datastore() -> DataStore:
    datastore = os.environ.get("DATASTORE")
    assert datastore is not None

    match datastore:
        case "chroma":
            from datastore.providers.chroma_datastore import ChromaDataStore

            return ChromaDataStore()
        case "llama":
            from datastore.providers.llama_datastore import LlamaDataStore

            return LlamaDataStore()

        case "pinecone":
            from datastore.providers.pinecone_datastore import PineconeDataStore

            return PineconeDataStore()
        case "weaviate":
            from datastore.providers.weaviate_datastore import WeaviateDataStore

            return WeaviateDataStore()
        case "milvus":
            from datastore.providers.milvus_datastore import MilvusDataStore

            return MilvusDataStore()
        case "zilliz":
            from datastore.providers.zilliz_datastore import ZillizDataStore

            return ZillizDataStore()
        case "redis":
            from datastore.providers.redis_datastore import RedisDataStore

            return await RedisDataStore.init()
        case "qdrant":
            from datastore.providers.qdrant_datastore import QdrantDataStore

            return QdrantDataStore()
        case "azuresearch":
            from datastore.providers.azuresearch_datastore import AzureSearchDataStore

            return AzureSearchDataStore()
        case "supabase":
            from datastore.providers.supabase_datastore import SupabaseDataStore

            return SupabaseDataStore()
        case "postgres":
            from datastore.providers.postgres_datastore import PostgresDataStore

            return PostgresDataStore()
        case "analyticdb":
            from datastore.providers.analyticdb_datastore import AnalyticDBDataStore

            return AnalyticDBDataStore()
        case "elasticsearch":
            from datastore.providers.elasticsearch_datastore import (
                ElasticsearchDataStore,
            )

            return ElasticsearchDataStore()
        case "diskann":
            from datastore.providers.diskann_datastore import DiskANNDataStore, \
                DynamicMemoryIndexDiskANNProvider, StaticMemoryIndexDiskANNProvider

            DISKANN_DATASTORE_TYPE = os.environ.get("DISKANN_DATASTORE_TYPE", "DynamicMemoryIndex").casefold()

            provider = None
            data_path = "diskann_data"
            match DISKANN_DATASTORE_TYPE:
                case "dynamicmemoryindex":
                    logger.debug("Loading dynamic DiskANN index provider")
                    provider = DynamicMemoryIndexDiskANNProvider(data_path=data_path)
                case "staticmemoryindex":
                    logger.debug("Loading static DiskANN index provider")
                    provider = StaticMemoryIndexDiskANNProvider()
                case _:
                    raise ValueError(f"Unsupported DiskANN index type: {DISKANN_DATASTORE_TYPE}")

            return DiskANNDataStore(provider, data_path=data_path)
        case _:
            raise ValueError(
                f"Unsupported vector database: {datastore}. "
                f"Try one of the following: llama, elasticsearch, pinecone, weaviate, milvus, zilliz, redis, or qdrant"
            )
