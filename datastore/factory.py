from datastore.datastore import DataStore
import os


async def get_datastore() -> DataStore:
    datastore = os.environ.get("DATASTORE")
    assert datastore is not None

    match datastore:
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
        case "diskann":
            from datastore.providers.diskann_datastore import DiskANNDataStore, \
                DynamicMemoryIndexDiskANNProvider, StaticMemoryIndexDiskANNProvider

            DISKANN_DATASTORE_TYPE = os.environ.get("DISKANN_DATASTORE_TYPE", "DynamicMemoryIndex").casefold()

            provider = None
            match DISKANN_DATASTORE_TYPE:
                case "dynamicmemoryindex":
                    provider = DynamicMemoryIndexDiskANNProvider()
                case "staticmemoryindex":
                    provider = StaticMemoryIndexDiskANNProvider()
                case _:
                    raise ValueError(f"Unsupported DiskANN index type: {DISKANN_DATASTORE_TYPE}")

            return DiskANNDataStore(provider)
        case _:
            raise ValueError(f"Unsupported vector database: {datastore}")
