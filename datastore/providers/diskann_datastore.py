import abc
import asyncio
from abc import ABC
from typing import Dict, List, Optional, Any

import sqlite3
import logging

from diskannpy import VectorLikeBatch, VectorIdentifierBatch, QueryResponse

from datastore.datastore import DataStore
from models.models import (
    DocumentChunk,
    DocumentMetadataFilter,
    QueryResult,
    QueryWithEmbedding,
    DocumentChunkWithScore, DocumentChunkMetadata,
)

import diskannpy as dap
import json
import numpy as np
from services.date import to_unix_timestamp, to_date_string


class DiskANNProvider(ABC):
    @abc.abstractmethod
    def search(self, embedding: np.array, k_neighbors: int, complexity: int):
        raise NotImplementedError

    def write(self, vectors: VectorLikeBatch, vector_ids: VectorIdentifierBatch):
        raise NotImplementedError

    def delete(self, vector_ids: VectorIdentifierBatch):
        raise NotImplementedError

    @abc.abstractmethod
    def is_read_only(self) -> bool:
        raise NotImplementedError


class DynamicMemoryIndexDiskANNProvider(DiskANNProvider):
    def is_read_only(self) -> bool:
        return False

    def __init__(
            self,
            vector_size: int = 1536,
            write_to_disk: bool = True,
    ):
        self._diskann_path = "diskann_index"
        logging.debug("Initializing DiskANN index")
        self._diskann_index = dap.DynamicMemoryIndex(
            distance_metric="cosine",
            vector_dtype=np.single,
            dimensions=vector_size,
            max_vectors=20_000,
            complexity=64,
            graph_degree=32,
            num_threads=16,
        )
        self._diskann_save_needed = True
        self._write_to_disk = write_to_disk
        if self._write_to_disk:
            self._save_task = asyncio.get_event_loop().create_task(self.save_async_loop())
        logging.debug("Finished initializing DiskANN index")

    def __del__(self):
        # Be sure we cancel the save task when shutting down.
        if self._diskann_save_needed and self._write_to_disk:
            self._save_task.cancel()

            self._do_save()

    async def save_async_loop(self):
        """
        Loops on the event thread and saven the index periodically.  This is to keep the web request from blocking
        when an upsert or delete comes in from the REST API.  Data could be lost from the index if the event thread
        is shutdown between the upsert/delete and the next save.

        :return: This method should run until Python shuts down.  Nothing to return.
        """
        while True:
            await self._do_save()

            # Save again in five minutes.  Since we are in an event loop this will allow
            # other REST requests to be serviced while we wait for the next save.
            await asyncio.sleep(60 * 5)

    async def _do_save(self):
        """
        Save the index if there are changes that need to be persisted

        :return: None
        """
        # Only save if there is something to save.
        if self._diskann_save_needed:
            logging.debug("Starting DiskANN save")
            self._diskann_index.save(self._diskann_path)
            self._diskann_save_needed = False
            logging.debug("Finished DiskANN save")

        return None

    def write(self, vectors: VectorLikeBatch, vector_ids: VectorIdentifierBatch):
        logging.debug(f"Writing {vector_ids=} to diskann")
        self._diskann_index.batch_insert(vectors=np.array(vectors).astype(np.single),
                                         vector_ids=np.array(vector_ids).astype(np.uintc))
        self._diskann_save_needed = True
        logging.debug(f"Finished DiskANN write diskann")

    def search(self, embedding: np.array, k_neighbors: int, complexity: int) -> QueryResponse:
        return self._diskann_index.search(
            embedding,
            k_neighbors=k_neighbors,
            complexity=complexity,
        )

    def delete(self, vector_ids: VectorIdentifierBatch):
        for vector_id in vector_ids:
            self._diskann_index.mark_deleted(vector_id)
        self._diskann_save_needed = True


class StaticMemoryIndexDiskANNProvider(DiskANNProvider):
    def is_read_only(self) -> bool:
        return True

    def __init__(
            self,
            vector_size: int = 1536,
    ):
        logging.debug("Loading static DiskANN index")
        self._diskann_index = dap.StaticMemoryIndex(
            index_directory="./",
            num_threads=4,
            initial_search_complexity=60,
            index_prefix="diskann_index",
        )
        logging.debug("Finished loading DiskANN index")

    def search(self, embedding: np.array, k_neighbors: int, complexity: int) -> QueryResponse:
        return self._diskann_index.search(
            embedding,
            k_neighbors=k_neighbors,
            complexity=complexity,
        )


class DiskANNDataStore(DataStore):
    def __init__(
            self,
            diskann_provider: DiskANNProvider,
            sqlite_database: str = "diskann_sqlite.db"
    ):
        """
        Args:
            diskann_provider: Provider that handles all interactions with DiskANN
        """
        logging.debug("Initializing Sqlite")
        self._conn = sqlite3.connect(sqlite_database)
        self._conn.row_factory = sqlite3.Row
        self._document_table_name = "documents"
        self._conn.cursor().execute(f"CREATE TABLE If NOT EXISTS {self._document_table_name}"
                                    f"(id INTEGER PRIMARY KEY AUTOINCREMENT, "
                                    f"externalId TEXT NOT NULL, "
                                    f"documentChunkId TEXT NOT NULL, "
                                    f"documentText TEXT NOT NULL, "
                                    f"source TEXT NULL, "
                                    f"source_id TEXT NULL, "
                                    f"url TEXT NULL, "
                                    f"created_at INTEGER NULL, "
                                    f"author TEXT NULL, "
                                    f"embedding TEXT NOT NULL"
                                    f")")
        logging.debug("Finished initializing Sqlite")
        self._diskann_provider = diskann_provider

    async def _query(
            self,
            queries: List[QueryWithEmbedding],
    ) -> List[QueryResult]:
        """
        Takes in a list of queries with embeddings and filters and returns a list of query results with matching document chunks and scores.
        """
        results = []
        for query in queries:
            diskann_neighbors, diskann_distances = self._diskann_provider.search(
                np.array(query.embedding).astype(np.single),
                k_neighbors=query.top_k,
                complexity=query.top_k * 2,
            )

            internal_id_to_distance_dict = dict(zip(diskann_neighbors, diskann_distances))

            parameter_marks = ','.join('?' * len(diskann_neighbors))
            cursor = self._conn.cursor().execute(f"SELECT "
                                                 f"id, "
                                                 f"externalId, "
                                                 f"documentChunkId, "
                                                 f"documentText, "
                                                 f"source, "
                                                 f"source_id, "
                                                 f"url, "
                                                 f"created_at, "
                                                 f"author, "
                                                 f"embedding "
                                                 f"from {self._document_table_name} WHERE id IN ({parameter_marks})",
                                                 diskann_neighbors.tolist())
            # Fetch all the data
            data = cursor.fetchall()
            query_results = []

            for item in data:
                internal_id = item['id']
                query_results.append(
                    DocumentChunkWithScore(
                        # "distance" goes up 0 to 1 but "score" goes from 1 to 0.
                        score=(1 - internal_id_to_distance_dict[internal_id]),
                        id=item['documentChunkId'],
                        text=item['documentText'],
                        metadata=DocumentChunkMetadata(
                            document_id=item['externalId'],
                            source=item['source'],
                            source_id=item['source_id'],
                            url=item['url'],
                            created_at=to_date_string(item['created_at']),
                            author=item['author'],
                        ),
                        embedding=json.loads(item['embedding']),
                    ))

            results.append(QueryResult(query=query.query,
                                       results=sorted(query_results, key=lambda result: result.score, reverse=True)))

        return results

    async def _upsert(self, chunks: Dict[str, List[DocumentChunk]]) -> List[str]:
        """
        Takes in a list of list of document chunks and inserts them into the database.
        Return a list of document ids.
        """
        if self._diskann_provider.is_read_only():
            raise NotImplementedError(f"Cannot upsert into a read-only DiskANN index")

        doc_ids: list[str] = []

        cursor = self._conn.cursor()

        index_vectors = []
        internal_ids = []

        for external_doc_id, doc_chunks in chunks.items():
            logging.debug(f"Upserting {external_doc_id} with {len(doc_chunks)} chunks")
            for doc_chunk in doc_chunks:
                created_at = (
                    to_unix_timestamp(doc_chunk.metadata.created_at)
                    if doc_chunk.metadata.created_at is not None
                    else None
                )

                index_vectors.append(np.array(doc_chunk.embedding))
                cursor.execute(f"INSERT INTO {self._document_table_name} "
                               f"(externalId, "
                               f"documentChunkId, "
                               f"documentText, "
                               f"source, "
                               f"source_id, "
                               f"url, "
                               f"created_at, "
                               f"author, "
                               f"embedding "
                               f") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (
                                   doc_chunk.metadata.document_id,
                                   doc_chunk.id,
                                   doc_chunk.text,
                                   doc_chunk.metadata.source,
                                   doc_chunk.metadata.source_id,
                                   doc_chunk.metadata.url,
                                   created_at,
                                   doc_chunk.metadata.author,
                                   json.dumps(doc_chunk.embedding),
                               )
                               )
                internal_ids.append(cursor.lastrowid)
            doc_ids.append(external_doc_id)

        self._diskann_provider.write(vectors=index_vectors, vector_ids=internal_ids)

        logging.debug("Committing transaction to sqlite")
        self._conn.commit()
        logging.debug("Transaction committed")
        return doc_ids

    async def delete(
            self,
            ids: Optional[List[str]] = None,
            filter: Optional[DocumentMetadataFilter] = None,
            delete_all: Optional[bool] = None,
    ) -> bool:
        """
        Removes vectors by ids, filter, or everything in the datastore.
        Multiple parameters can be used at once.
        Returns whether the operation was successful.
        """

        if self._diskann_provider.is_read_only():
            raise NotImplementedError(f"Cannot delete from a read-only DiskANN index")

        if delete_all:
            logging.debug(f"Deleting all vectors")
            logging.debug(f"Retrieving vector IDs from sqllite and deleting from DiskANN")
            cursor = self._conn.cursor().execute(f"SELECT id from {self._document_table_name}")
            while True:
                rows = cursor.fetchmany(100)
                if not rows:
                    break
                vector_ids = np.asarray([row[0] for row in rows])
                self._diskann_provider.delete(vector_ids=vector_ids)
            logging.debug("Truncating table in sqllite")
            self._conn.cursor().execute(f"DELETE FROM {self._document_table_name}")
            self._conn.commit()
            return True
        else:
            delete_succeeded = None
            if ids and len(ids) > 0:
                logging.debug("Retrieving vector IDS from sqllite and deleting from DiskANN")
                parameter_marks = ','.join('?' * len(ids))

                cursor = self._conn.cursor().execute(f"SELECT id from {self._document_table_name} "
                                                     f"WHERE externalId IN ({parameter_marks})", ids)
                while True:
                    rows = cursor.fetchmany(100)
                    if not rows:
                        break
                    vector_ids = np.asarray([row[0] for row in rows])
                    parameter_marks = ','.join('?' * len(vector_ids))
                    self._diskann_provider.delete(vector_ids=vector_ids)
                    logging.debug("Removing from sqllite")
                    self._conn.cursor().execute(
                        f"DELETE FROM {self._document_table_name} WHERE id IN ({parameter_marks})", vector_ids)

                delete_succeeded = True

            if filter:
                sql_filter = self._convert_metadata_filter_to_sqlite_filter(metadata_filter=filter)
                logging.debug(f"Deleting vectors from with filter {sql_filter}")
                cursor = self._conn.cursor().execute(f"SELECT id from {self._document_table_name} "
                                                     f"WHERE {sql_filter[0]}", sql_filter[1])
                while True:
                    rows = cursor.fetchmany(100)
                    if not rows:
                        break
                    vector_ids = np.asarray([row[0] for row in rows])
                    parameter_marks = ','.join('?' * len(vector_ids))
                    logging.debug(f"Deleting vectors from with filter {sql_filter}")
                    self._conn.cursor().execute(f"DELETE FROM {self._document_table_name} "
                                                f"WHERE id IN ({parameter_marks})", vector_ids)
                    self._diskann_provider.delete(vector_ids=vector_ids)
                delete_succeeded = True

            if delete_succeeded:
                self._conn.commit()
                return True
            else:
                return False

    def _convert_metadata_filter_to_sqlite_filter(
            self,
            metadata_filter: Optional[DocumentMetadataFilter] = None,
    ) -> Optional[tuple[str, Any]]:
        if metadata_filter is None:
            return None

        clauses = []
        parameters = []

        # Equality filters for the payload attributes
        if metadata_filter:
            if metadata_filter.document_id:
                clauses.append("externalId == ?")
                parameters.append(metadata_filter.document_id)
            if metadata_filter.source:
                raise NotImplementedError
            if metadata_filter.source_id:
                raise NotImplementedError
            if metadata_filter.author:
                raise NotImplementedError
            if metadata_filter.start_date:
                raise NotImplementedError
            if metadata_filter.end_date:
                raise NotImplementedError

        if 0 == len(clauses):
            return None

        return " OR ".join(clauses), parameters
