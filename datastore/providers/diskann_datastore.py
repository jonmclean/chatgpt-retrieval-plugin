from typing import Dict, List, Optional, Any

import sqlite3
import logging

from datastore.datastore import DataStore
from models.models import (
    DocumentChunk,
    DocumentMetadataFilter,
    QueryResult,
    QueryWithEmbedding,
    DocumentChunkWithScore, DocumentChunkMetadata,
)

import diskannpy as dap
import numpy as np
from services.date import to_unix_timestamp, to_date_string

class DiskANNDataStore(DataStore):
    def __init__(
        self,
        vector_size: int = 1536,
    ):
        """
        Args:
            vector_size: Size of the embedding stored in a collection
        """
        logging.debug("Initializing Sqlite")
        self._conn = sqlite3.connect("diskann_sqlite.db")
        self._conn.row_factory = sqlite3.Row
        self._document_table_name = "documents"
        self._conn.cursor().execute(f"CREATE TABLE If NOT EXISTS {self._document_table_name}"
                                    f"(id INTEGER PRIMARY KEY AUTOINCREMENT, "
                                    f"externalId TEXT NOT NULL, "
                                    f"documentText TEXT NOT NULL, "
                                    f"source TEXT NULL, "
                                    f"source_id TEXT NULL, "
                                    f"url TEXT NULL, "
                                    f"created_at INTEGER NULL, "
                                    f"author TEXT NULL"
                                    f")")
        logging.debug("Finished initializing Sqlite")
        self._diskann_path = "diskann_index"
        logging.debug("Initializing DiskANN index")
        self._diskann_index = dap.DynamicMemoryIndex(
            metric="cosine",
            vector_dtype=np.float32,
            # OpenAI's embeddings have a length of 1,536
            dim=1536,
            max_points=20_000,
            complexity=64,
            graph_degree=32,
            num_threads=16,
        )
        logging.debug("Finished initializing DiskANN index")

    async def _query(
        self,
        queries: List[QueryWithEmbedding],
    ) -> List[QueryResult]:
        """
        Takes in a list of queries with embeddings and filters and returns a list of query results with matching document chunks and scores.
        """
        results = []
        for query in queries:
            diskann_neighbors, diskann_distances = self._diskann_index.search(
                np.array(query.embedding).astype(np.float32),
                k_neighbors=query.top_k,
                complexity=query.top_k * 2,
            )

            internal_id_to_distance_dict = dict(zip(diskann_neighbors, diskann_distances))

            parameter_marks = ','.join('?'*len(diskann_neighbors))
            cursor = self._conn.cursor().execute(f"SELECT "
                                                 f"id, "  # 0
                                                 f"externalId, "  # 1
                                                 f"documentText, "  # 2
                                                 f"source, "  # 3
                                                 f"source_id, "  # 4
                                                 f"url, "  #5
                                                 f"created_at, "  # 6
                                                 f"author "  # 7
                                                 f"from {self._document_table_name} WHERE id IN ({parameter_marks})",
                                                 diskann_neighbors.tolist())
            # Fetch all the data
            data = cursor.fetchall()
            query_results = []

            for item in data:
                internal_id = item['id']
                external_id = item['externalId']
                query_results.append(
                    DocumentChunkWithScore(
                        # "distance" goes up 0 to 1 but "score" goes from 1 to 0.
                        score=(1 - internal_id_to_distance_dict[internal_id]),
                        id=external_id,
                        text=item['documentText'],
                        metadata=DocumentChunkMetadata(
                            document_id=external_id,
                            source=item['source'],
                            source_id=item['source_id'],
                            url=item['url'],
                            created_at=to_date_string(item['created_at']),
                            author=item['author'],
                        ),
                        embedding=[0.99, 0.88, 0.77],
                    ))

            results.append(QueryResult(query=query.query, results=query_results,))

        return results

    async def _upsert(self, chunks: Dict[str, List[DocumentChunk]]) -> List[str]:
        """
        Takes in a list of list of document chunks and inserts them into the database.
        Return a list of document ids.
        """
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
                               f"documentText, "
                               f"source, "
                               f"source_id, "
                               f"url, "
                               f"created_at, "
                               f"author"
                               f") VALUES (?, ?, ?, ?, ?, ?, ?)", (
                                doc_chunk.id,
                                doc_chunk.text,
                                doc_chunk.metadata.source,
                                doc_chunk.metadata.source_id,
                                doc_chunk.metadata.url,
                                created_at,
                                doc_chunk.metadata.author
                                )
                               )
                internal_ids.append(cursor.lastrowid)
                doc_ids.append(doc_chunk.id)


        logging.debug(f"Writing {internal_ids=} to diskann")
        self._diskann_index.batch_insert(vectors=np.array(index_vectors).astype(np.float32), vector_ids=np.array(internal_ids).astype(np.uintc))
        self._diskann_index.save(self._diskann_path)
        logging.debug(f"Finished DiskANN write diskann")

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

        delete_succeeded = None

        if delete_all:
            logging.debug(f"Deleting all vectors")
            self._conn.cursor().execute(f"DELETE FROM {self._document_table_name}")
            delete_succeeded = True

        if ids and len(ids) > 0:
            parameter_marks = ','.join('?'*len(ids))

            self._conn.cursor().execute(f"DELETE FROM {self._document_table_name} WHERE externalId IN {parameter_marks}", ids)
            delete_succeeded = True

        if filter:
            sql_filter = self._convert_metadata_filter_to_sqlite_filter(metadata_filter=filter)
            logging.debug(f"Deleting vectors with filter {sql_filter}")
            self._conn.cursor().execute(f"DELETE FROM {self._document_table_name} WHERE {sql_filter[0]}", sql_filter[1])
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
