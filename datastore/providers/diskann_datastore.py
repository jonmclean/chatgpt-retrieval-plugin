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
from qdrant_client.http import models as rest

import qdrant_client

from services.date import to_unix_timestamp

class DiskANNDataStore(DataStore):
    def __init__(
        self,
        vector_size: int = 1536,
    ):
        """
        Args:
            vector_size: Size of the embedding stored in a collection
        """
        self._conn = sqlite3.connect("diskann_sqlite.db")
        self._document_table_name = "documents"
        self._conn.cursor().execute(f"CREATE TABLE If NOT EXISTS {self._document_table_name}(id INTEGER PRIMARY KEY AUTOINCREMENT, externalId TEXT, documentText TEXT)")

    async def _query(
        self,
        queries: List[QueryWithEmbedding],
    ) -> List[QueryResult]:
        """
        Takes in a list of queries with embeddings and filters and returns a list of query results with matching document chunks and scores.
        """
        results = []
        for query in queries:
            document_ids = (1, 2, 3,)
            parameter_marks = ','.join('?'*len(document_ids))
            cursor = self._conn.cursor().execute(f"SELECT id, externalId, documentText from {self._document_table_name} WHERE id IN ({parameter_marks})", document_ids)
            # Fetch all the data
            data = cursor.fetchall()
            query_results = []

            for item in data:
                internal_id = item[0]
                external_id = item[1]
                doc_text = item[2]
                query_results.append(
                    DocumentChunkWithScore(
                        score = 0.55555,
                        id = external_id,
                        text = doc_text,
                        metadata = DocumentChunkMetadata(
                            document_id = external_id,
                            source = "email",
                            source_id = "mySourceId",
                            url = "myUrl",
                            created_at = "myCreatedAt",
                            author = "myAuthor",
                        ),
                        embedding= [0.99, 0.88, 0.77],
                    ))

            results.append(QueryResult(query=query.query, results=query_results,))

        return results

    async def _upsert(self, chunks: Dict[str, List[DocumentChunk]]) -> List[str]:
        """
        Takes in a list of list of document chunks and inserts them into the database.
        Return a list of document ids.
        """
        doc_ids = []

        cursor = self._conn.cursor()

        for external_doc_id, doc_chunks in chunks.items():
            logging.debug(f"Upserting {external_doc_id} with {len(doc_chunks)} chunks")
            for doc_chunk in doc_chunks:
                cursor.execute(f"INSERT INTO {self._document_table_name} (externalId, documentText) VALUES (?, ?)", (doc_chunk.id, doc_chunk.text))

        self._conn.commit()
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
        if metadata_filter is None and ids is None:
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
