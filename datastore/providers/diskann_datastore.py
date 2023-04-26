from typing import Dict, List, Optional

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
        self._document_table_name = "diskann.documents"
        self._conn.cursor().execute(f"CREATE TABLE If NOT EXISTS {self._document_table_name}(id INTEGER PRIMARY KEY AUTOINCREMENT, documentText TEXT)")

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
            cursor = self._conn.cursor().execute(f"select exists(SELECT id, documentText from {self._document_table_name} where id IN ?)", (document_ids, ))
            # Fetch all the data
            data = cursor.fetchall()
            query_results = []

            for item in data:
                query_results.append(
                    DocumentChunkWithScore(
                        score = 0.55555,
                        id = str(item[0]),
                        text = item[1],
                        metadata = DocumentChunkMetadata(
                            document_id = None,
                            source = "mySource",
                            source_id = "mySourceId",
                            url = "myUrl",
                            created_at = "myCreatedAt",
                            author = "myAuthor",
                        ),
                        embedding= [0.99, 0.88, 0.77],
                    ))

            results.append(QueryResult(query=query.query, results=results,))

        return results


    async def _upsert(self, chunks: Dict[str, List[DocumentChunk]]) -> List[str]:
        """
        Takes in a list of list of document chunks and inserts them into the database.
        Return a list of document ids.
        """
        doc_ids = []

        cursor = self._conn.cursor()

        for doc_id, doc_chunks in chunks.items():
            logger.debug(f"Upserting {doc_id} with {len(doc_chunks)} chunks")
            for doc_chunk in doc_chunks:
                cursor.execute(f"INSERT INTO {self._document_table_name} (documentText) VALUES (?)", (doc_chunk.text))
        return doc_ids
