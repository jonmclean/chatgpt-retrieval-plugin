import sqlite3
from datasets import load_dataset
import json
import numpy as np
from datetime import datetime
from tqdm.auto import tqdm

from models.models import Document, DocumentMetadata
from services.chunks import get_document_chunks
from services.date import to_unix_timestamp

data = load_dataset("squad", split="train")
data = data.to_pandas()
data = data.drop_duplicates(subset=["context"])

print(len(data))

conn = sqlite3.connect("diskann_sqlite.db")
conn.row_factory = sqlite3.Row
document_table_name = "documents"
batch_size = 500
res = conn.cursor().execute(f"CREATE TABLE If NOT EXISTS {document_table_name}"
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
documents = [
    Document(
        id=r['id'],
        text=r['context'],
        metadata=DocumentMetadata(
            source='file',
            source_id=f"source_id_{r['id']}",
            url=f"file://example/{r['id']}",
            author=f"author_{r['id']}",
            created_at=str(datetime.now()),
            title=r['title']
        )
    ) for r in data.to_dict(orient='records')
]

print("Splitting into chunks, generating embeddings, and writing to database")
for i in tqdm(range(0, len(documents), batch_size)):
    i_end = min(len(documents), i + batch_size)
    document_chunks = get_document_chunks(documents[i:i_end], chunk_token_size = None)

    cursor = conn.cursor()
    index_vectors = []
    internal_ids = []

    for id, doc_chunk_list in document_chunks.items():
        for doc_chunk in doc_chunk_list:
            created_at = (
                to_unix_timestamp(doc_chunk.metadata.created_at) if doc_chunk.metadata.created_at is not None else None
            )

            index_vectors.append(np.array(doc_chunk.embedding))
            res = cursor.execute(f"INSERT INTO {document_table_name} "
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
        conn.commit()
conn.close()
