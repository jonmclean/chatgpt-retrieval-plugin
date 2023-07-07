import sqlite3
import json
import numpy as np
import os

from tqdm.auto import tqdm

conn = sqlite3.connect("diskann_sqlite.db")
conn.row_factory = sqlite3.Row

document_table_name = "documents"
batch_size = 500
embedding_size = 1536

res = conn.cursor().execute(f"SELECT id, embedding, author, source, url FROM {document_table_name} ORDER BY id ASC")
rows = res.fetchall()

vectors_array = []
filters_array = []

next_internal_id = 0

for i in tqdm(range(0, len(rows), batch_size)):
    i_end = min(len(rows), i + batch_size)
    for item in rows[i:i_end]:
        internal_id = item['id']
        for q in range(next_internal_id, internal_id):
            vectors_array.append(np.zeros(embedding_size))
            print(f"missing {internal_id=} {next_internal_id=} {q=}")
        next_internal_id = internal_id + 1
        embedding = json.loads(item['embedding'])
        vectors_array.append(embedding)
        filters_array.append(','.join([item['author'], item['source'], item['url']]))

vectors = np.array(vectors_array, dtype=np.single)

with open('diskann.bin', 'wb') as file:
    _ = file.write(np.array(vectors.shape, dtype=np.intc).tobytes())
    _ = file.write(vectors.tobytes())

with open('diskann_filters.txt', 'w') as file:
    for filter in filters_array:
        _ = file.write(filter)
        _ = file.write(os.linesep)

conn.close()

from diskannpy._common import _write_index_metadata, _valid_metric
_write_index_metadata(
    index_path_and_prefix="diskann_index",
    dtype=np.single,
    metric=_valid_metric("l2"),
    num_points=next_internal_id,
    dimensions=embedding_size
)

# Use this command to generate the memory index using the bin file generated above
# ~/source/diskann/DiskANN/build/tests/build_memory_index --data_type float --dist_fn l2 --data_path ./diskann.bin --index_path_prefix diskann_index --label_file diskann_filters.txt
