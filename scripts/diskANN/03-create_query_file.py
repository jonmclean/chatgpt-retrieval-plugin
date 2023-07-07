import sqlite3
import json
import numpy as np
import os

from tqdm.auto import tqdm

conn = sqlite3.connect("diskann_sqlite.db")
conn.row_factory = sqlite3.Row

document_table_name = "documents"

res = conn.cursor().execute(f"SELECT id, embedding, author, source, url FROM {document_table_name} ORDER BY id ASC LIMIT 100")
rows = res.fetchall()

vectors_array = []
filters_array = []

for item in rows:
    internal_id = item['id']
    embedding = json.loads(item['embedding'])
    vectors_array.append(embedding)
    filters_array.append(','.join([item['author'], item['source'], item['url']]))
    #print(internal_id)

vectors = np.array(vectors_array, dtype=np.single)

with open('diskann.query', 'wb') as file:
    _ = file.write(np.array(vectors.shape, dtype=np.intc).tobytes())
    _ = file.write(vectors.tobytes())



# Prints 1 to 100

# ~/source/diskann/DiskANN/build/apps/search_memory_index --data_type float --dist_fn l2 --index_path_prefix diskann_index --result_path ./my_results -K 10 --query_file diskann.query -L 100
