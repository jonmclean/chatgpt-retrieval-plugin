# Queries
queries = data['question'].tolist()

# format into the structure needed by the /query endpoint
queries = [{'query': queries[i]} for i in range(len(queries))]

len(queries)

queries[:3]

res = requests.post(
    "http://0.0.0.0:8000/query",
    headers=headers,
    json={
        'queries': queries[:3]
    }
)

res
