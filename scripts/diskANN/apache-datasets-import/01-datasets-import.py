# pip install -qU datasets pandas tqdm

from datasets import load_dataset
from datetime import datetime
from tqdm.auto import tqdm
import requests
from requests.adapters import HTTPAdapter, Retry

data = load_dataset("squad", split="train")

# data
data = data.to_pandas()

# data.head()
data = data.drop_duplicates(subset=["context"])

# print(len(data))

# data.head()

documents = [
    {
        'id': r['id'],
        'text': r['context'],
        'metadata': {
            'source': 'file',
            'source_id': f"source_id_{r['id']}",
            'url': f"file://example/{r['id']}",
            'author': f"author_{r['id']}",
            'created_at': str(datetime.now()),
            'title': r['title']
        }
    } for r in data.to_dict(orient='records')
]

# documents[:3]

import os

BEARER_TOKEN = os.environ.get("BEARER_TOKEN") or "mytoken"

headers = {
    "Authorization": f"Bearer {BEARER_TOKEN}"
}

batch_size = 100

endpoint_url = "http://localhost:3333"

s = requests.Session()

# we setup a retry strategy to retry on 5xx errors
retries = Retry(
    total=5,  # number of retries before raising error
    backoff_factor=0.1,
    status_forcelist=[500, 502, 503, 504]
)

s.mount('http://', HTTPAdapter(max_retries=retries))

for i in tqdm(range(0, len(documents), batch_size)):
    i_end = min(len(documents), i + batch_size)

    # make post request that allows up to 5 retries
    res = s.post(
        f"{endpoint_url}/upsert",
        headers=headers,
        json={
            "documents": documents[i:i_end]
        })

    if res.status_code != 200:
        print(res)
