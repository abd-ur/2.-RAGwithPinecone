import os

os.environ["PINECONE_API_KEY"] = "YOUR_API_KEY"
os.environ["PINECONE_ENV"] = "us-east-1" 

from pinecone import Pinecone, ServerlessSpec

# init pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# index name
INDEX_NAME = "variants-index"

# check or create index
if INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=os.environ.get("PINECONE_ENV"))
    )

# connect ot index
index = pc.Index(INDEX_NAME)
print(f"Connected to Pinecone index: {INDEX_NAME}")
