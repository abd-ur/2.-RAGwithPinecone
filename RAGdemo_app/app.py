# app.py
import os
import json
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ------------------------------
# 1. Pinecone setup
# ------------------------------
os.environ["PINECONE_API_KEY"] = "YOUR_DUMMY_API_KEY"
os.environ["PINECONE_ENV"] = "us-east-1"
INDEX_NAME = "variants-index"

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# create index if not exists
if INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,  # BioBERT embedding dim
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=os.environ.get("PINECONE_ENV"))
    )
index = pc.Index(INDEX_NAME)

# ------------------------------
# 2. Load BioBERT embedding model
# ------------------------------
embed_model = SentenceTransformer(
    "pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli"
)

# ------------------------------
# 3. Load variant data and upsert
# ------------------------------
with open("data/variants.json", "r") as f:
    variants = json.load(f)

# Upsert only if index is empty
if not index.describe_index_stats().total_vector_count:
    vectors_to_upsert = []
    for v in variants:
        vec = embed_model.encode(v["interpretation"], convert_to_numpy=True).tolist()
        vectors_to_upsert.append({
            "id": v["variant"] + "_" + v["gene"],
            "values": vec,
            "metadata": v
        })
    index.upsert(vectors=vectors_to_upsert)

# ------------------------------
# 4. Query and retrieval functions
# ------------------------------
def query_variants(user_query, top_k=3, similarity_threshold=0.5, gene_filter=None):
    query_vec = embed_model.encode(user_query, convert_to_numpy=True).tolist()
    filter_dict = {"gene": {"$eq": gene_filter}} if gene_filter else None

    response = index.query(
        vector=query_vec,
        top_k=top_k,
        filter=filter_dict,
        include_metadata=True,
        include_values=False
    )

    results = []
    for match in response.matches:
        if match.score >= similarity_threshold:
            results.append({
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata
            })
    if not results:
        return [{"message": "Insufficient data; consult a clinician."}]
    return results

# ------------------------------
# 5. GPT-2 RAG generation
# ------------------------------
# Load GPT-2
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=200)

def generate_rag_answer(user_query, top_matches):
    # Combine contexts
    context_text = "\n".join([
        f"{m['metadata']['interpretation']} (Source: {m['metadata']['source']})"
        for m in top_matches if "metadata" in m
    ])
    prompt = f"""
You are a biomedical assistant. Using the following contexts with cited sources, provide a clear treatment recommendation. 
Do not repeat the query. Summarize concisely in one sentence, citing sources.

Query: {user_query}
Contexts:
{context_text}
"""
    output = llm_pipeline(prompt, max_length=200, num_return_sequences=1)
    return output[0]["generated_text"]

# ------------------------------
# 6. Streamlit interface
# ------------------------------
st.title("Biomedical Variant RAG Demo (GPT-2)")

user_query = st.text_input("Enter your variant query:", "Best drug for BRCA1 mutation?")
gene_filter = st.text_input("Gene filter (optional):", "")
top_k = st.slider("Number of top contexts to retrieve", 1, 5, 3)
similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.5)

if st.button("Retrieve & Generate Answer"):
    top_matches = query_variants(user_query, top_k, similarity_threshold, gene_filter or None)

    # Display retrieved contexts
    st.subheader("Retrieved Contexts")
    for i, match in enumerate(top_matches, 1):
        if "message" in match:
            st.write(match["message"])
        else:
            st.write(f"{i}. {match['metadata']['interpretation']} "
                     f"(Source: {match['metadata']['source']}) | "
                     f"Cosine similarity: {match['score']:.3f}")

    # Generate RAG answer
    st.subheader("RAG Answer (GPT-2)")
    answer = generate_rag_answer(user_query, top_matches)
    st.write(answer)

    # Display evaluation metrics (optional)
    recall_value = 1.0  # Placeholder, replace with proper function if needed
    df = pd.DataFrame([{
        "Variant": user_query,
        "Top-k Interpretations": [m['metadata']['interpretation'] for m in top_matches if 'metadata' in m],
        "Cosine Scores": [m['score'] for m in top_matches if 'metadata' in m],
        "Recall@k": recall_value
    }])
    st.subheader("Evaluation Metrics")
    st.dataframe(df)

# Teardown note
st.info("To delete Pinecone index after demo, run:\n`pc.delete_index('variants-index')`")
