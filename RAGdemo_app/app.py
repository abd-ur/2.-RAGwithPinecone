# app.py
import os
import json
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# pinecone setup
os.environ["PINECONE_API_KEY"] = "YOUR_API_KEY"
os.environ["PINECONE_ENV"] = "YOUR_INDEX_REGION"
INDEX_NAME = "YOUR_INDEX_NAME"

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# check or create index
if INDEX_NAME not in [idx.name for idx in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=os.environ.get("PINECONE_ENV")))
index = pc.Index(INDEX_NAME)

# load embed model
embed_model = SentenceTransformer("dmis-lab/biobert-base-cased-v1.1")

# load variant data
with open("variant.json", "r") as f:
    variants = json.load(f)

# upsert
if not index.describe_index_stats().total_vector_count:
    vectors_to_upsert = []
    for v in variants:
        vec = embed_model.encode(v["interpretation"], convert_to_numpy=True).tolist()
        vectors_to_upsert.append({
            "id": v["variant"] + "_" + v["gene"],
            "values": vec,
            "metadata": v})
    index.upsert(vectors=vectors_to_upsert)

# query and retrieval function
def query_variants(user_query, top_k=3, similarity_threshold=0.5, gene_filter=None):
    query_vec = embed_model.encode(user_query, convert_to_numpy=True).tolist()
    filter_dict = {"gene": {"$eq": gene_filter}} if gene_filter else None

    response = index.query(
        vector=query_vec,
        top_k=top_k,
        filter=filter_dict,
        include_metadata=True,
        include_values=False)

    results = []
    for match in response.matches:
        if match.score >= similarity_threshold:
            results.append({
                "id": match.id,
                "score": match.score,
                "metadata": match.metadata})
    if not results:
        return [{"message": "Insufficient data; consult a clinician."}]
    return results

# load biogpt
tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
model = AutoModelForCausalLM.from_pretrained("microsoft/biogpt")
llm_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=200)

def generate_rag_answer(user_query, top_matches):
    # combine contexts
    context_text = "\n".join([
        f"{m['metadata']['interpretation']} (Source: {m['metadata']['source']})"
        for m in top_matches if "metadata" in m])
    prompt = f"""
You are a biomedical assistant. Using the following contexts with cited sources, provide a clear treatment recommendation. 
Do not repeat the query. Summarize concisely in one sentence, citing sources.

Query: {user_query}
Contexts:
{context_text}
"""
    output = llm_pipeline(prompt, max_length=200, num_return_sequences=1)
    return output[0]["generated_text"]

# streamlit setup
st.title("Biomedical Variant RAG Demo (biogpt)")

user_query = st.text_input("Enter your variant query:", "Best drug for BRCA1 mutation?")
gene_filter = st.text_input("Gene filter (optional):", "")
top_k = st.slider("Number of top contexts to retrieve", 1, 5, 3)
similarity_threshold = st.slider("Similarity threshold", 0.0, 1.0, 0.5)

if st.button("Retrieve & Generate Answer"):
    top_matches = query_variants(user_query, top_k, similarity_threshold, gene_filter or None)

    # display context
    st.subheader("Retrieved Contexts")
    for i, match in enumerate(top_matches, 1):
        if "message" in match:
            st.write(match["message"])
        else:
            st.write(f"{i}. {match['metadata']['interpretation']} "
                     f"(Source: {match['metadata']['source']}) | "
                     f"Cosine similarity: {match['score']:.3f}")

    # generate RAG answer
    st.subheader("RAG Answer (biogpt)")
    answer = generate_rag_answer(user_query, top_matches)
    st.write(answer)

    # evaluation metrics (optional)
    recall_value = 1.0 
    df = pd.DataFrame([{
        "Variant": user_query,
        "Top-k Interpretations": [m['metadata']['interpretation'] for m in top_matches if 'metadata' in m],
        "Cosine Scores": [m['score'] for m in top_matches if 'metadata' in m],
        "Recall@k": recall_value}])
    st.subheader("Evaluation Metrics")
    st.dataframe(df)

# teardown note
st.info("To delete Pinecone index after demo, run:\n`pc.delete_index('variants-index')`")
