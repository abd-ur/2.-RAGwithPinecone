# 1.RAG with Pinecone (Assignment 2)
Here, we have developed RAG pipeline using Langchain and pinecone VD. Functions to retrieve most appropriate output for user query in biomedical/cancer context. Proposed method could be used for instant responses on cancer queries with a human in loop, making therapies much faster and accurate.

## Part 1
1. Loaded JSON variants and embedded using BioBERT model
2. Upsert to pinecone index('variants-index') with dim=768(instructed dim=384 showed poor performance)

## Part 2
1. Function developed for querying variants using BioBERT model and retrieve top 3 matches based on cosine sim > 0.7
2. Designed a RAG chain using gpt-2 as LLM for retrieving contexts and recommendations based on query
3. Handles edge cases
4. Added a feature for hybrid search using gene_filter (more efficient than cancer_type filter as instructed)

Run Instructions:
1. Open notebook in Google Colab @ https://colab.research.google.com/drive/1C29lLRvU7Nh8CPaa1vJ5YD5mMDpw06n1?usp=drive_link
2. Run first cell for installing dependencies -
```python
!pip install sentence-transformers pinecone langchain langchain-community sacremoses
```
4. Upload your variant.json(provided in this repo) file in the prompt itself
5. Run each notebook cell squentially to reproduce results
6. Results are discussed in results.md file

## Important Points to Consider:
1. Didnt use upserting dim = 384 as instructed, since it was in-efficient. Used dim = 784 on BioBERT model
2. NoteBook preview doesnt appears due to metadata error. Download or visit notebook @ https://colab.research.google.com/drive/1C29lLRvU7Nh8CPaa1vJ5YD5mMDpw06n1?usp=drive_link
3. Refer to set_pine.py for initialization settings of pinecone index

## Pinecone Teardown

After running the demo or experiments:
1. Delete the index to free up resources:
```python
pc.delete_index("variants-index")
```
2. Remove environment variables
```python
import os
os.environ.pop("PINECONE_API_KEY", None)
os.environ.pop("PINECONE_ENV", None)
```

## Ethics and Bias Mitigation
1. Fine-tuning model can improve results
2. Output includes sources of recommendation to reduce hallucinations
3. Pipeline is for research purpose only!
