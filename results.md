## Result for query and retrieval
| Query                          | Output  |
| ------------------------------ | ------- |
| Best drug for BRCA1 mutation?  | Likely pathogenic; PARP inhibitors recommended. (Source: ClinVar) (Cosine similarity: 0.889) <br/> Truncating variant; Radiation not advised. (Source: COSMIC) (Cosine similarity: 0.862) <br/> Truncating variant; Radiation not advised. (Source: ClinVar) (Cosine similarity: 0.862) |

## Result for LLM based prompt using biogpt
Input query for prompt generation: Best treatment for TP53 R130* mutation? <br/>
Output: Context - 1. Founder mutation; Genetic counseling required. (Sources: ClinVar, COSMIC)<br/>
        Answer - Genetic counseling required to support clinical decisions and counseling resources (Sources: ClinVar, COSMIC).

Discussion: Pipeline retrieves accurate and reproducible answer/recommendation along with citations confirming no hallucination on test runs as demonstrated below. However, performance can be made more accurate with fine-tuning biogpt using variant dataset.
