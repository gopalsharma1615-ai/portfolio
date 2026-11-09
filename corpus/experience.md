# Gopal Sharma: work experience

Gopal Sharma is a Senior Data Scientist at Newgen Software Technologies in New Delhi, India, working on Generative AI,
retrieval-augmented generation (RAG), GraphRAG and LLM systems. He has 4+ years of applied machine learning, the
last three shipping production Generative AI.

## Senior Data Scientist, Newgen Software (Oct 2025 - present)

### Multi-hop GraphRAG
Gopal designed a bidirectional meet-in-the-middle graph search that answers multi-hop questions over a
52,474-node, 60,475-edge LightRAG knowledge graph without knowing the target entity. It raised multi-hop QA
accuracy from 47% to 53% on a 70-question clean test set and lifted correct-path selection from 39% to 72-84%.
He built a fully local answer-candidate generator using breadth-first search over the reachable set, embedding
filtering and cross-encoder reranking. It cut candidate paths by 72%, from about 1,500 to about 420, with zero
API calls.

### Query routing
Gopal showed that query decomposition is a rescue, not a default: always decomposing lowered accuracy from 70.2%
to 62.8% on 952 HotpotQA questions. A single-first router that decomposes only when single-pass retrieval
abstains reached 76.9%, invoking the expensive decomposition path on only 18% of queries.

### Regulatory knowledge graph
Gopal built a resumable, layered knowledge graph over 371 financial-regulation documents (17,326 pages): 46,508
entities and 109,572 relations for $35.1 in LLM cost, about $2.03 per 1,000 pages, with document recall@20 of
0.93. A fully on-premises rebuild with Qwen3 on vLLM and bge-m3 embeddings matched the cloud-built graph
(document hit@20 0.91 vs 0.86). He migrated the graph backend from NetworkX to Neo4j through a ~150-line
adapter, reproducing 25/25 query results exactly and loading the full graph in 7 seconds.

### Document AI
Gopal built a CPU-only, LLM-free layout-aware PDF chunker using a DocLayNet YOLO layout detector,
typography-based heading levels and OCR fallback. It reached page hit@5 of 85.5% on OHR-Bench, against 95.5%
with ground-truth text. His table structure recognition, using Table Transformer with token-driven
post-processing, reached row boundary-F1 0.72 and column IoU-F1 0.94 on financial statements.

### Inference platforms
Gopal architected an asynchronous email-intelligence platform with FastAPI, Celery and Redis that runs 7 NLP
models (toxicity, PII, complaint type, sentiment) at about 8,400 emails per hour per node. A multiprocessing
engine keeps 28 transformer models preloaded in about 5 GB of RAM, bypassing the Python GIL. A k6 load test at
16 concurrent users showed 1.29 s median latency at 58% CPU. He split a monolithic document pipeline into 6
asynchronous microservices (acquisition, OCR, chunking, cleaning, embedding, storage) over 4 enterprise sources
and 6 file formats, with hybrid Milvus vector search for air-gapped, fully on-premises clients. He also shipped
tender technical-evaluation and similar-tender search APIs with page-level evidence. He mentors junior data
scientists.

## Data Scientist, Newgen Software (Oct 2023 - Oct 2025)
Gopal implemented LLM safety guardrails with 25+ parallel input, retrieval and output scanners completing in 1-2
seconds. They cover prompt injection, jailbreaks, 27 secret types, hidden Unicode, PII including Aadhaar and PAN,
toxicity, bias, factual consistency and hallucination across 22 languages. He ran Llama-3 teacher-to-student
knowledge distillation for text-to-SQL on 1,500 queries (4-bit student 54.6% vs teacher 83.3%) and assessed 8+
compression methods: QLoRA, AWQ, GPTQ, PEQA, LLM-QAT, pruning and low-rank factorization. He standardized LLM
evaluation on 7 metrics (SQuAD EM/F1, SAS, LERC, QAFactEval, QuestEval, BBH, MMLU) and generated synthetic
training data for 16 NLP tasks from 7 source formats.

## Data Scientist / AI-ML Solution Analyst, Number Theory (Jul 2022 - Aug 2023)
Number Theory was acquired by Newgen Software. There, Gopal built a 143-function feature-engineering library for
a loan-risk scorecard covering 5 credit products and 100,000 applicants from bank statements, tax returns and
bureau data. He trained PySpark gradient-boosted tree and random forest models, tuning thresholds for the highest
default recall at 0.70 precision.
