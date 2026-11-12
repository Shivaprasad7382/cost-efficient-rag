AskNova: Cost-Efficient RAG Question Answering

A question-answering service that answers from your own documents (PDF, HTML, Markdown, TXT) and cites the chunks it used. It is built to keep running costs low: the vector store is ChromaDB running inside the app and saving to disk, so there is no always-on vector database to pay for. Includes an evaluation harness for retrieval quality, answer quality, and cost/latency.

Stack: Python, FastAPI, ChromaDB, sentence-transformers (all-MiniLM-L6-v2), Claude API (Anthropic), Pydantic

How it works
Client -> FastAPI (/ingest, /query)
              |
   +----------+-----------+
   |                      |
Ingestion             RAG pipeline
(load -> chunk ->     (embed query -> top-k search ->
 hash -> dedup)        similarity threshold / fallback ->
   |                   grounded prompt -> Claude -> cited answer)
   v                      |
ChromaDB  <----embed------+
(persistent, on disk)
Ingest: each file is loaded (PDF, HTML, Markdown or TXT) and split into chunks of up to 500 characters with 50 characters of overlap, preferring paragraph, line, sentence and word boundaries.
Idempotent ids: every chunk gets a SHA-256 id built from its source path, chunk index and text. Re-running ingestion skips ids already in the store, so no duplicate vectors are created.
Embed and store: chunks are embedded with all-MiniLM-L6-v2 (384 dimensions) and saved in a persistent ChromaDB collection that uses cosine similarity.
Retrieve: the question is embedded and the top-k chunks are fetched, with an optional metadata filter such as {"file_type": "pdf"}.
Fallback: chunks below the similarity threshold (default 0.25) are dropped. If none are left, the model is not called at all. The service returns a fixed "insufficient context" message at zero cost instead of guessing.
Answer: the remaining chunks go into a grounded prompt, and Claude answers using only that context, citing each source as [Doc: <source>, Chunk: <chunk_index>].
Log: every query writes latency, token counts and estimated cost to a JSONL log.
Quick start
bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # then add your ANTHROPIC_API_KEY
uvicorn src.api:app --reload

Put your files in data/raw_documents/ (four sample files are included), then call the API:

bash
# ingest everything under data/raw_documents/ (safe to run twice)
curl -X POST http://localhost:8000/ingest -H "Content-Type: application/json" -d '{}'

# ask a question
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" \
  -d '{"query": "What is position bias in LLM judges?", "top_k": 5}'
Endpoint	What it does
POST /ingest	Ingests paths (or every file under data/raw_documents/). Returns files processed, chunks seen, inserted, skipped as duplicates, and total vectors.
POST /query	Takes query, optional top_k and metadata_filter. Returns the answer, citations with similarity scores, latency, token counts, estimated cost and whether the fallback fired.
GET /stats	Total vectors and collection name.
GET /health	Liveness check.
Configuration

All settings come from environment variables (see .env.example); nothing is hardcoded.

Variable	Default	Purpose
ANTHROPIC_API_KEY	none	Claude API key for answers and the answer judge
GENERATION_MODEL	claude-sonnet-4-6	Model used to write answers
EMBEDDING_BACKEND	sentence_transformers	sentence_transformers or hashing (offline fallback)
EMBEDDING_MODEL_NAME	all-MiniLM-L6-v2	Embedding model (384 dimensions)
DEFAULT_CHUNK_SIZE / DEFAULT_CHUNK_OVERLAP	500 / 50	Chunking in characters
DEFAULT_TOP_K	5	Chunks retrieved per question
SIMILARITY_THRESHOLD	0.25	Minimum cosine similarity for a chunk to be used
VECTOR_STORE_PATH	./data/chroma_store	Where ChromaDB saves vectors

Embedding backends. sentence_transformers gives real semantic embeddings and needs internet once to download the model. hashing is a deterministic, fully offline bag-of-hashed-words embedder. It is not semantically meaningful, but lets the whole pipeline (ingestion, dedup, storage, metrics, cost math) run in CI or network-restricted machines. If the model cannot be loaded, the app falls back to hashing automatically instead of crashing.

Evaluation
bash
python -m eval.evaluate_retrieval   # Recall@k, Hit Rate, MRR, nDCG@k, Context Precision
python -m eval.evaluate_answer      # Faithfulness + Relevance (LLM-as-judge, 1-5) + Exact Match / F1
python -m eval.cost_analysis        # cost table + p50/p95 latency from logs/query_log.jsonl

data/eval_dataset.json holds 15 fixed questions, each with one gold chunk id and a ground-truth answer. The gold ids were computed against the sample documents, so regenerate them if you replace the corpus. Each script writes its results to results/.

Sample run (offline hashing embedder, no LLM key):

Metric	Value
Mean Recall@5	0.933
Mean Hit Rate@5	0.933
Mean MRR	0.867
Mean nDCG@5	0.884
Mean Context Precision@5	0.187

These numbers come from the weaker offline embedder, so re-run the scripts with all-MiniLM-L6-v2 for representative results. Context precision is low only because top_k=5 on a small sample corpus with one gold chunk per question gives a large denominator. Faithfulness and relevance show as null when no ANTHROPIC_API_KEY is set, and the harness reports that instead of inventing a score.

Cost model

Embedded ChromaDB (disk plus a small shared compute share) compared with an illustrative managed vector database. Full assumptions are in eval/cost_analysis.py.

Vectors	Storage (GB)	Embedded cost / month	Managed DB cost / month	Savings
100K	0.19	~$25	~$70	64%
1M	1.9	~$25	~$280	91%
10M	19.0	~$27	~$1,200	98%

Assumptions: 384-dimension float32 vectors, about 0.5 KB metadata per vector, $0.08/GB/month disk, a flat $25/month compute share, and illustrative managed-DB price tiers ($70 / $280 / $1,200). The managed prices are placeholders, not a quote from any provider.

When to move to a managed database: once the index no longer fits on one machine (roughly tens of millions of vectors), or when you need multi-region replication, an uptime SLA, or query fan-out across machines. Below that, the embedded store is cheaper and fast enough.

Project structure
cost-efficient-rag/
├── data/
│   ├── raw_documents/       # sample .md / .html / .pdf / .txt files
│   └── eval_dataset.json    # 15 fixed questions with gold chunk ids
├── src/
│   ├── config.py            # settings from environment (pydantic-settings)
│   ├── ingestion.py         # loaders, chunking, SHA-256 chunk ids
│   ├── embeddings.py        # sentence-transformers or hashing fallback
│   ├── vector_store.py      # ChromaDB persistent client wrapper
│   ├── rag_pipeline.py      # retrieve -> fallback -> prompt -> Claude
│   ├── logger.py            # per-query latency, token and cost log (JSONL)
│   └── api.py               # FastAPI: /ingest /query /stats /health
├── eval/
│   ├── evaluate_retrieval.py
│   ├── evaluate_answer.py
│   └── cost_analysis.py
├── .env.example
├── requirements.txt
└── README.md
Author

Shiva Prasad Chinthoju: GitHub | LinkedIn
