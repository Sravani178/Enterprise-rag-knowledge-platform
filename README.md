# Enterprise AI Knowledge Platform

Production-oriented RAG platform, built incrementally from a runnable local MVP.

## Phase 1: project setup

Phase 1 provides:

- FastAPI backend with live and database-readiness health endpoints
- Async PostgreSQL connectivity through SQLAlchemy and Psycopg 3
- React + TypeScript frontend powered by Vite
- Docker Compose orchestration for the API, frontend, and PostgreSQL
- A small backend test suite and a documented local workflow

## Phase 2: document ingestion foundation

Phase 2 adds:

- PDF-only upload validation with a 25 MB limit
- S3-compatible private object storage through MinIO
- Redis-backed Celery worker orchestration
- PyMuPDF page extraction
- Document status tracking: `UPLOADED`, `PROCESSING`, `COMPLETED`, or `FAILED`
- Extracted page text stored as JSON in object storage for Phase 3 chunking

The temporary `X-Organization-ID` header selects a tenant for local development. It defaults to a stable demo organization and will be replaced by authenticated membership resolution in Phase 9.

## Phase 3: chunking, vector search, and basic RAG

Phase 3 adds:

- Hybrid semantic chunks: paragraph/sentence boundaries plus OpenAI embedding topic-shift detection, with hard size and overlap limits
- OpenAI `text-embedding-3-small` embeddings when configured for the Compose deployment
- Qdrant vector storage with organization filters
- Chunk metadata persisted in PostgreSQL
- `POST /api/v1/query` vector retrieval endpoint
- Grounded extractive answers with document/page citations
- OpenAI answer generation through the configured chat-completions endpoint

The application defaults and Docker Compose runtime use OpenAI embeddings and answer generation. The test suite overrides those providers with deterministic local implementations so tests never call an external API. Set `OPENAI_API_KEY` before starting Compose.

## Phase 4: hybrid retrieval

Phase 4 changes the query path from vector-only retrieval to a hybrid pipeline:

1. Qdrant retrieves a candidate set using the embedding similarity search.
2. Tenant-scoped document chunks are loaded and searched with a local BM25 keyword index, which helps with exact names, numbers, identifiers, and terminology.
3. Reciprocal Rank Fusion (RRF) combines both ranked lists without requiring their raw scores to be comparable.
4. The fused results are limited to the requested `top_k` and passed to the grounded answer generator.

The default candidate sizes are 20 vector results and 20 BM25 results, with `RRF_K=60`. They can be tuned through `VECTOR_CANDIDATE_K`, `BM25_CANDIDATE_K`, and `RRF_K`. The query response metadata reports the vector, keyword, fused-candidate, and returned counts.

For this MVP, PostgreSQL stores the authoritative chunks, while BM25 performs lexical ranking in application memory for each query. This keeps the retrieval algorithm database-independent. At larger scale, the BM25 index should be made persistent or moved to a dedicated lexical-search service.

## Run with Docker Compose

Prerequisites: Docker Desktop with Compose v2.

```powershell
docker compose up --build
```

After the services are running, execute the end-to-end smoke test from the backend container:

```powershell
docker compose exec backend python scripts/smoke_test.py
```

The smoke test waits for database readiness, uploads a generated PDF, waits for the Celery worker to finish, queries the indexed document, and verifies the grounded answer and page-one citation. It is intentionally separate from the fast unit test suite because it requires the complete Docker Compose stack.

The worker also uses late task acknowledgements, rejects tasks when a worker process is lost, limits prefetching to one task, and retries transient PostgreSQL, object-storage, and Qdrant failures up to three times with backoff. Document chunks have a database uniqueness constraint on `(document_id, chunk_index)` to prevent duplicate chunk rows. Each task claims the document under a PostgreSQL row lock, permits a redelivered Celery task to reclaim its own lease after a worker crash, rejects active duplicate tasks, and can reclaim a stale `PROCESSING` document after `PROCESSING_STALE_AFTER_SECONDS`.

The API also applies the organization filter to both chunks and their parent documents. If document metadata cannot be committed after object upload, it attempts to delete the uploaded object so failed requests do not leave avoidable storage orphans. Processing-state columns, idempotency constraints, authentication tables, and the chunk search index are managed by Alembic migrations.

## Production hardening foundation

The repository now includes an Alembic baseline migration in `backend/migrations`. Docker starts with `alembic upgrade head`; use a fresh database or explicitly baseline an older development database before upgrading because the original prototype used `create_all()`.

JWT authentication and organization roles are available through:

```text
POST /api/v1/auth/register
POST /api/v1/auth/login
```

Set `AUTH_REQUIRED=true` and replace `JWT_SECRET` in production. The API verifies the token user and current organization membership in PostgreSQL. Upload/reprocess require `OWNER`, `ADMIN`, or `MEMBER`; deletion requires `OWNER` or `ADMIN`; reading and querying allow all membership roles.

Keyword retrieval uses the tenant-scoped `BM25Index` in `app/retrieval/bm25.py`. PostgreSQL is still used to store and load chunks, but PostgreSQL full-text ranking is not used by the query path.

The OpenAI runtime uses `text-embedding-3-small` with dimension 1536 and `gpt-4o-mini` for answer generation. The Qdrant collection name includes the embedding dimension so old 384-dimensional local vectors are not mixed with the new vectors; documents must be reprocessed after switching providers.

Ingestion uses `SEMANTIC_CHUNKING_ENABLED=true` by default. It first creates paragraph/sentence units, embeds those units, starts a new chunk when adjacent-unit cosine similarity falls below `SEMANTIC_CHUNK_SIMILARITY_THRESHOLD`, and still enforces `CHUNK_SIZE` and `CHUNK_OVERLAP`. Final chunks are embedded again for Qdrant retrieval.

The query path also has a tenant-scoped semantic FAQ cache in Redis database 2. It stores successful grounded responses with their question embedding, ranks cached questions by hit count, checks exact normalized questions first, then checks semantic similarity using `QUESTION_CACHE_SIMILARITY_THRESHOLD`. Upload, reprocess, and deletion bump the organization cache version so older answers are ignored. Cache failures are fail-open: retrieval continues normally.

Uploads support an optional `Idempotency-Key` header. Redis-backed rate limiting covers authentication, document writes, and queries when `RATE_LIMIT_ENABLED=true`. The default local Compose configuration leaves authentication and rate limiting disabled for the existing demo workflow.

`DELETE /api/v1/documents/{document_id}` removes Qdrant vectors, original/extracted objects, and PostgreSQL chunks before marking the document `DELETED`. The Docker smoke test exercises upload replay, ingestion, querying, citation validation, and deletion.

## Phase 3: concurrency and race-condition hardening

Query capacity is protected by a local semaphore and, when Redis is available, a shared sorted-set lease at `query:concurrency:active`. This keeps the configured 100-query limit across API replicas; Redis failure fails open to the per-process semaphore so query correctness is preserved. Document workers claim rows with PostgreSQL locks, renew processing leases while alive, reject duplicate fresh deliveries, and allow only stale redeliveries to reclaim a task. Celery Beat also requeues stale `PROCESSING` documents. Upload/delete/query visibility and cache-version checks remain tenant-scoped.

## Phase 4: failure handling

Transient document-processing failures from PostgreSQL, MinIO/S3, Qdrant, or embeddings are retried by Celery up to `max_retries=3` with backoff and jitter. Before a retry, the worker changes the document from `PROCESSING` back to `UPLOADED` and clears its processing lease, so the redelivered task can claim it safely. When the final retry is exhausted, the document becomes `FAILED` with a bounded error message. PDF extraction errors remain terminal because retrying invalid PDF data cannot repair it. OpenAI requests retry HTTP 408/429/5xx responses and network errors with bounded backoff, including numeric `Retry-After` handling. Redis question-cache and concurrency failures fail open where correctness permits, while API dependency failures return HTTP 503. MinIO/S3 and Redis clients use bounded connect/read or socket timeouts so an outage does not hold request or worker capacity indefinitely.

## Phase 5: distributed consistency

Document indexing treats PostgreSQL as the metadata authority, but does not expose a document as `COMPLETED` until Qdrant vectors and PostgreSQL chunks have both been written. If a worker fails after an external write, the retry replaces the document's vectors and SQL chunks idempotently. The periodic `documents.reconcile` task compares Qdrant document references and vector counts with PostgreSQL, removes vectors for missing or non-completed documents, requeues completed documents whose vector count is inconsistent, and finalizes `DELETING` documents after a previous MinIO/Qdrant cleanup failure. Object-storage reconciliation removes old unreferenced objects after `STORAGE_ORPHAN_GRACE_SECONDS`. All cleanup operations are tenant- and document-filtered.

## Phase 6: database and Redis connection pools

PostgreSQL async and sync engines use bounded pools configured by `DB_POOL_SIZE`, `DB_MAX_OVERFLOW`, `DB_POOL_TIMEOUT_SECONDS`, and `DB_POOL_RECYCLE_SECONDS`, with connection pre-ping enabled. All application Redis clients use a shared construction pattern with a bounded `BlockingConnectionPool`, finite pool wait time, and configured socket/connect timeouts. This applies to the semantic question cache, distributed query concurrency, rate limiting, and readiness checks. Celery also uses a bounded broker pool through `CELERY_BROKER_POOL_LIMIT`. Pool exhaustion is therefore observable as a handled dependency error instead of creating unlimited sockets or waiting indefinitely.

## Phase 7: persistent BM25 indexing

BM25 no longer loads every completed chunk and rebuilds an index for each query. Migration `0003_persistent_bm25` adds tenant-scoped `bm25_documents` statistics and `bm25_postings` term-frequency rows. Ingestion writes those rows in the same transaction as PostgreSQL chunk metadata, while reprocessing and deletion remove the old postings before replacing or deleting chunks. Queries read only postings for the requested terms, calculate BM25 using persisted document counts and average lengths, then join the matching chunks for citations. PostgreSQL remains the durable metadata/index store; PostgreSQL full-text ranking is not used for lexical retrieval.

## Phase 8: Celery timeouts and dead-letter handling

Celery workers use configurable soft and hard time limits through `CELERY_SOFT_TIME_LIMIT_SECONDS` and `CELERY_TIME_LIMIT_SECONDS`. Soft timeouts are retryable document-processing failures; hard limits terminate a worker process, and late acknowledgements plus worker-lost rejection allow the task lease to be recovered. After the configured retry budget is exhausted, or when a non-retryable task fails, the worker writes one idempotent row to `task_dead_letters` with task ID, document ID, payload, exception, retry count, and `PENDING` status. Invalid PDF extraction failures are also recorded there. The database-backed DLQ is used because Redis transport does not provide a portable durable dead-letter queue across deployments; if PostgreSQL itself is unavailable, the original task failure remains authoritative and normal task/lease recovery continues.

## Phase 9: observability

The API assigns or propagates a validated `X-Request-ID`, returns it on every response, and emits structured JSON logs containing correlation ID, route, status, and latency without logging query contents. `/metrics` exposes process-local Prometheus-compatible counters and duration summaries for HTTP requests, errors, document-processing outcomes, and Celery dead-letter events. Route labels use registered templates rather than raw path values to avoid high-cardinality document IDs. Worker failures include task metadata in structured logs, while the existing health endpoint remains the dependency-readiness signal.

## Phase 10: load testing

`backend/scripts/load_test.py` runs synchronized asynchronous virtual-user scenarios at the 10, 100, and 1,000-user levels. It reports request totals, status codes, error rate, throughput, and p50/p95/p99/max latency. The 1,000-user level requires an explicit `--allow-large-load` guard. Use a live API for deployment testing:

```powershell
python backend/scripts/load_test.py --base-url http://localhost:8000 `
  --path /api/v1/health/live --levels 10,100,1000 --allow-large-load
```

For no-Docker verification, `--in-process` mounts the FastAPI application through HTTPX and can exercise endpoints that do not require external services. `--json-output` produces machine-readable output; in-process application logs are suppressed for that mode so the output remains valid JSON:

```powershell
.\.venv\Scripts\python.exe backend\scripts\load_test.py `
  --in-process --path /metrics --levels 10,100 --requests-per-user 1 --json-output
```

The harness is intentionally bounded and does not claim to prove live PostgreSQL, Redis, MinIO, Qdrant, Celery, or OpenAI behavior. Those require the Dockerized or separately provisioned integration environment.

## Phase 11: security and tenant attack testing

Authenticated requests derive the tenant exclusively from the JWT organization claim. An `X-Organization-ID` header can be used only in local authentication-disabled mode; when a principal exists, a mismatched header returns `403`. Qdrant searches apply an organization payload filter, PostgreSQL visibility checks require the same organization and `COMPLETED` status, BM25 postings are tenant-scoped, and question-cache keys include the organization ID. Document list/get/reprocess/delete queries also include the organization predicate.

`backend/tests/test_security.py` covers production fail-closed configuration, default/unsupported JWT settings, mandatory JWT time claims, tenant-header spoofing, blank/oversized input, request-ID header-injection attempts, and security response headers. Existing phase tests additionally prove Qdrant tenant filtering, tenant cache invalidation, and grounded query citation behavior. Production configuration must set `AUTH_REQUIRED=true`, `RATE_LIMIT_ENABLED=true`, and a non-default JWT secret of at least 32 characters.

## Phase 12: production deployment

Production uses [docker-compose.prod.yml](docker-compose.prod.yml), separate from the development stack. Copy `.env.production.example` to `.env.production`, replace every placeholder with real credentials, and keep the database/Redis URLs synchronized with those credentials. The production stack does not publish PostgreSQL, Redis, MinIO, or Qdrant ports to the host; only the frontend port is exposed.

The `migrate` one-shot service runs `alembic upgrade head` before the API, worker, and beat services start. The backend image runs as non-root `appuser`, is read-only except for `/tmp` and the model-cache volume, and has a liveness healthcheck. The frontend is a multi-stage Vite build served by Nginx; Nginx proxies `/api/` to the backend and supports client-side routing.

Start the production stack with:

```powershell
Copy-Item .env.production.example .env.production
# Edit .env.production and replace all replace-with-* values.
docker compose --env-file .env.production -f docker-compose.prod.yml config
docker compose --env-file .env.production -f docker-compose.prod.yml up --build -d
```

Verify the deployment with `docker compose --env-file .env.production -f docker-compose.prod.yml ps` and `curl http://localhost:8080/api/v1/health/live`. Stop it with `docker compose --env-file .env.production -f docker-compose.prod.yml down`; named volumes preserve database, object-storage, vector, Redis, and model-cache data. Use a managed PostgreSQL/Redis/object store/vector service and an external TLS reverse proxy for a real multi-node production deployment.

Docker image builds and live service recovery remain environment-dependent because Docker Desktop is not installed in the current development environment. The Compose file is validated statically here, while the existing no-Docker test suite verifies application behavior.

## Phase 1: query correction and rewriting

When enabled, `QueryUnderstandingService` uses OpenAI with a JSON-only prompt to correct obvious spelling mistakes and create one retrieval-oriented rewrite. The original query is retained for the answer-generation prompt. Candidate rewrites are accepted only when they are non-empty, length-bounded, and preserve tokens that look like IDs, product codes, filenames, acronyms, or proper names. If OpenAI is unavailable or returns invalid output, retrieval fails open to the original query.

Query responses expose `original_query`, `corrected_query`, and `retrieval_query` in metadata. Configure the stage with `QUERY_UNDERSTANDING_ENABLED` and `QUERY_UNDERSTANDING_MAX_TOKENS`.

## Phase 5: answer-quality evaluation

The query pipeline applies the configured cross-encoder to the RRF candidates before the bounded answer-quality loop. The generator creates a draft, the evaluator checks its evidence overlap and citations locally or asks OpenAI to judge grounding, and a failed draft is revised once using the evaluator feedback. If the revised answer still fails, the API returns a conservative insufficient-evidence response.

The pipeline is now:

```text
Vector top 20 + BM25 top 20
        ↓
RRF candidate set
        ↓
Cross-encoder candidate ordering
        ↓
Draft answer
        ↓
Grounding and citation evaluator
        ↓
Optional revision loop (maximum 2 attempts)
        ↓
Grounded answer and citations
```

The default cross-encoder is `cross-encoder/ms-marco-MiniLM-L-6-v2`. Docker installs the `reranking` dependency extra, and the model is loaded lazily on the first query. Candidate text is bounded by `RERANKER_MAX_TEXT_CHARS`, model input length and batch size are configurable, inference concurrency is bounded, and the API timeout-protects reranking. If the model cannot load or score, the configured lexical fallback is reported as `lexical_fallback`; the response never claims Cross-Encoder usage when fallback ranking was used. Pin a model revision with `RERANKER_MODEL_REVISION` for reproducible deployments. A Celery Beat reconciliation task runs every five minutes to remove orphaned Qdrant vectors and requeue completed documents whose vector count no longer matches the PostgreSQL chunk count. The evaluation settings are controlled by `ANSWER_EVALUATION_ENABLED`, `ANSWER_EVALUATION_MAX_ATTEMPTS`, and `ANSWER_EVALUATION_MIN_SCORE`. The API reports evaluation attempts, score, and pass/fail status in query metadata.

Open:

- Frontend: http://localhost:5173
- API documentation: http://localhost:8000/docs
- Live health: http://localhost:8000/api/v1/health/live
- Database readiness: http://localhost:8000/api/v1/health/ready
- MinIO console: http://localhost:9001 (`minioadmin` / `minioadmin`)
- Qdrant dashboard/API: http://localhost:6333

Stop the stack with `Ctrl+C`, or run `docker compose down`. PostgreSQL data is kept in the named `postgres_data` volume.

## Run the backend locally

Python 3.12+ is required. Install the backend and development dependencies:

```powershell
python -m pip install -e ".\backend[dev]"
```

Copy `.env.example` to `.env`, make sure PostgreSQL is available, and start the API:

```powershell
python -m app.db.init
uvicorn app.main:app --app-dir backend --reload
```

For Phases 2 and 3 locally, also run Redis, MinIO, Qdrant, and a Celery worker, then upload a PDF:

```powershell
celery -A app.worker.celery_app:celery_app worker --loglevel=INFO
```

```powershell
curl.exe -X POST http://localhost:8000/api/v1/documents `
  -F "file=@C:\path\to\document.pdf"
```

List processing status with:

```powershell
curl.exe http://localhost:8000/api/v1/documents
```

Ask the indexed documents:

```powershell
curl.exe -X POST http://localhost:8000/api/v1/query `
  -H "Content-Type: application/json" `
  -d '{"query":"What was FY2025 revenue?","top_k":5}'
```

Run tests and lint checks:

```powershell
pytest backend/tests
ruff check backend
```

### No-Docker verification

When Docker Desktop or local PostgreSQL/Redis/MinIO/Qdrant are unavailable, the repository can still run its deterministic verification suite:

```powershell
.\.venv\Scripts\ruff.exe check backend --no-cache
.\.venv\Scripts\python.exe -m compileall -q backend\app backend\tests
.\.venv\Scripts\python.exe -m pytest backend\tests -p no:cacheprovider
```

This suite uses in-memory Qdrant, local PDF generation/extraction, local test embeddings, mocked OpenAI requests, BM25, RRF, and the extractive test generator. It does not prove live PostgreSQL, Redis, MinIO, Qdrant, or OpenAI connectivity; those require external services.

## Repository layout

```text
backend/       FastAPI application and tests
frontend/      React + TypeScript application
docker-compose.yml
```
