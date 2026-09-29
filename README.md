# Enterprise RAG Knowledge Platform

A production-oriented, multi-tenant Retrieval-Augmented Generation platform for uploading enterprise PDFs and asking grounded questions with citations.

The platform combines OpenAI embeddings and LLM generation, Qdrant vector search, persistent PostgreSQL BM25 search, Cross-Encoder reranking, semantic question caching, asynchronous Celery ingestion, JWT/RBAC security, and failure recovery.

## What it provides

- PDF upload with size, MIME, filename, and content validation
- Tenant-scoped document storage and retrieval
- Hybrid semantic and lexical search
- Hybrid semantic/fixed-size chunking with overlap limits
- OpenAI `text-embedding-3-small` embeddings
- OpenAI `gpt-4o-mini` answer generation
- Cross-Encoder reranking with lexical fallback
- Grounding and citation evaluation with bounded revision
- Exact and semantic frequently asked question caching
- Idempotent uploads and duplicate-content protection
- JWT authentication and organization-level RBAC
- Redis-backed rate limiting and distributed query concurrency
- Celery retries, worker leases, heartbeats, reconciliation, and a database-backed DLQ
- Structured logs, request IDs, health checks, and process metrics
- Development and production Docker Compose configurations

## Architecture

```text
                         ┌──────────────────────┐
                         │ React + Nginx frontend│
                         └──────────┬───────────┘
                                    │ /api
                         ┌──────────▼───────────┐
                         │ FastAPI API           │
                         │ JWT/RBAC, limits,     │
                         │ cache, observability  │
                         └──────┬────────┬───────┘
                                │        │
                         ┌──────▼───┐ ┌──▼─────────┐
                         │PostgreSQL│ │   Redis    │
                         │metadata, │ │cache, rate │
                         │BM25, DLQ │ │limits, locks│
                         └──────┬───┘ └────────────┘
                                │
       ┌────────────────────────┼────────────────────────┐
       │                        │                        │
┌──────▼─────┐          ┌───────▼────────┐       ┌───────▼──────┐
│   MinIO    │          │ Celery worker  │       │    Qdrant    │
│ PDF/files  │          │ extraction and │       │ vector index │
└────────────┘          │ indexing       │       └──────────────┘
                        └────────────────┘
```

### Query flow

```text
Request
  → authentication and tenant resolution
  → rate limiting and concurrency control
  → exact/semantic question cache
  → query correction and conservative rewriting
  → OpenAI query embedding
  → Qdrant vector search + persistent BM25 search
  → Reciprocal Rank Fusion
  → bounded Cross-Encoder reranking
  → grounded answer generation
  → answer evaluation and bounded revision
  → citations and response cache
```

The original user question is preserved for answer generation. The corrected retrieval query is used only for search:

```text
original_query → corrected_query → retrieval_query → retrieval

original_query → answer generation
```

### Document flow

```text
Upload
  → validation and idempotency checks
  → MinIO object upload
  → PostgreSQL document metadata
  → Celery task
  → worker lease and heartbeat
  → PDF extraction
  → hybrid semantic chunking
  → embeddings and Qdrant indexing
  → PostgreSQL chunks and BM25 postings
  → COMPLETED state and cache invalidation
```

PostgreSQL is the lifecycle authority. Qdrant and MinIO are external projections repaired by the reconciliation task when drift occurs.

## Technology stack

| Area | Technology |
| --- | --- |
| API | FastAPI, Python 3.12+ |
| Database | PostgreSQL, SQLAlchemy 2, Psycopg 3, Alembic |
| Object storage | MinIO locally, S3-compatible storage in production |
| Vector search | Qdrant |
| Lexical search | Persistent tenant-scoped BM25 postings in PostgreSQL |
| Cache and coordination | Redis |
| Background processing | Celery and Celery Beat |
| Embeddings | OpenAI `text-embedding-3-small` |
| Generation | OpenAI `gpt-4o-mini` |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` with lexical fallback |
| PDF extraction | PyMuPDF |
| Frontend | React, TypeScript, Vite, Nginx |
| Deployment | Docker Compose |

## Repository layout

```text
backend/
  app/
    api/              API dependencies, rate limits, concurrency, routes
    auth/             Password hashing, JWT creation and validation
    cache/            Tenant-scoped semantic question cache
    core/             Settings, Redis clients, OpenAI client, observability
    db/               SQLAlchemy engines and database initialization
    embeddings/       OpenAI and deterministic test embeddings
    generation/       Answer generation and answer evaluation
    ingestion/        PDF extraction and hybrid chunking
    models/           PostgreSQL models
    query/            Query correction and rewriting
    reranking/        Cross-Encoder and lexical reranking
    retrieval/        BM25, RRF, visibility checks, retrieval types
    storage/          S3/MinIO adapter
    vectorstore/      Qdrant adapter
    worker/           Celery tasks and reconciliation
  migrations/         Alembic migrations
  scripts/            Load-test and smoke-test utilities
  tests/              Unit and deterministic integration-style tests
frontend/             React application and production Nginx image
docker-compose.yml    Development stack
docker-compose.prod.yml Production-oriented stack
.env.example          Local configuration template
.env.production.example Production configuration template
```

## API endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `POST` | `/api/v1/auth/register` | Create an organization owner and return a JWT |
| `POST` | `/api/v1/auth/login` | Authenticate a user and return a JWT |
| `POST` | `/api/v1/documents` | Upload and enqueue a PDF |
| `GET` | `/api/v1/documents` | List tenant-visible documents |
| `GET` | `/api/v1/documents/{id}` | Get a tenant-visible document |
| `POST` | `/api/v1/documents/{id}/reprocess` | Requeue document processing |
| `DELETE` | `/api/v1/documents/{id}` | Delete document and external index state |
| `POST` | `/api/v1/query` | Ask a grounded question with citations |
| `GET` | `/api/v1/health/live` | Liveness check |
| `GET` | `/api/v1/health/ready` | PostgreSQL, Redis, MinIO, and Qdrant readiness |
| `GET` | `/metrics` | Process-local Prometheus-style metrics |

## Run locally with Docker Compose

Prerequisite: Docker Desktop with Compose v2.

1. Copy the local environment template:

```powershell
Copy-Item .env.example .env
```

2. Set `OPENAI_API_KEY` in `.env` if using OpenAI embeddings and generation.

3. Start the development stack:

```powershell
docker compose up --build
```

4. Open:

- Frontend: <http://localhost:5173>
- API documentation: <http://localhost:8000/docs>
- Liveness: <http://localhost:8000/api/v1/health/live>
- Readiness: <http://localhost:8000/api/v1/health/ready>
- MinIO console: <http://localhost:9001>
- Qdrant API: <http://localhost:6333>

5. Run the complete Docker smoke test:

```powershell
docker compose exec backend python scripts/smoke_test.py
```

The smoke test exercises upload, duplicate submission, worker processing, querying, citation validation, and deletion. It requires PostgreSQL, Redis, MinIO, Qdrant, Celery, and the API to be running.

Stop the development stack with:

```powershell
docker compose down
```

## Run without Docker

Python 3.12+ is required.

Install backend development dependencies:

```powershell
python -m pip install -e ".\backend[dev]"
```

The deterministic test suite does not require PostgreSQL, Redis, MinIO, Qdrant, OpenAI, or Docker. It uses in-memory Qdrant, local PDF extraction, deterministic test embeddings, mocked provider calls, and an extractive test generator.

Run verification from the repository root:

```powershell
.\.venv\Scripts\ruff.exe check backend --no-cache
.\.venv\Scripts\python.exe -m compileall -q backend\app backend\scripts backend\tests
.\.venv\Scripts\python.exe -m pytest backend\tests -p no:cacheprovider
```

The current repository verification result is 68 passing tests. These tests prove isolated behavior; they do not replace live Docker/service integration testing.

### Offline load testing

The bounded load harness supports 10, 100, and 1,000 virtual-user levels. The in-process mode is useful without Docker:

```powershell
.\.venv\Scripts\python.exe backend\scripts\load_test.py `
  --in-process --path /metrics --levels 10,100 --requests-per-user 1 --json-output
```

The 1,000-user level requires an explicit guard:

```powershell
.\.venv\Scripts\python.exe backend\scripts\load_test.py `
  --in-process --path /metrics --levels 1000 --allow-large-load --json-output
```

This measures the API harness and endpoint concurrency only. It does not represent full RAG query capacity.

## Production deployment

Use the separate production Compose file and never deploy the development credentials or development bypass configuration.

1. Create the production environment file:

```powershell
Copy-Item .env.production.example .env.production
```

2. Replace every `replace-with-*` value with real secrets. At minimum configure:

- PostgreSQL credentials and `DATABASE_URL`
- Redis password and all Redis URLs
- MinIO/S3 credentials
- `OPENAI_API_KEY`
- A random `JWT_SECRET` with at least 32 characters
- Production `CORS_ORIGINS`

3. Validate the rendered Compose configuration:

```powershell
docker compose --env-file .env.production -f docker-compose.prod.yml config
```

4. Build and start:

```powershell
docker compose --env-file .env.production -f docker-compose.prod.yml up --build -d
```

5. Verify:

```powershell
docker compose --env-file .env.production -f docker-compose.prod.yml ps
curl.exe http://localhost:8080/api/v1/health/live
```

The production stack provides:

- A one-shot Alembic migration service before API and workers start
- Non-root backend containers
- Read-only backend and worker filesystems with temporary/model-cache volumes
- Multi-stage frontend build served by Nginx
- Nginx `/api/` proxying to the backend
- Internal-only PostgreSQL, Redis, MinIO, and Qdrant services
- Redis authentication
- Persistent named volumes for PostgreSQL, Redis, MinIO, Qdrant, and model cache
- Production-enforced `AUTH_REQUIRED=true` and `RATE_LIMIT_ENABLED=true`

Stop the production stack with:

```powershell
docker compose --env-file .env.production -f docker-compose.prod.yml down
```

Docker image builds and live recovery tests must still be executed in an environment with Docker Desktop or a provisioned container host.

## Security and tenancy

- JWTs are signed and require `exp`, `iat`, user, organization, role, and email claims.
- The current membership and role are checked in PostgreSQL on every authenticated request.
- Upload/reprocess require `OWNER`, `ADMIN`, or `MEMBER`.
- Delete requires `OWNER` or `ADMIN`.
- Query and read access are available to all organization membership roles.
- Production configuration refuses to start with authentication disabled.
- PostgreSQL document, chunk, and BM25 queries include organization filters.
- Qdrant searches filter on the organization payload.
- Redis cache and rate-limit keys include the organization ID.
- Upload filenames are sanitized before storage-key construction.
- PDF size, MIME type, and file signature are validated.
- Document text is explicitly treated as untrusted evidence in generation and evaluation prompts.

Authorization is organization-level. The `uploaded_by` field is recorded for audit context, but individual document ownership policies are not currently enforced.

## Reliability and consistency

### Query concurrency

Each API process has a local semaphore. When Redis is available, a shared sorted-set lease enforces the configured limit across API replicas. Lease expiry prevents crashed requests from permanently consuming capacity. During Redis failure, only per-process protection is guaranteed.

### Worker safety

Workers claim documents using PostgreSQL row locks and processing leases. Heartbeats renew active leases. Fresh duplicate deliveries are rejected, while stale redeliveries can be reclaimed. Celery uses late acknowledgement, worker-lost rejection, prefetch-one behavior, retries, time limits, and a database-backed dead-letter table.

### Distributed consistency

PostgreSQL is the document lifecycle authority. Qdrant vectors, MinIO objects, SQL chunks, and BM25 postings are cleaned up idempotently. Celery Beat reconciliation removes orphan vectors and objects, repairs vector-count mismatches, requeues stale processing, and finalizes interrupted deletion.

## Important configuration

| Variable | Purpose | Typical value |
| --- | --- | --- |
| `OPENAI_API_KEY` | OpenAI embeddings and LLM access | Required for OpenAI providers |
| `EMBEDDING_MODEL` | Embedding model | `text-embedding-3-small` |
| `EMBEDDING_DIMENSION` | Qdrant vector dimension | `1536` |
| `LLM_MODEL` | Answer-generation model | `gpt-4o-mini` |
| `CHUNK_SIZE` | Hard chunk character limit | `800` |
| `CHUNK_OVERLAP` | Chunk overlap limit | `120` |
| `SEMANTIC_CHUNK_SIMILARITY_THRESHOLD` | Topic-shift threshold | `0.78` |
| `VECTOR_CANDIDATE_K` | Qdrant candidate count | `20` |
| `BM25_CANDIDATE_K` | BM25 candidate count | `20` |
| `RERANKER_CANDIDATE_K` | Fused Cross-Encoder candidate cap | `20`, maximum `50` |
| `RERANKER_MODEL_REVISION` | Optional pinned model revision | Deployment-specific |
| `RERANKER_MAX_CONCURRENCY` | Per-process reranker concurrency | `2` |
| `QUERY_CONCURRENCY_LIMIT` | Distributed query limit | `100` |
| `QUERY_CONCURRENCY_LEASE_SECONDS` | Redis query lease duration | `120` |
| `QUESTION_CACHE_SIMILARITY_THRESHOLD` | Semantic cache match threshold | `0.94` |
| `PROCESSING_STALE_AFTER_SECONDS` | Worker lease timeout | `900` |
| `CELERY_MAX_RETRIES` | Document retry budget | `3` |
| `AUTH_REQUIRED` | Require JWT authentication | `true` in production |
| `RATE_LIMIT_ENABLED` | Enable Redis rate limiting | `true` in production |

All settings are environment-driven through [`backend/app/core/config.py`](backend/app/core/config.py). See [.env.example](.env.example) and [.env.production.example](.env.production.example).

## Current verification status

The repository currently passes:

```text
Ruff:       passed
Compilation: passed
Pytest:      68 passed
```

The following remain environment-dependent and require live integration testing:

- Docker image builds and Compose startup
- Complete upload → Celery → extraction → embedding → Qdrant/BM25 → query workflow
- PostgreSQL transaction and lock races
- Redis multi-replica leases and failure recovery
- MinIO and Qdrant recovery after partial failure
- Actual Celery worker crash and redelivery
- Live OpenAI and Hugging Face model behavior
- Full `/query` load testing against real dependencies
- Large-corpus performance at approximately 100,000 chunks

## Current architectural tradeoffs

- PostgreSQL stores BM25 postings, avoiding full index rebuilds but still requiring corpus-statistics work during search.
- Redis failure does not break cache correctness, but global query concurrency becomes per-process only.
- Cross-Encoder reranking improves relevance but adds model latency and resource usage.
- Celery and reconciliation provide eventual consistency because PostgreSQL, Qdrant, and MinIO do not share one transaction.
- The current application is organization-scoped rather than document-owner-scoped.
- Metrics are process-local and should be replaced or aggregated through a production observability system for multi-replica deployments.
