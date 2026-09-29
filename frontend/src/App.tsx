import { useEffect, useState, type FormEvent } from "react";

type ServiceHealth = {
  status: "ok" | "degraded";
  service: string;
  database: "ok" | "unavailable";
};

type DocumentSummary = {
  id: string;
  filename: string;
  file_size: number;
  status: "UPLOADED" | "PROCESSING" | "COMPLETED" | "FAILED" | "DELETED";
  page_count: number | null;
  chunk_count: number;
  error_message: string | null;
  created_at: string;
};

type QueryCitation = {
  citation_id: number;
  document_name: string;
  page: number;
  score: number;
};

type QueryResponse = {
  answer: string;
  citations: QueryCitation[];
  metadata: {
    retrieval_count: number;
    reranked_count: number;
    reranker: string;
    returned_count: number;
    rerank_latency_ms: number;
    answer_evaluation_attempts: number;
    answer_evaluation_score: number;
    answer_evaluation_passed: boolean;
    cache_hit: boolean;
    cache_similarity: number | null;
    latency_ms: number;
  };
};

function App() {
  const [health, setHealth] = useState<ServiceHealth | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadMessage, setUploadMessage] = useState<string | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [question, setQuestion] = useState("");
  const [queryResult, setQueryResult] = useState<QueryResponse | null>(null);
  const [queryError, setQueryError] = useState<string | null>(null);
  const [isQuerying, setIsQuerying] = useState(false);

  useEffect(() => {
    const loadHealth = async () => {
      try {
        const response = await fetch("/api/v1/health");
        const payload = (await response.json()) as ServiceHealth;
        if (!response.ok) {
          throw new Error(payload.database === "unavailable" ? "Database is unavailable" : "API is unavailable");
        }
        setHealth(payload);
      } catch (requestError) {
        setError(requestError instanceof Error ? requestError.message : "Unable to reach the API");
      }
    };

    void loadHealth();
  }, []);

  const isReady = health?.status === "ok" && health.database === "ok";

  useEffect(() => {
    if (!isReady) {
      return;
    }

    const loadDocuments = async () => {
      try {
        const response = await fetch("/api/v1/documents");
        if (!response.ok) {
          throw new Error("Unable to load documents");
        }
        setDocuments((await response.json()) as DocumentSummary[]);
      } catch (requestError) {
        setError(requestError instanceof Error ? requestError.message : "Unable to load documents");
      }
    };

    void loadDocuments();
    const intervalId = window.setInterval(loadDocuments, 3000);
    return () => window.clearInterval(intervalId);
  }, [isReady]);

  const handleUpload = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!selectedFile) {
      setUploadMessage("Choose a PDF first.");
      return;
    }

    setIsUploading(true);
    setUploadMessage(null);
    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
      const response = await fetch("/api/v1/documents", { method: "POST", body: formData });
      const payload = (await response.json()) as { detail?: string; filename?: string };
      if (!response.ok) {
        throw new Error(payload.detail ?? "Upload failed");
      }
      setUploadMessage(`${payload.filename ?? selectedFile.name} queued for processing.`);
      setSelectedFile(null);
      event.currentTarget.reset();
    } catch (uploadError) {
      setUploadMessage(uploadError instanceof Error ? uploadError.message : "Upload failed");
    } finally {
      setIsUploading(false);
    }
  };

  const formatBytes = (bytes: number) => `${(bytes / 1024 / 1024).toFixed(2)} MB`;

  const statusLabel = (status: DocumentSummary["status"]) => status.toLowerCase();

  const handleDeleteDocument = async (documentId: string, filename: string) => {
    if (!window.confirm(`Delete ${filename}?`)) {
      return;
    }
    try {
      const response = await fetch(`/api/v1/documents/${documentId}`, { method: "DELETE" });
      if (!response.ok) {
        const payload = (await response.json()) as { detail?: string };
        throw new Error(payload.detail ?? "Unable to delete document");
      }
      setDocuments((current) => current.filter((document) => document.id !== documentId));
    } catch (deleteError) {
      setError(deleteError instanceof Error ? deleteError.message : "Unable to delete document");
    }
  };

  const handleQuery = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!question.trim()) {
      setQueryError("Ask a question about your uploaded documents.");
      return;
    }

    setIsQuerying(true);
    setQueryError(null);
    try {
      const response = await fetch("/api/v1/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: question, top_k: 5 }),
      });
      const payload = (await response.json()) as QueryResponse & { detail?: string };
      if (!response.ok) {
        throw new Error(payload.detail ?? "Question failed");
      }
      setQueryResult(payload);
    } catch (requestError) {
      setQueryError(requestError instanceof Error ? requestError.message : "Question failed");
      setQueryResult(null);
    } finally {
      setIsQuerying(false);
    }
  };

  return (
    <main className="page-shell">
      <nav className="topbar">
        <div className="brand-mark">EK</div>
        <div>
          <p className="eyebrow">Enterprise AI</p>
          <p className="brand-name">Knowledge Platform</p>
        </div>
        <span className="phase-chip">Phase 5 · Reranked Retrieval</span>
      </nav>

      <section className="hero">
        <div>
          <p className="eyebrow accent">Secure knowledge, grounded answers</p>
          <h1>A reliable home for your organization’s knowledge.</h1>
          <p className="hero-copy">
            The platform foundation is online. Upload documents, let the ingestion worker index them, and ask
            grounded questions against the resulting knowledge base.
          </p>
        </div>
        <div className={`readiness-card ${isReady ? "ready" : "waiting"}`}>
          <span className="status-dot" />
          <div>
            <p className="card-label">Platform readiness</p>
            <p className="card-value">
              {isReady ? "API + PostgreSQL connected" : error ?? "Waiting for API + PostgreSQL"}
            </p>
          </div>
        </div>
      </section>

      <section className="foundation-grid" aria-label="Phase 1 foundation status">
        <article className="info-card">
          <span className="card-index">01</span>
          <h2>FastAPI service</h2>
          <p>Typed HTTP foundation with OpenAPI documentation and health contracts.</p>
          <span className="card-status">Online</span>
        </article>
        <article className="info-card">
          <span className="card-index">02</span>
          <h2>PostgreSQL</h2>
          <p>Transactional source of truth, connected asynchronously through SQLAlchemy.</p>
          <span className={`card-status ${health?.database === "ok" ? "" : "muted"}`}>
            {health?.database === "ok" ? "Connected" : "Pending"}
          </span>
        </article>
        <article className="info-card">
          <span className="card-index">03</span>
          <h2>React workspace</h2>
          <p>Vite-powered shell ready for documents, chat, citations, and administration.</p>
          <span className="card-status">Online</span>
        </article>
      </section>

      <section className="documents-section" aria-label="Document ingestion">
        <div className="section-heading">
          <div>
            <p className="eyebrow accent">Phase 2 · Document ingestion</p>
            <h2>Bring your knowledge into the platform.</h2>
          </div>
          <p>PDF files are stored privately, extracted in the background, and indexed into searchable chunks.</p>
        </div>

        <form className="upload-panel" onSubmit={handleUpload}>
          <div>
            <p className="card-label">Upload a PDF</p>
            <p className="upload-hint">Maximum 25 MB · processing runs in Celery</p>
          </div>
          <label className="file-picker">
            <span>{selectedFile?.name ?? "Choose file"}</span>
            <input
              type="file"
              accept="application/pdf,.pdf"
              onChange={(event) => setSelectedFile(event.target.files?.[0] ?? null)}
            />
          </label>
          <button type="submit" disabled={isUploading || !isReady}>
            {isUploading ? "Uploading…" : "Upload PDF"}
          </button>
          {uploadMessage && <p className="upload-message">{uploadMessage}</p>}
        </form>

        <div className="documents-panel">
          <div className="documents-heading">
            <p className="card-label">Document library</p>
            <span>{documents.length} {documents.length === 1 ? "document" : "documents"}</span>
          </div>
          {documents.length === 0 ? (
            <p className="empty-state">No documents yet. Upload a PDF to start the ingestion pipeline.</p>
          ) : (
            <div className="document-list">
              {documents.map((document) => (
                <div className="document-row" key={document.id}>
                  <div>
                    <p className="document-name">{document.filename}</p>
                    <p className="document-meta">
                      {formatBytes(document.file_size)} · {document.page_count ?? "—"} pages · {document.chunk_count} chunks
                    </p>
                  </div>
                  <div className="document-status">
                    <span className={`status-pill ${statusLabel(document.status)}`}>{statusLabel(document.status)}</span>
                    {document.error_message && <span className="document-error">{document.error_message}</span>}
                    <button
                      type="button"
                      className="document-delete"
                      onClick={() => void handleDeleteDocument(document.id, document.filename)}
                    >
                      Delete
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </section>

      <section className="query-section" aria-label="Document question answering">
        <div className="section-heading">
          <div>
            <p className="eyebrow accent">Phase 5 · Reranked RAG</p>
            <h2>Ask the documents, not the void.</h2>
          </div>
          <p>Search is filtered to the current demo organization and every result keeps its source page.</p>
        </div>

        <form className="query-panel" onSubmit={handleQuery}>
          <textarea
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            placeholder="What does the uploaded documentation say about…?"
            rows={3}
            maxLength={2000}
          />
          <button type="submit" disabled={isQuerying || !isReady}>
            {isQuerying ? "Searching…" : "Ask documents"}
          </button>
        </form>

        {queryError && <p className="query-error">{queryError}</p>}
        {queryResult && (
          <div className="answer-panel">
            <p className="card-label">Grounded response</p>
            <p className="answer-text">{queryResult.answer}</p>
            <div className="citation-list">
              {queryResult.citations.map((citation) => (
                <span className="citation" key={`${citation.citation_id}-${citation.document_name}-${citation.page}`}>
                  [{citation.citation_id}] {citation.document_name} · Page {citation.page}
                </span>
              ))}
            </div>
            <p className="query-meta">
              {queryResult.metadata.returned_count} sources · {queryResult.metadata.latency_ms} ms · {queryResult.metadata.reranker} · evaluation {queryResult.metadata.answer_evaluation_passed ? "passed" : "failed"}{queryResult.metadata.cache_hit ? " · FAQ cache" : ""}
            </p>
          </div>
        )}
      </section>

      <footer className="footer-line">
        <span>Build status</span>
        <span className="footer-rule" />
        <span>{health?.service ?? "Enterprise AI Knowledge Platform"}</span>
      </footer>
    </main>
  );
}

export default App;
