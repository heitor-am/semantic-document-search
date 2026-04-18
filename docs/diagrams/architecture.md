# Architecture — package-by-feature

Three top-level features (`ingestion/`, `retrieval/`, `evaluation/`) sit alongside `shared/` for genuinely cross-cutting code. Each feature owns its own `router`, `service`, `schemas`, and (where relevant) `repository` — features never reach into another feature's internals.

```mermaid
flowchart TB
  subgraph CLIENT["Client"]
    USER[curl / notebook / eval harness]
  end

  subgraph APP["FastAPI process"]
    direction TB

    subgraph ING["app/ingestion/"]
      IROUTER["router.py<br/>POST /ingest<br/>GET /ingest/jobs"]
      ISERVICE["service.py<br/>orchestrates pipeline"]
      IFSM["state.py<br/>FSM (transitions)"]
      ICHUNK["chunker.py<br/>parser.py<br/>indexer.py"]
      IREPO["repository.py<br/>SQLite jobs"]
    end

    subgraph RET["app/retrieval/"]
      RROUTER["router.py<br/>GET /search"]
      RSERVICE["service.py<br/>build_pipeline(strategy)"]
      RPIPE["pipeline.py<br/>Context + Stage executor"]
      RSTAGES["stages/<br/>HybridSearch · Reranker · ParentChild"]
    end

    subgraph EVAL["app/evaluation/"]
      RUNNER["runner.py<br/>replays golden YAML"]
      METRICS["metrics.py<br/>P@k · R@k · MRR · NDCG"]
    end

    subgraph SHARED["app/shared/"]
      AI["ai/<br/>OpenRouter client<br/>embeddings · rerank"]
      QDRANT["qdrant/<br/>VectorRepository<br/>sparse_encoder (FastEmbed)"]
      DB["db/<br/>SQLAlchemy session"]
      CORE["core/<br/>config · errors · logging"]
    end
  end

  subgraph EXT["External"]
    QC["(Qdrant Cloud)"]
    OR["(OpenRouter)"]
    SQ["(SQLite /data/jobs.db)"]
  end

  USER -->|HTTP| IROUTER
  USER -->|HTTP| RROUTER
  USER -->|HTTP| RUNNER

  IROUTER --> ISERVICE
  ISERVICE --> IFSM
  ISERVICE --> ICHUNK
  ISERVICE --> IREPO
  ICHUNK --> AI
  ICHUNK --> QDRANT
  IREPO --> DB

  RROUTER --> RSERVICE
  RSERVICE --> RPIPE
  RPIPE --> RSTAGES
  RSTAGES --> AI
  RSTAGES --> QDRANT

  RUNNER -->|GET /search| RROUTER
  RUNNER --> METRICS

  AI -->|HTTPS| OR
  QDRANT -->|HTTPS| QC
  DB --> SQ

  classDef feature fill:#1f3a5f,stroke:#3a6ea5,color:#fff
  classDef shared fill:#2d2d44,stroke:#666,color:#fff
  classDef external fill:#5a3a3a,stroke:#a06060,color:#fff
  class ING,RET,EVAL feature
  class SHARED shared
  class EXT external
```

## How to read this

- **Vertical layering inside each feature is allowed** (`router → service → repository`), but every arrow stops at the feature boundary unless it goes through `shared/`.
- **`shared/` is small on purpose.** Anything that lives there is used by ≥ 2 features. The OpenRouter client, the Qdrant repository, the DB session, and the config / error / logging primitives all qualify. Source-specific code (the dev.to fetcher, the markdown parser) does *not* — that lives inside the feature that uses it.
- **The eval harness talks to the live HTTP surface, not the service layer.** That's deliberate (ADR-013): the numbers in `docs/eval-results-prod.txt` reflect end-to-end behaviour including FastAPI middleware, error handlers, and serialization — not just the algorithm.
- **External boxes** are the only out-of-process dependencies: Qdrant Cloud (vector store), OpenRouter (LLM gateway), and a Fly volume-mounted SQLite file (job persistence).
- ADR-001 explains the package-by-feature choice; ADR-011 explains why Qdrant access is wrapped in a Protocol; ADR-013 explains the per-stage degradation that's invisible in this diagram but lives in the `Pipeline` executor.
