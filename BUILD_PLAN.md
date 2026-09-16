# Civil Procedure Law Assistant — Production Build Plan

**From notebook to deployed application.**

---

## Where you are now

| | |
|---|---|
| Corpus documents | 18 extracted successfully (3 failed, all correctly abandoned) |
| Total characters | 1,349,234 |
| Estimated chunks | ~1,900 at 800 chars / 100 overlap |
| Extraction quality | All born-digital — no OCR needed |
| Amendment chain | Complete: Acts 4/2005, 11/2010, 8/2017, 5/2022, 17/2022, 36/2022, 7/2023, 20/2023, 29/2023, 43/2024 |
| Current state | Single Colab notebook, 44 cells, doing ingestion + retrieval + generation together |

The corpus is finished. Nothing below requires collecting more documents.

---

## The shape of what you're building

Four pieces, built in this order:

1. **A Python package** — retrieval logic extracted from the notebook into proper files
2. **An ingestion job** — run occasionally, fills the database with vectors
3. **An API** — runs constantly, answers questions over HTTP
4. **A frontend** — the chat interface users see

Right now one notebook does 1, 2, and 3 mixed together. The first job is separating them.

### Why ingestion and serving must be separate

These are two different kinds of work.

**Ingestion** is slow, runs rarely, and is a *job*. It starts, does heavy work for minutes, writes the vector store, exits. You run it when statutes change.

**Serving** is fast, runs constantly, and is a *service*. It waits for HTTP requests, answers each in seconds, never exits.

If they are one process, every API restart re-embeds the entire corpus before serving anyone — minutes of downtime for a one-line fix. Split, the API starts in seconds because vectors are already in the database.

It also makes the API **stateless**: it holds nothing that cannot be thrown away, so any copy of the container can serve any request.

---

## Phase 0 — Fix the corpus metadata

**Time: half a day. Do this first — everything downstream reads this file.**

### 0.1 Fix the consolidated CPC tagging

Currently `cpc-consolidated-lankalaw` is tagged `act_number: "43", year: 2024`. That is wrong. It is not Act No. 43 of 2024; it is the CPC *as amended up to* that act.

It also collides exactly with the real `cpc-amend-43-2024`, which has identical number and year. Your number-plus-year fix does not separate these — the `doc_type != principal_act` fallback saves you by accident.

```json
{
  "slug": "cpc-consolidated-lankalaw",
  "act_number": null,
  "year": null,
  "consolidated_to": "43/2024"
}
```

### 0.2 Mark supersession status

Add to every amendment act:

```json
{ "status": "incorporated" }
```

Meaning: this act's text is already merged into the consolidated code. This is what lets you handle the duplicate-provision problem deliberately later.

### 0.3 Decide the duplicate policy now

The consolidated CPC already contains all amendments merged in. Indexing the amendment acts as well means each amended provision exists twice: once as final text, once as the instruction that produced it (*"in section 46, substitute the following..."*).

Retrieval will return both. The amending instruction reads as gibberish out of context.

**Recommended policy:** index both, but rank the consolidated code higher by default. Amendment acts surface only when the user cites a specific act by number — you already have the regex that detects this.

Write the policy down in the README. An interviewer will ask.

### 0.4 Verify one data oddity

`interpretation-ordinance` reports 1 page but 21,444 characters. Every other document averages ~2,500 chars/page. Either the page count is wrong or extraction merged pages. Thirty seconds to check.

---

## Phase 1 — Turn the notebook into a package

**Time: 3–5 days.**

### What "a package" means

A notebook is cells you run top to bottom. A package is a folder of `.py` files where each file has one job and you import functions from them.

The difference that matters: **you can test a function, but you cannot test a cell.**

### Target structure

```
civil-procedure-assistant/
  src/
    cpa/
      __init__.py
      config.py         settings, read from environment
      corpus.py         load manifest, build Documents
      chunking.py       splitter logic
      embedding.py      embedding model wrapper
      store.py          vector store read/write
      retrieval.py      search logic + Act regex
      generation.py     LLM calls
      chain.py          retrieval + generation together
  tests/
    test_act_extraction.py
    test_retrieval.py
    test_chunking.py
  ingest/
    run.py              the batch job
  eval/
    gold_set.json
    run_eval.py
  pyproject.toml
  README.md
  .gitignore            put .env in here on day one
```

### The rule that makes this work

**No function reads a file path directly.** Paths come from `config.py`, which reads them from environment variables.

This sounds pedantic. It is the thing that lets identical code run on your laptop, in CI, and in production without editing.

### Write tests as you extract

Start with `extract_act_reference` — pure logic, no dependencies:

```python
def test_extracts_full_citation():
    assert extract_act_reference("Act No. 43 of 2024") == ("43", 2024)

def test_returns_none_on_garbage():
    assert extract_act_reference("hello world") == (None, None)

def test_handles_slash_format():
    assert extract_act_reference("Act 43/2024") == ("43", 2024)
```

The third test will fail against your current regex. Good — that is a real bug, now documented.

### Fix the two known notebook bugs while you are here

1. **Double retrieval.** `retrieve_and_format` runs retrieval inside the chain and discards the documents, so `ask()` retrieves a second time to show sources. Fix with `RunnableParallel` — fan out once, keep both answer and documents.
2. **Prompt leakage patch.** `answer.split("Answer:")[-1]` is a band-aid. Fix the prompt template instead.

### Exit criteria

- `pytest` passes
- `python ingest/run.py` rebuilds the vector store from scratch
- The notebook is deleted or moved to `notebooks/exploration.ipynb`

---

## Phase 2 — Gold set and evaluation harness

**Time: 3–4 days. This is your differentiator. Do not skip or defer it.**

### What a gold set is

A hand-written list of questions where you already know the correct answer, and which document contains it.

```json
{
  "id": "q001",
  "question": "What is the time limit for filing an answer?",
  "expected_doc": "cpc-consolidated-lankalaw",
  "expected_section": "75",
  "category": "basic_lookup",
  "answerable": true
}
```

### Target: 50 questions

| Category | Count | What it tests |
|---|---|---|
| Basic lookup | 20 | Ordinary retrieval |
| Act-number citation | 8 | Your regex + metadata filter path |
| Cross-document | 8 | CPC referencing Evidence Ordinance, Judicature Act |
| Amendment-aware | 6 | Current text vs superseded text |
| Unanswerable | 8 | Refusal instead of fabrication |

Unanswerable questions are about criminal procedure, tax law, anything outside the corpus. These catch hallucination.

**Write these before optimising anything.** You cannot tell whether a change helped without them.

### Metrics

**Recall@5** — of answerable questions, in what fraction did the correct document appear in the top 5 chunks? Your headline retrieval number.

**MRR (Mean Reciprocal Rank)** — right chunk ranked 1st scores 1.0, 2nd scores 0.5, 3rd scores 0.33, averaged. Rewards ranking the right thing higher, not merely including it.

**Refusal accuracy** — of 8 unanswerable questions, how many were refused rather than answered?

**Latency p50 / p95** — median and 95th percentile. Measure retrieval and generation separately; the split shows where your bottleneck is.

### Build it as a script

`eval/run_eval.py` prints a table and **exits nonzero if Recall@5 falls below a threshold you set**. That exit code is what CI will consume in Phase 5.

### Run the ablation

| Config | Recall@5 | MRR | Refusal acc | p95 latency |
|---|---|---|---|---|
| Naive dense, k=5 | | | | |
| + Act-number filtering | | | | |
| + BM25 hybrid (RRF) | | | | |
| + Cross-encoder rerank | | | | |

**This table is the single most valuable artifact in the project for your CV.** It proves you measured rather than assumed.

Hybrid retrieval: combine BM25 keyword search with dense vector search, fused by Reciprocal Rank Fusion (`score = 1/(60 + rank)`). BM25 catches exact rare tokens — section numbers, form names — that embeddings blur together.

Reranking: retrieve 25 cheaply, then score each against the query with a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`, ~90MB) and keep the best 5.

---

## Phase 3 — Move vectors into Postgres

**Time: 2 days.**

### Why change from Chroma

You will run Postgres anyway for users and chat history. One database instead of two means one backup, one connection string, one thing to deploy. `pgvector` is a Postgres extension adding vector similarity search, so metadata filtering becomes plain SQL.

This is a defensible decision you can explain in an interview.

### Schema

```sql
CREATE EXTENSION vector;

CREATE TABLE chunks (
  id          SERIAL PRIMARY KEY,
  doc_slug    TEXT NOT NULL,
  doc_type    TEXT NOT NULL,
  act_number  TEXT,
  year        INTEGER,
  section     TEXT,
  content     TEXT NOT NULL,
  embedding   vector(384)
);

CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);
```

The HNSW index is approximate nearest neighbour — it keeps search fast as the table grows.

### Your Act-number query becomes readable

```sql
SELECT content FROM chunks
WHERE act_number = '43' AND year = 2024
ORDER BY embedding <=> $1
LIMIT 5;
```

### Critical step

**Re-run the eval after migrating.** Numbers should be near-identical. If they shift meaningfully, the port has a bug and you want to find it now, not in Phase 6.

---

## Phase 4 — Build the API

**Time: 4–6 days.**

### What FastAPI gives you

A way to expose Python functions as HTTP endpoints. A request arrives at `/chat`, your function runs, JSON comes back. It also auto-generates interactive documentation at `/docs`.

### Endpoints

```
POST /auth/signup
POST /auth/login
POST /auth/refresh
GET  /conversations
POST /conversations
GET  /conversations/{id}/messages
POST /conversations/{id}/messages    <- the RAG endpoint
GET  /health
```

### Database tables

```
users          id, email, password_hash, created_at
conversations  id, user_id, title, created_at, deleted_at
messages       id, conversation_id, role, content, created_at
citations      id, message_id, doc_slug, section, content
```

Citations in their own table means sources survive a page refresh, and you can later ask which chunks get cited most.

Use **soft delete** (`deleted_at` timestamp) rather than removing rows.

### Auth, briefly

**Password hashing** — never store passwords. Store the output of `argon2` or `bcrypt`, deliberately slow one-way functions. Login re-hashes the submitted password and compares.

**JWT** — a signed token issued at login. The client sends it with every request; the server verifies the signature rather than looking anything up. Use a short-lived access token (15 min) plus a longer refresh token with rotation.

### Streaming

Users should see tokens appear as generated, not wait five seconds for a wall of text. Use Server-Sent Events — one-way server-to-browser, simpler than WebSockets and sufficient here.

### Make the LLM swappable

```python
class Generator(Protocol):
    def generate(self, prompt: str) -> Iterator[str]: ...
```

Implement `GroqGenerator` for production and keep `LocalQwenGenerator` for offline testing.

**Replace Qwen 0.5B in production.** Half a billion parameters cannot reliably follow a refusal instruction. In a legal domain that is a safety problem, not a cosmetic one.

### Domain-specific safety

- A persistent, visible disclaimer: research assistance, not legal advice
- Refuse when retrieval returns nothing above a similarity threshold
- Per-user rate limiting — you are paying per LLM call
- Treat retrieved statute text as untrusted input in principle (prompt injection)

### Observability

**Request IDs.** When a request arrives, generate a random ID and attach it to every log line that request produces:

```
[a3f9] retrieval_complete chunks=5 latency_ms=210
[b7c2] retrieval_complete chunks=0 latency_ms=340
[a3f9] generation_complete latency_ms=2900
[b7c2] generation_error reason=timeout
```

Without this, concurrent users' log lines interleave and you cannot tell whose request failed.

**JSON, not prose.** Same information as structured data:

```json
{"request_id": "b7c2", "event": "retrieval_complete", "chunks": 0, "latency_ms": 340, "user_id": 17}
```

Ugly to read, but now queryable: *"show me every request where chunks was 0"*, *"what is p95 of latency_ms"*. With prose logs you would be writing regex against English.

Log retrieval and generation latency separately. Log which chunks were retrieved per query — this is how you debug quality complaints later.

---

## Phase 5 — Containerise and automate

**Time: 3–4 days.**

### Docker in one paragraph

A container is your app plus everything it needs to run, packaged so it behaves identically everywhere. A `Dockerfile` describes how to build it. The payoff: *"works on my machine"* stops being a sentence anyone says.

Build two: one for the API, one for the ingestion job. Use multi-stage builds (compile dependencies in one stage, copy only results into a slim final image) and run as a non-root user.

`docker-compose.yml` brings up Postgres and the API together locally with one command.

### The CI pipeline

GitHub Actions runs on every push:

```
lint (ruff)
  -> type check (mypy)
    -> unit tests (pytest)
      -> retrieval eval
        -> build image
          -> deploy
```

Each step runs only if the previous passed.

**Lint** — catches unused imports, undefined variables, inconsistent formatting.

**Type check** — you write `def get_chunks(q: str) -> list[Document]`; mypy verifies you never pass an integer where a string was promised.

**Unit tests** — your own, from Phase 1.

**Retrieval eval** — the unusual step, and the reason this project stands out. If you change chunk size and Recall@5 drops below threshold, the build fails and the merge is blocked.

Almost nobody does this. It converts *"I built a RAG system"* into *"I built a RAG system with regression testing on retrieval quality"*.

### Secrets

API keys and database passwords go in GitHub Secrets (Settings → Secrets → Actions) and your platform's secret store. **Never in code.** Bots scan public repos for leaked keys within minutes of a push.

Add `.env` to `.gitignore` on day one — not day thirty, by which point it is in your git history and painful to remove.

Rule of thumb: if you would have to change it before handing the repo to a stranger, it is a secret.

---

## Phase 6 — Deploy

**Time: 2 days.**

| Component | Platform | Note |
|---|---|---|
| Database | Neon or Supabase | Free tier, pgvector supported |
| API | Fly.io or Render | Push a Dockerfile, they run it |
| Frontend | Vercel | Free tier |

**Set a billing alert on day one**, whichever platform you use.

Add a `/health` endpoint that checks database connectivity, so the platform knows when the app is genuinely ready rather than merely running.

### On AWS and Azure

Nothing in this plan uses them. That is deliberate — the platforms above are dramatically simpler, and a badly configured AWS deployment reads worse in an interview than a clean Fly.io one.

The argument for AWS is purely job-market keyword matching. If you want it, the middle path is using it for one narrow thing: S3 for source PDFs and ingestion artifacts. An afternoon's work, genuinely useful, and true to say.

Port the whole API to App Runner or ECS Fargate only *after* the system works end to end.

---

## Phase 7 — Frontend

**Time: 4–6 days. Cut this first if time runs short.**

Next.js with TypeScript, which you already know.

Screens: login, signup, chat with a conversation sidebar. Render citations as expandable cards under each answer showing section and source document. Handle the streaming response.

Resist polishing. A clean plain interface that works is better use of your remaining time than an elaborate one.

---

## What to deliberately not build

Kubernetes. Microservices. A message queue. Redis caching before you have measured that you need it. Multi-tenancy. Admin dashboards.

Each adds complexity an interviewer reads as cargo-culting rather than judgement. *"One API container and a managed database did not justify Kubernetes"* is a better answer than a half-configured cluster.

---

## Timeline

| Phase | Days | Cuttable? |
|---|---|---|
| 0 — Metadata fixes | 0.5 | No |
| 1 — Package + tests | 3–5 | No |
| 2 — Gold set + eval | 3–4 | No |
| 3 — Postgres + pgvector | 2 | Yes (keep Chroma) |
| 4 — API | 4–6 | No |
| 5 — Docker + CI | 3–4 | Partially |
| 6 — Deploy | 2 | No |
| 7 — Frontend | 4–6 | Yes |

**Roughly 5–7 weeks part-time.**

---

## One honest caution

This is a large scope alongside a final year and job hunting. The failure mode is an unfinished app that is weaker on your CV than the finished notebook.

If you have to choose: **Phases 1, 2, and 4 plus deployment give you nearly all the interview value.** The chat UI adds polish but adds less than you would think. The evaluation harness adds more than you would think.

---

## Interview questions this plan prepares you for

1. *"Why did embeddings fail on Act numbers?"* — MiniLM's vocabulary tokenizes citations into common fragments that look near-identical in vector space, so cosine similarity cannot separate them. Regex plus metadata filtering is the right tool, not a bigger embedding model.

2. *"How do you know your retrieval works?"* — the gold set, Recall@5, and the ablation table. This is why Phase 2 is not optional.

3. *"Why Postgres with pgvector rather than a dedicated vector database?"* — you were already running Postgres for users and chats; one database means one backup and one deployment, and metadata filtering in SQL is more maintainable than Chroma's filter syntax. At 1,900 chunks, a dedicated vector database would be premature.

4. *"You have the same provision twice — how do you handle that?"* — the deliberate duplicate policy from Phase 0.3.
