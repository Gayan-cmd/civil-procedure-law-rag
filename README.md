# Civil Procedure Law Assistant

An amendment-aware **retrieval-augmented generation (RAG)** system over Sri Lankan civil procedure statutes. Answers legal questions grounded in cited source legislation — every claim traces back to a specific act, not the model's own (unreliable) memory of the law.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-RAG_orchestration-1C3C3C)
![ChromaDB](https://img.shields.io/badge/ChromaDB-vector_store-FF6F61)
![Colab](https://img.shields.io/badge/Google_Colab-notebook-F9AB00?logo=googlecolab&logoColor=white)

## Overview

Sri Lankan civil procedure law is scattered across a principal code, a dozen separate amendment acts spanning decades, and several related statutes — with no single, structurally consolidated source. This project:

1. **Builds a verified corpus** — downloads primary-source legal PDFs, audits every document for genuine extractable text (rejecting scanned pages rather than silently feeding garbage into the pipeline), and records full provenance in a machine-readable manifest.
2. **Runs a metadata-aware RAG pipeline** — chunks and embeds the corpus, retrieves with citation-preserving metadata (so retrieval can distinguish between different acts that happen to share the same act number), and generates answers that cite their source act by name.

The corpus-build side is fully scripted and reproducible (`build_corpus.py`). The RAG pipeline runs in Google Colab (`Civil_Procedure_Law_Assistant.ipynb`) using a small open-source LLM — no fine-tuning, no paid API.

## Key features

- **Extraction quality auditing, not just extraction.** Every PDF is checked page-by-page (specifically the *middle* page, not just the cover) to catch documents that look born-digital but are actually scans with an OCR'd first page.
- **Metadata-aware retrieval.** Every chunk carries its source act's number, year, and document type. Retrieval can filter to a specific act, and correctly resolves real citation collisions in the corpus (e.g. two different acts both being "Act No. 11", four decades apart).
- **Citation-aware generation.** The prompt requires the model to name its source act for every claim, and to say so explicitly rather than invent a section number when the retrieved text doesn't specify one.
- **Genuine section-level amendment tracing.** The consolidated code carries 345 inline `[section, Act N of YYYY]` citation markers spanning 1977–2024 — found after the first candidate edition turned out to just append amendment acts rather than merge them, and swapped for one that actually does.

## Repository structure

```
.
├── Civil_Procedure_Law_Assistant.ipynb   # RAG pipeline (Google Colab)
├── build_corpus.py                       # corpus acquisition + extraction pipeline
├── build_corpus_batch2.py                # follow-up pass: recovered related statutes
├── build_corpus_batch3.py                # gap-fill pass: court rules, interpretation ordinance, base edition swap
├── scan_annotations.py                   # regex scan for inline amendment annotations
└── corpus/
    ├── manifest.json                     # metadata + extraction audit, one entry per document
    ├── REPORT.md                         # full extraction audit write-up
    ├── amendment_annotation_scan.json    # raw output of the annotation scan
    ├── raw/                              # 18 untouched source files (17 PDFs + 1 HTML)
    └── text/                             # 18 extracted plain-text files
```

## The corpus

| | |
|---|---|
| Documents in corpus | 18 |
| Failed / absent (documented, non-recoverable) | 3 |
| Total extractable characters | 1,349,234 |
| Document types | 1 principal act, 10 amendment acts, 7 related statutes |

**What's included:** the Civil Procedure Code (consolidated to Act 43/2024, section-merged), 10 CPC amendment acts spanning 2005–2024, and 7 related statutes (Judicature Act, Evidence Ordinance, Prescription Ordinance, Arbitration Act, Mediation Boards Act, Supreme Court Rules [incl. Court of Appeal Appellate Procedure Rules], Interpretation Ordinance).

**What's excluded, and why** (all verified, not guessed around):
- **CommonLII's consolidated edition** — the server returns a genuine Apache-level 403 Forbidden, confirmed in a real browser session, not a bot-detection challenge.
- **LawNet's consolidated edition** — `lawnet.gov.lk` serves a TLS certificate for an unrelated domain; treated as untrustworthy rather than bypassed.
- **"Act 50 of 2024"** — turned out to be a labeling error at the source; the real Act No. 50 of 2024 is an unrelated statute (Reciprocal Recognition of Foreign Judgments), not a Civil Procedure Code amendment.
- **Debt Recovery (Special Provisions) Act** — the only PDF found is a genuine scan (0 characters extracted from a 24.5MB file); the one HTML alternative found is a dead link. Documented as an open gap rather than forced through a bad source.

Amendment-chain completeness was independently verified (Sept 2026): cross-checked against the publisher's own complete legislative history and Sri Lanka's Parliament's official 2025 acts list — no CPC amendments are missing.

Full extraction audit, per-document verdicts, and reasoning: [`corpus/REPORT.md`](corpus/REPORT.md).

## How it works

```
21 legal source documents
   │  build_corpus.py + build_corpus_batch2/3.py  (requests + pypdf, page-level extraction audit)
   ▼
18 verified born-digital .txt files + manifest.json
   │  chunk (RecursiveCharacterTextSplitter, 800 chars / 100 overlap)
   │  attach metadata: doc_type, act_number, year, amends
   ▼
Chroma vector store  (sentence-transformers/all-MiniLM-L6-v2 embeddings)
   │  query ──► detect "Act N of YYYY" reference?
   │             ├─ yes → metadata-filtered similarity search
   │             └─ no  → plain similarity search
   ▼
retrieved chunks, tagged with [Source: <act title>]
   │  citation-required prompt template
   ▼
Qwen2.5-Instruct (local, no fine-tuning)
   ▼
answer, with sources cited and listed
```

## Getting started

1. Open `Civil_Procedure_Law_Assistant.ipynb` in [Google Colab](https://colab.research.google.com).
2. Upload `corpus/text/*.txt` and `corpus/manifest.json` to a folder in your Google Drive.
3. Run the notebook cells in order — it mounts Drive, installs dependencies, builds the vector store, loads the LLM, and exposes an `ask()` function.
4. Ask a question:
   ```python
   ask("What did Act No. 43 of 2024 change in the Civil Procedure Code?")
   ```

No local setup, GPU, or paid API key required — everything runs on Colab's free tier (a GPU runtime speeds up generation but isn't required).

## Known limitations

- **Small model, no fine-tuning.** Qwen2.5-Instruct (0.5B/1.5B) is used as-is. Treat answers as a pointer to the likely source act/section — always verify against the cited text, not as authoritative legal advice.
- **Corpus coverage gaps.** CommonLII's and LawNet's editions remain unreachable (access-restricted / bad TLS cert), and the Debt Recovery (Special Provisions) Act has no usable source yet (scan / dead link) — see `corpus/REPORT.md` for full detail. This is not a complete restatement of Sri Lankan civil procedure law.
- **No case law.** Statutes and court rules only — judicial interpretation, which routinely resolves real procedural ambiguity in practice, isn't part of this corpus.
- **The notebook needs re-running against the updated corpus — and its retrieval filter needs a code change, not just new data.** The base CPC edition was swapped for a genuine section-level consolidation, two documents were added, and (after external review) the metadata schema changed: the consolidated code's `act_number`/`year` are now `null` (they used to collide with the real Act 43/2024), and every document now carries `content_role` (`primary_consolidated_text` vs. `amending_instrument`) plus `superseded_by` on the amendment acts, since the consolidated text now contains every amendment merged in and the 10 amendment acts would otherwise surface as confusing duplicates for general queries. The notebook's retrieval filter needs to actually use these fields — see `corpus/REPORT.md` → "Open items" for the exact code change.

## Tech stack

**Corpus pipeline:** Python, `requests`, `pypdf`

**RAG pipeline:** LangChain (`langchain`, `langchain-community`, `langchain-huggingface`), ChromaDB, `sentence-transformers` (`all-MiniLM-L6-v2`), HuggingFace `transformers`, Qwen2.5-Instruct

**Environment:** Google Colab, Google Drive

## Data sources

All source documents are publicly available Sri Lankan legal texts, downloaded from [lankalaw.net](https://lankalaw.net), [parliament.lk](https://www.parliament.lk), [srilankalaw.lk](https://www.srilankalaw.lk), the [Sri Lanka National Arbitration Centre](https://www.slnarbcentre.com), and the [Supreme Court of Sri Lanka](https://supremecourt.lk). Full source URLs and download provenance for every document are recorded in [`corpus/manifest.json`](corpus/manifest.json).
