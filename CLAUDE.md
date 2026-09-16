# Project: Civil Procedure Law Assistant

## What this is

A production RAG system over Sri Lankan civil procedure law. Users sign up,
chat with the system, and their conversations persist. Built from a working
Colab notebook that is being extracted into a proper application.

`BUILD_PLAN.md` in the repo root is the authoritative spec. Read it before
doing anything.

## Who you are working with

A final-year Computer Science undergraduate who is building this to learn,
not just to ship. He has written Python, FastAPI, and React before, but has
never built a production system with authentication, CI, or a deployment
pipeline. Treat every infrastructure concept as new.

He will be asked about this code in job interviews. If he cannot explain a
piece of it, that piece is a liability regardless of how well it works.

## How you must work

### One sub-phase at a time. Always stop.

Never build a whole phase in one go. The loop is:

1. Read the relevant phase in `BUILD_PLAN.md`
2. Break it into sub-phases, each 1–3 hours of work
3. Show the breakdown and **stop** — wait for approval
4. Build **one** sub-phase
5. Explain what you built (format below)
6. **Stop** and wait before starting the next sub-phase

Do not chain sub-phases together, even if the next one seems trivial. Do not
say "I'll continue with the next step" and keep going. Stop means stop.

### Explain like he is learning, because he is

After each sub-phase, write an explanation with these sections:

**What I built** — plain English, no jargon. What can the system do now that
it could not do ten minutes ago?

**Files changed** — a short list with one line each on why that file exists.

**New concepts** — anything he has not met before, explained from zero with
an analogy where one helps. If you used a decorator, a context manager, a
protocol, a migration, a fixture — explain it. Assume no prior exposure.

**Why this way** — the decision you made and the alternative you rejected,
with the reason. This is what he will be asked in interviews.

**How to verify** — the exact command to run, and what he should see.

**What is next** — one sentence on the next sub-phase. Then stop.

Keep the explanation readable. Do not paste large code blocks into it — he
can read the files. Explain the shape and the reasoning instead.

### Ask before assuming

Ask before: adding a dependency, changing the database schema, modifying
`manifest.json` or anything in `corpus/`, or deviating from `BUILD_PLAN.md`.

If `BUILD_PLAN.md` is ambiguous, say so and propose an option rather than
silently choosing.

### Code standards

- Type hints on every function signature
- Tests alongside the code, in the same sub-phase, not deferred
- No hardcoded paths or secrets — everything through `config.py` and env vars
- `.env` in `.gitignore` from the very first commit
- `src/cpa/` must never import FastAPI. The package is callable from a plain
  script, the ingestion job, and the eval harness. HTTP lives only in `api/`.
- One git commit per sub-phase, with a message explaining what and why

### Do not build

Kubernetes, microservices, message queues, Redis caching, multi-tenancy,
admin dashboards. If you think one is genuinely needed, argue for it first.

### Honesty

If something does not work, say so plainly. If a test is failing, do not
disable it. If you are unsure a design is right, say that too — a flagged
uncertainty is far more useful than false confidence.
