# Breakpoint — failure forensics for AI pipelines

When a multi-step LLM pipeline fails, the last step is usually the one that *looks* broken. The actual error often entered two steps earlier. Breakpoint traces every step, then walks the trace backward to name the root-cause span — with the quoted input, output, and confidence that justify the call.

It runs a 4-step document pipeline (intake → extraction → classification → summarization) behind FastAPI, stores every run as a JSON trace plus a SQLite index, and exposes a Streamlit explorer so you can paste a document, inspect spans, and diagnose a failure in one screen.

**Correctly diagnosed the root-cause step in 8 / 8 deliberately-broken test runs** (trap keywords aimed at intake, extraction, classification, or summarization). 7 unit tests cover stub variation, SQLite commits, and that evidence quotes real span data rather than a generic “something went wrong.”

## What I designed

- **Step-targeted failures, not cascades.** Trap keywords (`GARBLED`, `REDACTED`, `AMBIGUOUS`, `LOREM IPSUM`) degrade one specific step. Downstream steps still see the original source text, so diagnosis can land on “extraction broke” instead of “everything after intake looks bad.”
- **Backward walk, earliest fault.** `diagnose_failure` walks spans from summarization back to intake and keeps updating the candidate, so the reported root cause is where bad data entered — not a healthy-looking tail or a poisoned last step.
- **Two-layer persistence.** Full span detail lives in `traces/{trace_id}.json` (source of truth). SQLite is only an index (`trace_id`, timestamp, status, avg confidence) and every write `commit()`s explicitly.
- **Same function signature for stub and real LLM.** `fake_llm_call(prompt, task)` tries Anthropic or OpenAI if a key is set, and always falls back to `fallback_llm_call` on missing key, timeout, or parse failure. The demo never depends on an API being up.

## Tradeoff I chose

Rule-based confidence scoring vs. LLM-as-judge: I used deterministic rules first (keyword extraction, cue-word classification, grounding checks, a 3.0 confidence threshold) so the system is free to demo, fast to iterate on, and reproducible in tests. LLM-as-judge is the obvious next step — swap the span checks in `diagnose_failure` for a model call that reads `(input, output)` and returns a structured verdict, without changing the trace format or the UI.

## Run it

Python 3.11+ (developed on 3.13). No API key required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# terminal 1
uvicorn main:app --reload --port 8000

# terminal 2
streamlit run app.py
```

- API docs: http://127.0.0.1:8000/docs
- Explorer: http://localhost:8501

Optional real LLM: copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`.

```bash
pytest -q
```

## How to trigger a known failure

Paste any of these into the explorer (or `POST /pipeline/run`) and hit **Diagnose**:

| Keyword | Fails at |
|---|---|
| `GARBLED` | intake |
| `REDACTED` | extraction |
| `AMBIGUOUS` | classification |
| `LOREM IPSUM` | summarization |

Short text (< 8 words) also degrades intake; under 20 words degrades summarization. About 20% of other inputs get a deterministic hash-based quality drop so the trace list is not all green.

## API

| Method | Path | What it returns |
|---|---|---|
| `POST` | `/pipeline/run` | `PipelineResult` (`trace_id`, `final_status`, spans) |
| `GET` | `/traces` | SQLite summary list |
| `GET` | `/traces/{trace_id}` | Full trace with every span |
| `POST` | `/traces/{trace_id}/diagnose` | `{root_cause_step, reason, evidence}` — or `root_cause_step: null` if the run already succeeded |

## Layout

```
app.py          Streamlit explorer (Breakpoint)
main.py         FastAPI routes
pipeline.py     run_pipeline + diagnose_failure
llm_stub.py     fallback_llm_call + optional real API
models.py       Pydantic models
store.py        JSON traces + SQLite index
```
