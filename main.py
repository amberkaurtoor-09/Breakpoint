"""FastAPI app: run the pipeline, inspect traces, diagnose failures."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import PipelineResult, PipelineRunRequest, Trace, TraceSummary
from pipeline import diagnose_failure, run_pipeline
from store import list_traces, load_trace

app = FastAPI(
    title="Failure Forensics",
    description="Trace and diagnose failures in a 4-step document LLM pipeline.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/pipeline/run", response_model=PipelineResult)
def pipeline_run(body: PipelineRunRequest) -> PipelineResult:
    return run_pipeline(body.raw_text)


@app.get("/traces", response_model=list[TraceSummary])
def traces_index() -> list[TraceSummary]:
    return list_traces()


@app.get("/traces/{trace_id}", response_model=Trace)
def traces_get(trace_id: str) -> Trace:
    trace = load_trace(trace_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")
    return trace


@app.post("/traces/{trace_id}/diagnose")
def traces_diagnose(trace_id: str) -> dict:
    if load_trace(trace_id) is None:
        raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")
    return diagnose_failure(trace_id)
