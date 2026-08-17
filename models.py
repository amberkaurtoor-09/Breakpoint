"""Pydantic models for the 4-step document pipeline, traces, and diagnosis."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class Entity(BaseModel):
    name: str
    value: str


class IntakeInput(BaseModel):
    raw_text: str


class IntakeOutput(BaseModel):
    cleaned_text: str
    word_count: int
    language: str = "en"
    issues: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=1.0, le=5.0)


class ExtractionOutput(BaseModel):
    entities: list[Entity]
    confidence: float = Field(ge=1.0, le=5.0)


class ClassificationOutput(BaseModel):
    doc_type: str
    confidence: float = Field(ge=1.0, le=5.0)


class SummarizationOutput(BaseModel):
    summary: str
    confidence: float = Field(ge=1.0, le=5.0)


class Span(BaseModel):
    span_id: str
    step_name: str
    input: str
    output: str
    confidence: float | None = None
    latency_ms: float
    error: str | None = None


class PipelineResult(BaseModel):
    trace_id: str
    final_status: Literal["success", "failure", "degraded"]
    steps: list[Span]


class Trace(BaseModel):
    trace_id: str
    timestamp: datetime
    final_status: Literal["success", "failure", "degraded"]
    avg_confidence: float
    spans: list[Span]

    @classmethod
    def from_result(cls, result: PipelineResult) -> Trace:
        confidences = [s.confidence for s in result.steps if s.confidence is not None]
        avg = sum(confidences) / len(confidences) if confidences else 1.0
        return cls(
            trace_id=result.trace_id,
            timestamp=datetime.now(timezone.utc),
            final_status=result.final_status,
            avg_confidence=round(avg, 3),
            spans=result.steps,
        )


class TraceSummary(BaseModel):
    trace_id: str
    timestamp: str
    final_status: str
    avg_confidence: float


class DiagnosisEvidence(BaseModel):
    input: str
    output: str
    confidence: float | None
    checks_failed: list[str] = Field(default_factory=list)


class Diagnosis(BaseModel):
    root_cause_step: str | None
    reason: str
    evidence: DiagnosisEvidence | dict[str, Any] | None = None


class PipelineRunRequest(BaseModel):
    raw_text: str = Field(min_length=1)
