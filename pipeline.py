"""4-step document pipeline, tracing, and backward root-cause analysis."""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from llm_stub import DOC_TYPE_KEYWORDS, fake_llm_call
from models import (
    ClassificationOutput,
    Diagnosis,
    DiagnosisEvidence,
    ExtractionOutput,
    IntakeInput,
    IntakeOutput,
    PipelineResult,
    Span,
    SummarizationOutput,
    Trace,
)
from store import load_trace, save_trace

LOW_CONFIDENCE = 3.0
FAILURE_CONFIDENCE = 2.0
SHORT_OUTPUT_CHARS = 8


def _serialize(payload: Any) -> str:
    if hasattr(payload, "model_dump"):
        return json.dumps(payload.model_dump(), ensure_ascii=False)
    if isinstance(payload, (dict, list)):
        return json.dumps(payload, ensure_ascii=False)
    return str(payload)


def _prompt_from_input(step_input: Any) -> str:
    if isinstance(step_input, str):
        return step_input
    if isinstance(step_input, dict):
        return str(
            step_input.get("text")
            or step_input.get("cleaned_text")
            or _serialize(step_input)
        )
    if hasattr(step_input, "raw_text"):
        return step_input.raw_text
    if hasattr(step_input, "cleaned_text"):
        return step_input.cleaned_text
    return _serialize(step_input)


def _parse_confidence(raw: dict[str, Any]) -> float | None:
    value = raw.get("confidence")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _run_step(step_name: str, step_input: Any) -> tuple[Span, Any]:
    span_id = str(uuid.uuid4())
    serialized_input = _serialize(step_input)
    prompt = _prompt_from_input(step_input)

    started = time.perf_counter()
    error: str | None = None
    raw_output: dict[str, Any] | None = None
    try:
        raw_output = fake_llm_call(prompt, step_name)
        if step_name == "intake":
            parsed: Any = IntakeOutput.model_validate(raw_output)
        elif step_name == "extraction":
            parsed = ExtractionOutput.model_validate(raw_output)
        elif step_name == "classification":
            parsed = ClassificationOutput.model_validate(raw_output)
        else:
            parsed = SummarizationOutput.model_validate(raw_output)
        serialized_output = _serialize(parsed)
        confidence = parsed.confidence
    except Exception as exc:  # validation or stub errors become span errors
        error = f"{type(exc).__name__}: {exc}"
        serialized_output = _serialize(raw_output) if raw_output is not None else ""
        confidence = _parse_confidence(raw_output or {})
        parsed = None
    latency_ms = (time.perf_counter() - started) * 1000.0

    span = Span(
        span_id=span_id,
        step_name=step_name,
        input=serialized_input if not isinstance(step_input, str) else json.dumps({"text": step_input}),
        output=serialized_output,
        confidence=confidence,
        latency_ms=round(latency_ms, 3),
        error=error,
    )
    return span, parsed


def _status_from_spans(spans: list[Span]) -> str:
    if any(s.error for s in spans):
        return "failure"
    confidences = [s.confidence for s in spans if s.confidence is not None]
    if not confidences:
        return "failure"
    if any(c < FAILURE_CONFIDENCE for c in confidences):
        return "failure"
    if any(c < LOW_CONFIDENCE for c in confidences):
        return "degraded"
    return "success"


def run_pipeline(raw_text: str) -> PipelineResult:
    """Run intake → extraction → classification → summarization and persist a trace."""
    trace_id = str(uuid.uuid4())
    spans: list[Span] = []

    intake_span, intake = _run_step("intake", IntakeInput(raw_text=raw_text))
    spans.append(intake_span)
    source_text = intake.cleaned_text if intake and intake.cleaned_text else raw_text

    extract_span, extraction = _run_step("extraction", source_text)
    spans.append(extract_span)

    class_payload = {
        "cleaned_text": source_text,
        "entities": extraction.model_dump()["entities"] if extraction else [],
    }
    class_span, classification = _run_step("classification", source_text)
    spans.append(class_span)
    # Keep the richer payload in the span input for later RCA, even though the
    # stub classifies from source_text so a bad extraction does not auto-cascade.
    class_span.input = _serialize(class_payload)

    summary_payload = {
        "cleaned_text": source_text,
        "entities": extraction.model_dump()["entities"] if extraction else [],
        "doc_type": classification.doc_type if classification else "unknown",
    }
    sum_span, _summary = _run_step("summarization", source_text)
    spans.append(sum_span)
    sum_span.input = _serialize(summary_payload)

    result = PipelineResult(
        trace_id=trace_id,
        final_status=_status_from_spans(spans),  # type: ignore[arg-type]
        steps=spans,
    )
    save_trace(Trace.from_result(result))
    return result


def _load_json(blob: str) -> Any:
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        return blob


def _as_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        for key in ("cleaned_text", "raw_text", "text", "summary"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return json.dumps(payload, ensure_ascii=False)
    return str(payload)


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in text.split() if len(t) >= 4}


def _quote(text: str, limit: int = 280) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "…"


def _span_checks(span: Span, original_text: str) -> list[str]:
    """Return names of checks this span failed. Empty list means the span looks healthy."""
    failed: list[str] = []
    output = _load_json(span.output)
    output_text = _as_text(output)

    if span.error:
        failed.append(f"step_error:{span.error}")

    if span.confidence is None:
        failed.append("missing_confidence")
    elif span.confidence < LOW_CONFIDENCE:
        failed.append(
            f"low_confidence:{span.confidence:.2f}<{LOW_CONFIDENCE:.1f}"
        )

    if not output_text or not str(output_text).strip():
        failed.append("empty_output")
    elif len(str(output_text).strip()) < SHORT_OUTPUT_CHARS:
        failed.append(
            f"suspiciously_short_output:{len(str(output_text).strip())}_chars"
        )

    if span.step_name == "extraction" and isinstance(output, dict):
        entities = output.get("entities") or []
        if not entities:
            failed.append("no_entities_extracted")
        else:
            source = original_text.lower()
            ungrounded = []
            for ent in entities:
                value = str(ent.get("value", "")).strip()
                if value and value.lower() not in source:
                    ungrounded.append(value)
            if ungrounded and len(ungrounded) / max(len(entities), 1) >= 0.5:
                quoted = ", ".join(f"'{v}'" for v in ungrounded[:4])
                failed.append(f"ungrounded_entities:{quoted}")

    if span.step_name == "classification" and isinstance(output, dict):
        doc_type = str(output.get("doc_type", "")).strip()
        if not doc_type or doc_type.lower() == "unknown":
            failed.append("unclassified_or_unknown_doc_type")
        else:
            # A classification that names a type whose cue words are absent is suspicious.
            cues = DOC_TYPE_KEYWORDS.get(doc_type, ())
            lowered = original_text.lower()
            if cues and not any(cue in lowered for cue in cues):
                failed.append(
                    f"doc_type_mismatch:'{doc_type}' has no supporting cues in the source text"
                )

    if span.step_name == "summarization":
        overlap = _tokens(output_text) & _tokens(original_text)
        if output_text and _tokens(output_text) and not overlap:
            failed.append(
                f"summary_ungrounded:none of the summary's content words appear in the source "
                f"(summary starts '{_quote(output_text, 80)}')"
            )

    if span.step_name == "intake" and isinstance(output, dict):
        cleaned = str(output.get("cleaned_text") or "")
        if original_text.strip() and not cleaned.strip():
            failed.append("intake_dropped_all_text")
        elif cleaned and _tokens(original_text) and not (_tokens(cleaned) & _tokens(original_text)):
            failed.append("intake_output_does_not_resemble_input")

    return failed


def diagnose_failure(trace_id: str) -> dict[str, Any]:
    """Walk spans backward; the earliest failing step is the root cause.

    Walking backward and keeping the last failure we see lands on the first
    broken step in pipeline order — that is where bad data entered, even if
    later steps also look unhealthy. Successful traces are not diagnosed.
    """
    trace = load_trace(trace_id)
    if trace is None:
        return {
            "root_cause_step": None,
            "reason": f"Trace {trace_id} not found",
            "evidence": None,
        }

    original_text = ""
    if trace.spans:
        intake_in = _load_json(trace.spans[0].input)
        original_text = _as_text(intake_in)

    if trace.final_status == "success":
        return Diagnosis(
            root_cause_step=None,
            reason="No failure detected",
            evidence=None,
        ).model_dump()

    # Walk backward, updating on every failure so `candidate` ends as the
    # earliest failing span (root cause), not a downstream cascade.
    candidate: Span | None = None
    candidate_failed: list[str] = []
    for span in reversed(trace.spans):
        failed = _span_checks(span, original_text)
        if failed:
            candidate = span
            candidate_failed = failed

    if candidate is None:
        return Diagnosis(
            root_cause_step=None,
            reason="No failure detected",
            evidence=None,
        ).model_dump()

    reason = (
        f"Walking backward from summarization, '{candidate.step_name}' is the earliest "
        f"failing step in the error chain. "
        + " ".join(_humanize_check(c, candidate) for c in candidate_failed)
    )
    evidence = DiagnosisEvidence(
        input=_quote(_as_text(_load_json(candidate.input))),
        output=_quote(_as_text(_load_json(candidate.output))),
        confidence=candidate.confidence,
        checks_failed=candidate_failed,
    )
    return Diagnosis(
        root_cause_step=candidate.step_name,
        reason=reason,
        evidence=evidence,
    ).model_dump()


def _humanize_check(check: str, span: Span) -> str:
    if check.startswith("low_confidence:"):
        return (
            f"Self-reported confidence was {span.confidence} (threshold {LOW_CONFIDENCE})."
        )
    if check.startswith("ungrounded_entities:"):
        values = check.split(":", 1)[1]
        return f"Extracted values not found in the source text: {values}."
    if check.startswith("doc_type_mismatch:"):
        return check.split(":", 1)[1] + "."
    if check.startswith("summary_ungrounded:"):
        return check.split(":", 1)[1] + "."
    if check.startswith("suspiciously_short_output:"):
        return f"Output was only {check.split(':')[1].replace('_chars', '')} characters."
    if check == "empty_output":
        return "The step returned an empty output."
    if check == "intake_dropped_all_text":
        return "Intake produced an empty cleaned_text even though the source document was non-empty."
    if check == "no_entities_extracted":
        return "Extraction returned no entities."
    if check == "unclassified_or_unknown_doc_type":
        return "Classification could not assign a document type."
    if check.startswith("step_error:"):
        return f"The step raised {check.split(':', 1)[1]}."
    return check.replace("_", " ") + "."
