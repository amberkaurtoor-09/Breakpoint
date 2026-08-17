"""Simulated LLM plus optional real-API swap.

`fake_llm_call(prompt, task)` is the function the pipeline imports. It tries a
real provider if ANTHROPIC_API_KEY or OPENAI_API_KEY is set, and always falls
back to `fallback_llm_call` on missing key, timeout, or parse failure.

`fallback_llm_call` is deliberately rule-based and *varied*:
  - Different document types, entities, and summaries for different inputs
  - Self-reported confidence on a 1–5 scale, not a constant
  - Named trap keywords that fail a *specific* pipeline step (~plus a
    deterministic ~20% extra failure rate from a content hash)

Trap keywords (paste these to force a known failure):
  GARBLED              → intake confidence collapse / empty cleaned text
  REDACTED             → extraction returns ungrounded / empty entities
  AMBIGUOUS            → classification returns a mismatched doc_type
  LOREM IPSUM          → summarization returns a short generic stub
  Short text (< 8 words)  → intake degrades
  Short text (< 20 words) → summarization degrades
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

TASKS = ("intake", "extraction", "classification", "summarization")

# Step-targeted traps so RCA can land on a specific span, not a cascade.
TRAP_INTAKE = "GARBLED"
TRAP_EXTRACTION = "REDACTED"
TRAP_CLASSIFICATION = "AMBIGUOUS"
TRAP_SUMMARIZATION = "LOREM IPSUM"

DOC_TYPE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "invoice": ("invoice", "amount due", "bill to", "qty", "subtotal", "remit"),
    "contract": ("agreement", "hereby", "party", "whereas", "term of", "signature"),
    "medical_record": ("patient", "diagnosis", "prescription", "mg", "blood pressure", "chart"),
    "resume": ("experience", "education", "skills", "bachelor", "linkedin", "objective"),
    "email": ("from:", "to:", "subject:", "dear", "regards", "cc:"),
    "legal_notice": ("hereby notified", "pursuant", "statute", "court", "plaintiff", "defendant"),
    "news_article": ("reported", "according to", "yesterday", "sources", "breaking"),
}

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+\w")
MONEY_RE = re.compile(r"\$[\d,]+(?:\.\d{2})?")
DATE_RE = re.compile(
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
    r"|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b",
    re.I,
)
PHONE_RE = re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")
PROPER_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")


def _digest(prompt: str, task: str) -> int:
    """Stable integer from content — same input always yields the same stub output."""
    return int(hashlib.md5(f"{task}::{prompt}".encode("utf-8")).hexdigest(), 16)


def _clip_confidence(value: float) -> float:
    return round(min(5.0, max(1.0, value)), 2)


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


def _contains(text: str, needle: str) -> bool:
    return needle.lower() in text.lower()


def _forced_fail_step(prompt: str) -> str | None:
    """Return the step that should be degraded for this prompt, or None."""
    if _contains(prompt, TRAP_INTAKE):
        return "intake"
    if _contains(prompt, TRAP_EXTRACTION):
        return "extraction"
    if _contains(prompt, TRAP_CLASSIFICATION):
        return "classification"
    if _contains(prompt, TRAP_SUMMARIZATION):
        return "summarization"
    words = _word_count(prompt)
    if words < 8:
        return "intake"
    if words < 20:
        return "summarization"
    # Deterministic extra ~20% (hash % 5 == 0) aimed at one step.
    if _digest(prompt, "fail-gate") % 5 == 0:
        return TASKS[_digest(prompt, "fail-step") % len(TASKS)]
    return None


def _detect_doc_type(text: str) -> tuple[str, float]:
    lowered = text.lower()
    scores: list[tuple[int, str]] = []
    for doc_type, keywords in DOC_TYPE_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw in lowered)
        if hits:
            scores.append((hits, doc_type))
    if not scores:
        # Spread "unknown" confidence so it isn't a constant.
        jitter = (_digest(text, "unknown-type") % 12) / 10.0
        return "unknown", _clip_confidence(2.4 + jitter)
    scores.sort(reverse=True)
    top_hits, top_type = scores[0]
    runner_up = scores[1][0] if len(scores) > 1 else 0
    confidence = 3.2 + min(top_hits, 3) * 0.5 - runner_up * 0.3
    confidence += (_digest(text, "class-jitter") % 8) / 10.0
    return top_type, _clip_confidence(confidence)


def _extract_entities(text: str) -> list[dict[str, str]]:
    entities: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(name: str, value: str) -> None:
        key = (name, value)
        if value and key not in seen:
            seen.add(key)
            entities.append({"name": name, "value": value})

    for match in EMAIL_RE.findall(text):
        add("email", match)
    for match in MONEY_RE.findall(text):
        add("amount", match)
    for match in DATE_RE.findall(text):
        add("date", match)
    for match in PHONE_RE.findall(text):
        add("phone", match)
    for match in PROPER_RE.findall(text):
        if match.lower() not in {"the", "this", "that"} and len(match.split()) <= 4:
            add("person_or_org", match)

    return entities[:12]


def _summarize(text: str, doc_type: str | None = None) -> str:
    words = re.findall(r"\b\w+\b", text)
    if not words:
        return ""
    lead = " ".join(words[:18])
    kind = doc_type or "document"
    extras = []
    emails = EMAIL_RE.findall(text)
    amounts = MONEY_RE.findall(text)
    if emails:
        extras.append(f"Contact {emails[0]}.")
    if amounts:
        extras.append(f"Mentions {amounts[0]}.")
    extra = " " + " ".join(extras) if extras else ""
    return f"This {kind} opens with: {lead}.{extra}"


def _intake(prompt: str, fail: bool) -> dict[str, Any]:
    cleaned = re.sub(r"\s+", " ", prompt).strip()
    issues: list[str] = []
    words = _word_count(cleaned)
    confidence = 4.2
    if words < 40:
        confidence -= 0.4
    if words < 20:
        confidence -= 0.6
        issues.append("short_document")
    # Slight per-document jitter so scores aren't identical across similar lengths.
    confidence += (_digest(prompt, "intake-jitter") % 7) / 10.0

    if fail:
        issues.append("intake_trap_or_hash_failure")
        if _contains(prompt, TRAP_INTAKE):
            cleaned = ""
            confidence = 1.2
            issues.append("garbled_unreadable")
        else:
            cleaned = cleaned[: max(8, len(cleaned) // 6)]
            confidence = 1.0 + (_digest(prompt, "intake-fail") % 10) / 10.0

    language = "en"
    if re.search(r"[àâçéèêëîïôùûüÿñ]", prompt, re.I):
        language = "fr-or-es-loanwords"
        confidence = min(confidence, 3.4)

    return {
        "cleaned_text": cleaned,
        "word_count": _word_count(cleaned),
        "language": language,
        "issues": issues,
        "confidence": _clip_confidence(confidence),
    }


def _extraction(prompt: str, fail: bool) -> dict[str, Any]:
    entities = _extract_entities(prompt)
    confidence = 3.1 + min(len(entities), 6) * 0.25
    confidence += (_digest(prompt, "extract-jitter") % 5) / 10.0
    if not entities:
        confidence = min(confidence, 2.6)

    if fail:
        if _contains(prompt, TRAP_EXTRACTION):
            entities = [{"name": "content", "value": "[REDACTED]"}]
            confidence = 1.5
        else:
            entities = [{"name": "guess", "value": "UNKNOWN_ENTITY"}]
            confidence = 1.0 + (_digest(prompt, "extract-fail") % 12) / 10.0

    return {"entities": entities, "confidence": _clip_confidence(confidence)}


def _classification(prompt: str, fail: bool) -> dict[str, Any]:
    doc_type, confidence = _detect_doc_type(prompt)
    if fail:
        if _contains(prompt, TRAP_CLASSIFICATION):
            # Intentionally wrong type, not just a low score.
            doc_type = "invoice" if doc_type != "invoice" else "resume"
            confidence = 1.8
        else:
            doc_type = "unknown"
            confidence = 1.1 + (_digest(prompt, "class-fail") % 10) / 10.0
    return {"doc_type": doc_type, "confidence": _clip_confidence(confidence)}


def _summarization(prompt: str, fail: bool) -> dict[str, Any]:
    doc_type, _ = _detect_doc_type(prompt)
    summary = _summarize(prompt, doc_type)
    confidence = 3.5 + min(_word_count(prompt), 80) / 80.0
    confidence += (_digest(prompt, "sum-jitter") % 6) / 10.0

    if fail:
        if _contains(prompt, TRAP_SUMMARIZATION):
            # Must not share tokens with the source, or the grounding check will pass.
            summary = "NO_SUMMARY_AVAILABLE"
            confidence = 1.4
        else:
            summary = "..."
            confidence = 1.0 + (_digest(prompt, "sum-fail") % 11) / 10.0

    return {"summary": summary, "confidence": _clip_confidence(summary and confidence or 1.0)}


def fallback_llm_call(prompt: str, task: str) -> dict[str, Any]:
    """Rule-based stand-in for an LLM. No network. Deterministic for a given prompt+task."""
    if task not in TASKS:
        raise ValueError(f"Unknown task {task!r}; expected one of {TASKS}")
    fail = _forced_fail_step(prompt) == task
    if task == "intake":
        return _intake(prompt, fail)
    if task == "extraction":
        return _extraction(prompt, fail)
    if task == "classification":
        return _classification(prompt, fail)
    return _summarization(prompt, fail)


def _task_instructions(task: str) -> str:
    if task == "intake":
        return (
            "Normalize the document. Return JSON with keys: "
            "cleaned_text (string), word_count (int), language (string), "
            "issues (list of strings), confidence (float 1-5)."
        )
    if task == "extraction":
        return (
            "Extract named entities. Return JSON with keys: "
            "entities (list of {name, value}), confidence (float 1-5)."
        )
    if task == "classification":
        return (
            "Classify the document type. Return JSON with keys: "
            "doc_type (string), confidence (float 1-5)."
        )
    return (
        "Summarize the document in 1-3 sentences. Return JSON with keys: "
        "summary (string), confidence (float 1-5)."
    )


def _parse_llm_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    return json.loads(text)


def _call_anthropic(prompt: str, task: str, api_key: str) -> dict[str, Any]:
    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": os.environ.get("ANTHROPIC_MODEL", "claude-3-5-haiku-latest"),
            "max_tokens": 800,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"{_task_instructions(task)}\n\n"
                        "Respond with JSON only.\n\n"
                        f"Document:\n{prompt}"
                    ),
                }
            ],
        },
        timeout=20,
    )
    response.raise_for_status()
    body = response.json()
    content = body["content"][0]["text"]
    return _parse_llm_json(content)


def _call_openai(prompt: str, task: str, api_key: str) -> dict[str, Any]:
    response = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "content-type": "application/json",
        },
        json={
            "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": _task_instructions(task)},
                {"role": "user", "content": prompt},
            ],
        },
        timeout=20,
    )
    response.raise_for_status()
    body = response.json()
    content = body["choices"][0]["message"]["content"]
    return _parse_llm_json(content)


def fake_llm_call(prompt: str, task: str) -> dict[str, Any]:
    """Pipeline entry point. Real API if a key exists, otherwise the rule-based stub."""
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    try:
        if anthropic_key:
            return _call_anthropic(prompt, task, anthropic_key)
        if openai_key:
            return _call_openai(prompt, task, openai_key)
    except (requests.RequestException, KeyError, IndexError, json.JSONDecodeError, ValueError):
        return fallback_llm_call(prompt, task)
    return fallback_llm_call(prompt, task)
