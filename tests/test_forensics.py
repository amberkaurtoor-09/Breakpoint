"""Checks for stub variation, persistence, and trap-keyword root-cause analysis."""

from __future__ import annotations

import sqlite3

from llm_stub import fallback_llm_call
from pipeline import diagnose_failure, run_pipeline
from store import DB_PATH, TRACES_DIR

INVOICE = (
    "Invoice 1042 issued to Acme Corp on March 3, 2026. "
    "Bill to Jane Smith at jane.smith@acme.example. Amount due $1,240.00. "
    "Please remit within 30 days. Qty 4 of widget SKU-9."
)

CONTRACT = (
    "This agreement is entered into by Northwind LLC and Contoso Inc. "
    "The parties hereby agree that the term of this contract is 12 months "
    "beginning April 1, 2026. Signature required from both party representatives."
)

MEDICAL = (
    "Patient Maria Lopez presented with elevated blood pressure. "
    "Diagnosis: hypertension. Prescription: lisinopril 10 mg daily. "
    "Follow up recorded in the chart on 04/12/2026 by Dr. Chen."
)

RESUME = (
    "Jordan Lee. Objective: backend engineer. Experience at Riverview Labs. "
    "Education: Bachelor of Science, University of Oregon. Skills: Python, SQL. "
    "LinkedIn profile available on request."
)

TRAPS = {
    "intake": (
        "Patient Maria Lopez chart arrived GARBLED during the OCR scan at clinic intake. "
        "Diagnosis notes for hypertension are unreadable. Prescription for lisinopril 10 mg "
        "and the blood pressure reading of 150/92 cannot be recovered from this page."
    ),
    "extraction": (
        "Patient Maria Lopez chart marked REDACTED by the clinic. "
        "Diagnosis hypertension. Prescription lisinopril 10 mg. Blood pressure 150/92."
    ),
    "classification": (
        "Patient Maria Lopez presented with elevated blood pressure. "
        "Diagnosis: hypertension. The record is AMBIGUOUS after a system merge. "
        "Prescription: lisinopril 10 mg daily."
    ),
    "summarization": (
        "Invoice 1042 issued to Acme Corp on March 3, 2026. "
        "Bill to Jane Smith. Amount due $1,240.00. "
        "The remainder of this file is LOREM IPSUM filler from the template."
    ),
}


def test_stub_outputs_differ_across_documents():
    invoices = fallback_llm_call(INVOICE, "classification")
    medical = fallback_llm_call(MEDICAL, "classification")
    resume = fallback_llm_call(RESUME, "classification")
    assert invoices["doc_type"] == "invoice"
    assert medical["doc_type"] == "medical_record"
    assert resume["doc_type"] == "resume"

    inv_entities = fallback_llm_call(INVOICE, "extraction")["entities"]
    med_entities = fallback_llm_call(MEDICAL, "extraction")["entities"]
    assert any(e["value"] == "$1,240.00" for e in inv_entities)
    assert any("Maria Lopez" in e["value"] for e in med_entities)

    confidences = {
        fallback_llm_call(INVOICE, "summarization")["confidence"],
        fallback_llm_call(CONTRACT, "summarization")["confidence"],
        fallback_llm_call(MEDICAL, "summarization")["confidence"],
        fallback_llm_call(RESUME, "summarization")["confidence"],
        fallback_llm_call("hi", "summarization")["confidence"],
    }
    assert len(confidences) > 1


def test_stub_is_deterministic():
    a = fallback_llm_call(INVOICE, "extraction")
    b = fallback_llm_call(INVOICE, "extraction")
    assert a == b


def test_traps_degrade_the_targeted_step():
    for step, text in TRAPS.items():
        result = fallback_llm_call(text, step)
        assert result["confidence"] < 3.0, step


def test_pipeline_writes_json_and_commits_sqlite():
    result = run_pipeline(INVOICE)
    json_path = TRACES_DIR / f"{result.trace_id}.json"
    assert json_path.exists()
    assert json_path.stat().st_size > 10

    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute(
            "SELECT trace_id, final_status, avg_confidence FROM traces WHERE trace_id = ?",
            (result.trace_id,),
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    assert row[0] == result.trace_id
    assert row[1] in {"success", "failure", "degraded"}
    assert row[2] > 0


def test_diagnose_successful_trace_is_null_root_cause():
    result = run_pipeline(CONTRACT)
    if result.final_status != "success":
        return
    diagnosis = diagnose_failure(result.trace_id)
    assert diagnosis["root_cause_step"] is None
    assert diagnosis["reason"] == "No failure detected"


def test_trap_diagnosis_names_the_right_step():
    for expected_step, text in TRAPS.items():
        result = run_pipeline(text)
        diagnosis = diagnose_failure(result.trace_id)
        assert result.final_status in {"failure", "degraded"}
        assert diagnosis["root_cause_step"] == expected_step, (
            expected_step,
            diagnosis,
        )
        evidence = diagnosis["evidence"]
        assert evidence["input"]
        assert evidence["output"]
        assert evidence["checks_failed"]
        assert "something went wrong" not in diagnosis["reason"].lower()


def test_evidence_quotes_redacted_output():
    result = run_pipeline(TRAPS["extraction"])
    diagnosis = diagnose_failure(result.trace_id)
    output = diagnosis["evidence"]["output"]
    assert "REDACTED" in output or "[REDACTED]" in output
