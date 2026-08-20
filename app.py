"""Streamlit trace explorer. Run separately from FastAPI:

    uvicorn main:app --reload --port 8000
    streamlit run app.py

Theme config lives in .streamlit/config.toml (light, white/green).
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_BASE = os.environ.get("FORENSICS_API", "http://127.0.0.1:8000")

STATUS_COLOR = {
    "success": "#16A34A",
    "degraded": "#D97706",
    "failure": "#DC2626",
}

TRAP_HELP = """
**Trap keywords** (force a known failing step):

| Keyword | Fails at |
|---|---|
| `GARBLED` | intake |
| `REDACTED` | extraction |
| `AMBIGUOUS` | classification |
| `LOREM IPSUM` | summarization |
"""

CUSTOM_CSS = """
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1200px; }

    .bp-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #059669, #16A34A);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .bp-subtitle { color: #4B5563; font-size: 1rem; margin-bottom: 1.5rem; }

    .bp-section { font-size: 1.1rem; font-weight: 700; color: #111827;
                  margin-bottom: 0.6rem; border-bottom: 1px solid #D1FAE5;
                  padding-bottom: 0.4rem; }

    section[data-testid="stSidebar"] { background: #F0FDF4; }
    .bp-sidebar-title { font-size: 0.95rem; font-weight: 700; color: #059669;
                         text-transform: uppercase; letter-spacing: 0.05em;
                         margin: 0.5rem 0 1rem 0; }

    div[data-testid="stSidebar"] button {
        background: #FFFFFF !important;
        border: 1px solid #D1FAE5 !important;
        border-radius: 10px !important;
        color: #111827 !important;
        text-align: left !important;
        font-family: monospace !important;
        font-size: 0.82rem !important;
        padding: 10px 14px !important;
        transition: border-color 0.15s ease;
    }
    div[data-testid="stSidebar"] button:hover {
        border-color: #16A34A !important;
        background: #ECFDF5 !important;
    }

    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #059669, #16A34A) !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        padding: 0.6rem 1.4rem !important;
        color: white !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(90deg, #047857, #15803D) !important;
    }

    textarea { border-radius: 10px !important; border: 1px solid #D1FAE5 !important; }

    .bp-card {
        background: #FFFFFF;
        border: 1px solid #D1FAE5;
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }

    .bp-empty {
        background: #F9FAFB;
        border: 1px dashed #D1D5DB;
        border-radius: 14px;
        padding: 2.5rem 1.5rem;
        text-align: center;
        color: #6B7280;
    }
</style>
"""


WAKE_UP_MESSAGE = "Backend is waking up, please wait 30 seconds and refresh."


def api_get(path: str):
    response = requests.get(f"{API_BASE}{path}", timeout=15)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict | None = None):
    response = requests.post(f"{API_BASE}{path}", json=payload or {}, timeout=30)
    response.raise_for_status()
    return response.json()


def show_api_error(exc: BaseException, fallback: str) -> None:
    if isinstance(exc, requests.Timeout) or "timed out" in str(exc).lower():
        st.warning(WAKE_UP_MESSAGE)
        return
    st.error(fallback)


def status_badge(status: str) -> str:
    color = STATUS_COLOR.get(status, "#666")
    return (
        f"<span style='background:{color}18;color:{color};padding:3px 10px;"
        f"border-radius:999px;font-size:0.8rem;font-weight:700;border:1px solid {color}55'>"
        f"{status.upper()}</span>"
    )


st.set_page_config(page_title="Breakpoint", layout="wide", page_icon="🟢")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown('<div class="bp-title">Breakpoint</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="bp-subtitle">Run a 4-step document pipeline, inspect every trace, '
    'find exactly where it broke.</div>',
    unsafe_allow_html=True,
)

try:
    traces = api_get("/traces")
    api_ok = True
except requests.RequestException as exc:
    traces = []
    api_ok = False
    show_api_error(
        exc,
        f"Cannot reach FastAPI at `{API_BASE}`. "
        f"Start it with `uvicorn main:app --reload --port 8000`.",
    )

with st.sidebar:
    st.markdown(f'<div class="bp-sidebar-title">Past traces ({len(traces)})</div>', unsafe_allow_html=True)
    if not traces:
        st.caption("No traces yet." if api_ok else "Backend offline.")

    for item in traces[:12]:
        status = item["final_status"]
        icon = {"success": "✓", "degraded": "◐", "failure": "✕"}.get(status, "•")
        label = f"{icon}  {item['trace_id'][:8]}   ·   conf {item['avg_confidence']:.2f}"
        if st.button(label, key=f"trace-{item['trace_id']}", use_container_width=True):
            st.session_state["selected_trace"] = item["trace_id"]

    if len(traces) > 12:
        st.caption(f"+{len(traces) - 12} more not shown")

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="bp-section">Run pipeline</div>', unsafe_allow_html=True)
    raw_text = st.text_area(
        "Paste a document",
        height=200,
        placeholder="Invoice 1042 to Acme Corp, amount due $1,240.00 on March 3, 2026...",
        label_visibility="collapsed",
    )
    with st.expander("How to trigger a known failure"):
        st.markdown(TRAP_HELP)
    if st.button("Run Pipeline", type="primary", disabled=not api_ok):
        if not raw_text.strip():
            st.warning("Paste some text first.")
        else:
            try:
                result = api_post("/pipeline/run", {"raw_text": raw_text})
                st.session_state["selected_trace"] = result["trace_id"]
                st.rerun()
            except requests.RequestException as exc:
                show_api_error(exc, "Pipeline request failed.")

with right:
    st.markdown('<div class="bp-section">Trace detail</div>', unsafe_allow_html=True)
    selected_id = st.session_state.get("selected_trace")
    if not selected_id:
        st.markdown(
            '<div class="bp-empty">Run a pipeline, or pick a trace from the sidebar,<br>'
            'to see its step-by-step breakdown here.</div>',
            unsafe_allow_html=True,
        )
    else:
        try:
            trace = api_get(f"/traces/{selected_id}")
        except requests.RequestException as exc:
            show_api_error(exc, "Could not load trace.")
            trace = None

        if trace:
            st.markdown(
                f'<div class="bp-card">'
                f"<code style='color:#059669'>{trace['trace_id']}</code> &nbsp; "
                f"{status_badge(trace['final_status'])} &nbsp; "
                f"<span style='color:#6B7280'>avg confidence</span> "
                f"<b>{trace['avg_confidence']:.2f}</b>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.caption(trace.get("timestamp", ""))

            for span in trace.get("spans", []):
                conf = span.get("confidence")
                conf_label = f"{conf:.2f}" if isinstance(conf, (int, float)) else "—"
                with st.expander(
                    f"{span['step_name']}   ·   confidence {conf_label}   ·   {span['latency_ms']:.1f} ms",
                    expanded=True,
                ):
                    if span.get("error"):
                        st.error(span["error"])
                    st.markdown("**Input**")
                    st.code(span.get("input") or "", language="json")
                    st.markdown("**Output**")
                    st.code(span.get("output") or "", language="json")

            if st.button("Diagnose", type="primary"):
                try:
                    st.session_state["diagnosis"] = {
                        "trace_id": selected_id,
                        "payload": api_post(f"/traces/{selected_id}/diagnose"),
                    }
                except requests.RequestException as exc:
                    show_api_error(exc, "Diagnose failed.")

            stored = st.session_state.get("diagnosis") or {}
            if stored.get("trace_id") == selected_id:
                diagnosis = stored.get("payload") or {}
                st.markdown('<div class="bp-section">Root-cause diagnosis</div>', unsafe_allow_html=True)
                step = diagnosis.get("root_cause_step")
                if step:
                    st.error(f"Root cause step: **{step}**")
                else:
                    st.success(diagnosis.get("reason", "No failure detected"))
                st.write(diagnosis.get("reason", ""))
                evidence = diagnosis.get("evidence") or {}
                if evidence:
                    st.markdown("**Evidence**")
                    st.json(
                        {
                            "confidence": evidence.get("confidence"),
                            "checks_failed": evidence.get("checks_failed"),
                            "input": evidence.get("input"),
                            "output": evidence.get("output"),
                        }
                    )
                    st.caption("Quoted from the failing span — not a generic error string.")
