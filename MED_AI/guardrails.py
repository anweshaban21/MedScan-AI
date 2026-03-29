"""
guardrails.py – soft-warning guardrails for MedScan AI
=======================================================

INPUT guardrails  (run before the agent pipeline)
  1. is_medical_document()  – reject non-medical text (LLM-based)
  2. redact_pii()           – detect & redact name / phone / address

OUTPUT guardrails  (run after synthesis)
  3. check_harmful_advice() – flag dangerous medical recommendations (LLM-based)
  4. ensure_disclaimer()    – append disclaimer if missing
  5. cap_length()           – trim to max_words if the summary is too long

All functions return a GuardrailResult so the caller decides what to show.
Violations are WARNINGS only — the pipeline continues regardless.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

# Shared LLM instance (cheap / fast — same model as agent)
_llm = ChatAnthropic(model="claude-haiku-4-5", temperature=0, max_tokens=256)


# ── Result container ──────────────────────────────────────────────────────────

@dataclass
class GuardrailResult:
    passed: bool                        # True = no issue found
    warnings: list[str] = field(default_factory=list)
    modified_text: str = ""             # redacted / trimmed text (if changed)


# ─────────────────────────────────────────────────────────────────────────────
# INPUT GUARDRAIL 1 — Medical document classifier
# ─────────────────────────────────────────────────────────────────────────────

def is_medical_document(text: str) -> GuardrailResult:
    """
    Ask Claude Haiku whether the text looks like a medical document.
    Returns passed=True if it IS medical, passed=False with a warning if not.
    """
    system = SystemMessage(content="""You are a document classifier.
Respond with exactly one word — YES or NO.
YES  → the text is a medical document (USG report, prescription, lab report,
       discharge summary, radiology report, clinical notes, etc.)
NO   → the text is something else entirely (news article, recipe, story, etc.)
""")
    human = HumanMessage(content=f"Document (first 800 chars):\n\n{text[:800]}")

    try:
        resp = _llm.invoke([system, human])
        answer = resp.content.strip().upper()
    except Exception:
        # If the LLM call fails, allow through (fail-open)
        return GuardrailResult(passed=True, warnings=["⚠️ Medical-document check could not run — proceeding anyway."])

    if answer.startswith("NO"):
        return GuardrailResult(
            passed=False,
            warnings=["⚠️ This document doesn't appear to be a medical report or prescription. "
                      "Results may be inaccurate — please upload a valid medical document."],
            modified_text=text,
        )
    return GuardrailResult(passed=True, modified_text=text)


# ─────────────────────────────────────────────────────────────────────────────
# INPUT GUARDRAIL 2 — PII redaction
# ─────────────────────────────────────────────────────────────────────────────

# Patterns for common PII
_PHONE_RE  = re.compile(r"(\+?\d[\d\s\-().]{7,}\d)")
_EMAIL_RE  = re.compile(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}")
_AADHAR_RE = re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")   # Indian Aadhaar
_PAN_RE    = re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")             # Indian PAN

def redact_pii(text: str) -> GuardrailResult:
    """
    Redact phone numbers, emails, Aadhaar, and PAN from the raw text.
    Patient name is left in place (needed for the summary) but a warning
    is shown to inform the user that names are present.
    """
    warnings: list[str] = []
    redacted = text

    # Phone
    phones = _PHONE_RE.findall(redacted)
    if phones:
        redacted = _PHONE_RE.sub("[PHONE REDACTED]", redacted)
        warnings.append(f"⚠️ {len(phones)} phone number(s) detected and redacted before processing.")

    # Email
    emails = _EMAIL_RE.findall(redacted)
    if emails:
        redacted = _EMAIL_RE.sub("[EMAIL REDACTED]", redacted)
        warnings.append(f"⚠️ {len(emails)} email address(es) detected and redacted before processing.")

    # Aadhaar
    if _AADHAR_RE.search(redacted):
        redacted = _AADHAR_RE.sub("[AADHAAR REDACTED]", redacted)
        warnings.append("⚠️ Aadhaar number detected and redacted before processing.")

    # PAN
    if _PAN_RE.search(redacted):
        redacted = _PAN_RE.sub("[PAN REDACTED]", redacted)
        warnings.append("⚠️ PAN number detected and redacted before processing.")

    # Soft name notice (we keep the name — agent needs it)
    if not warnings:
        # No sensitive IDs found — still note name may be present
        warnings.append("ℹ️ Patient name (if present) is retained for the summary but not stored.")

    passed = redacted == text or bool(warnings)   # always soft-pass
    return GuardrailResult(passed=True, warnings=warnings, modified_text=redacted)


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT GUARDRAIL 3 — Harmful / dangerous advice detector
# ─────────────────────────────────────────────────────────────────────────────

_HARMFUL_KEYWORDS = [
    "take", "prescribe", "inject", "consume", "administer", "stop taking",
    "discontinue", "overdose", "double the dose", "increase dose",
    "self-medicate", "do not go to", "avoid hospital",
]

def check_harmful_advice(summary: str) -> GuardrailResult:
    """
    Two-layer check:
      1. Fast keyword scan
      2. If keywords found, ask LLM to confirm whether it's actually harmful
    """
    lowered = summary.lower()
    keyword_hit = any(kw in lowered for kw in _HARMFUL_KEYWORDS)

    if not keyword_hit:
        return GuardrailResult(passed=True, modified_text=summary)

    # LLM confirmation pass
    system = SystemMessage(content="""You are a medical safety reviewer.
Read the summary below and answer with exactly YES or NO.
YES → the summary contains specific harmful medical advice
      (e.g. recommends a drug dose, tells patient to self-medicate,
       discourages seeking medical care, makes a definitive diagnosis).
NO  → the summary is safe and appropriately cautious.
""")
    human = HumanMessage(content=summary)

    try:
        resp = _llm.invoke([system, human])
        answer = resp.content.strip().upper()
    except Exception:
        return GuardrailResult(passed=True, warnings=["⚠️ Harmful-advice check could not run."], modified_text=summary)

    if answer.startswith("YES"):
        return GuardrailResult(
            passed=False,
            warnings=["⚠️ The generated summary may contain specific medical recommendations. "
                      "Please review carefully and consult a qualified medical professional."],
            modified_text=summary,
        )
    return GuardrailResult(passed=True, modified_text=summary)


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT GUARDRAIL 4 — Disclaimer enforcement
# ─────────────────────────────────────────────────────────────────────────────

DISCLAIMER = (
    "This summary is for informational purposes only and is not a substitute "
    "for professional medical advice."
)

def ensure_disclaimer(summary: str) -> GuardrailResult:
    """Append the standard disclaimer if it's missing from the summary."""
    if "informational purposes only" in summary.lower():
        return GuardrailResult(passed=True, modified_text=summary)

    fixed = summary.rstrip() + f"\n\n{DISCLAIMER}"
    return GuardrailResult(
        passed=False,
        warnings=["ℹ️ Standard medical disclaimer was missing and has been automatically appended."],
        modified_text=fixed,
    )


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT GUARDRAIL 5 — Length cap (anti-hallucination sprawl)
# ─────────────────────────────────────────────────────────────────────────────

MAX_WORDS = 400

def cap_length(summary: str, max_words: int = MAX_WORDS) -> GuardrailResult:
    """Trim the summary to max_words if it exceeds the limit."""
    words = summary.split()
    if len(words) <= max_words:
        return GuardrailResult(passed=True, modified_text=summary)

    trimmed = " ".join(words[:max_words]) + "…"
    # Re-append disclaimer if it got cut off
    if "informational purposes only" not in trimmed.lower():
        trimmed += f"\n\n{DISCLAIMER}"

    return GuardrailResult(
        passed=False,
        warnings=[f"⚠️ Summary exceeded {max_words} words and was trimmed to reduce risk of hallucination."],
        modified_text=trimmed,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience runners (used by agent.py nodes)
# ─────────────────────────────────────────────────────────────────────────────

def run_input_guardrails(text: str) -> tuple[str, list[str]]:
    """
    Run all input guardrails in order.
    Returns (possibly_redacted_text, list_of_warning_strings).
    """
    all_warnings: list[str] = []

    # 1. Medical document check
    r1 = is_medical_document(text)
    all_warnings.extend(r1.warnings)

    # 2. PII redaction (always run regardless of doc-type result)
    r2 = redact_pii(text)
    all_warnings.extend(r2.warnings)
    text = r2.modified_text or text   # use redacted version going forward

    return text, all_warnings


def run_output_guardrails(summary: str) -> tuple[str, list[str]]:
    """
    Run all output guardrails in order.
    Returns (final_summary, list_of_warning_strings).
    """
    all_warnings: list[str] = []

    # 3. Harmful advice check
    r3 = check_harmful_advice(summary)
    all_warnings.extend(r3.warnings)
    summary = r3.modified_text or summary

    # 4. Disclaimer enforcement
    r4 = ensure_disclaimer(summary)
    all_warnings.extend(r4.warnings)
    summary = r4.modified_text or summary

    # 5. Length cap
    r5 = cap_length(summary)
    all_warnings.extend(r5.warnings)
    summary = r5.modified_text or summary

    return summary, all_warnings