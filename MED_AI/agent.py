"""
LangGraph agent – medical report analyser
=========================================
Graph flow:
    input_guard → extract_info → web_search → synthesise → output_guard → END

LLM    : Anthropic Claude  (claude-sonnet-4-5)
Search : Tavily            (key from .env)
Guards : guardrails.py
"""

from __future__ import annotations

import json
import os
from typing import TypedDict, Annotated
import operator

import dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from tavily import TavilyClient

from guardrails import run_input_guardrails, run_output_guardrails

dotenv.load_dotenv()


# ── LLM + search client ───────────────────────────────────────────────────────

llm = ChatAnthropic(
    model="claude-sonnet-4-5",
    temperature=0.3,
    max_tokens=2048,
)

client = TavilyClient(os.getenv("TAVILY_API_KEY"))


# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    raw_text: str                                   
    extracted_info: dict                            
    search_results: Annotated[list, operator.add]   
    final_summary: str                              
    warnings: Annotated[list, operator.add]         #addition


# ── Node 0 – input guardrails ─────────────────────────────────────────────────

def input_guard(state: AgentState) -> dict:
    """
    Runs before extraction:
      • Classifies document as medical or not
      • Redacts PII (phones, emails, Aadhaar, PAN)
    Pipeline always continues regardless of result.
    """
    cleaned_text, warns = run_input_guardrails(state["raw_text"])
    return {"raw_text": cleaned_text, "warnings": warns}


# ── Node 1 – extract structured patient info ──────────────────────────────────

def extract_info(state: AgentState) -> dict:
    system = SystemMessage(content="""You are a medical document parser.
Extract structured information from the given medical document text.
Return a JSON object (no markdown) with these keys (use null if not found):
  patient_name, age, gender, date_of_report, referring_doctor,
  findings, impression, medical_terms, organ_systems_mentioned
""")
    human = HumanMessage(content=f"Document text:\n\n{state['raw_text']}")

    response = llm.invoke([system, human])
    text = response.content.strip()

    # Strip accidental markdown fences
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

    try:
        info = json.loads(text)
    except json.JSONDecodeError:
        info = {"raw_extraction": text}

    return {"extracted_info": info}


# ── Node 2 – web search for context ──────────────────────────────────────────

def web_search(state: AgentState) -> dict:
    info = state["extracted_info"]

    query = (
        info.get("impression") or
        info.get("findings") or
        info.get("raw_extraction") or
        "USG abdominal report findings explained"
    )

    response = client.search(
        query=str(query)[:300],   # cap query length
        search_depth="advanced"
    )

    results = [
        {"content": r.get("content", "")}
        for r in response.get("results", [])[:3]
    ]

    return {"search_results": results}


# ── Node 3 – synthesise final paragraph ──────────────────────────────────────

def synthesise(state: AgentState) -> dict:
    info = state["extracted_info"]
    search_snippets = "\n\n".join(
        f"- {r.get('content', '')[:300]}" for r in state["search_results"]
    )

    system = SystemMessage(content="""You are a senior medical report writer.
Using the extracted patient information AND the web-search context, write a
single, well-structured paragraph (200-350 words) that:
  • Introduces the patient (name, age, gender if available)
  • Describes the key findings of the report in plain English
  • Explains what those findings typically indicate medically (use the web context)
  • Notes any important observations or recommendations implied by the report
Write in a professional yet accessible tone. Do NOT make any definitive diagnosis.
Always end with: "This summary is for informational purposes only and is not a substitute for professional medical advice."
""")

    human = HumanMessage(content=f"""
Extracted patient info (JSON):
{json.dumps(info, indent=2, default=str)}

Web research context:
{search_snippets or 'No additional context found.'}
""")

    response = llm.invoke([system, human])
    return {"final_summary": response.content.strip()}


# ── Node 4 – output guardrails ────────────────────────────────────────────────

def output_guard(state: AgentState) -> dict:
    """
    Runs after synthesis:
      • Flags harmful / dangerous medical advice
      • Ensures disclaimer is present
      • Trims summary if too long
    Pipeline always continues regardless of result.
    """
    final, warns = run_output_guardrails(state["final_summary"])
    return {"final_summary": final, "warnings": warns}


# ── Build and compile the graph ───────────────────────────────────────────────

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("input_guard",  input_guard)
    g.add_node("extract_info", extract_info)
    g.add_node("web_search",   web_search)
    g.add_node("synthesise",   synthesise)
    g.add_node("output_guard", output_guard)

    g.set_entry_point("input_guard")
    g.add_edge("input_guard",  "extract_info")
    g.add_edge("extract_info", "web_search")
    g.add_edge("web_search",   "synthesise")
    g.add_edge("synthesise",   "output_guard")
    g.add_edge("output_guard", END)

    return g.compile()


_graph = build_graph()




def run_agent(raw_text: str) -> tuple[str, list[str]]:
    """
    Run the full LangGraph pipeline.
    Returns (final_summary, warnings).
    """
    result = _graph.invoke({
        "raw_text": raw_text,
        "search_results": [],
        "warnings": [],
    })
    return result["final_summary"], result.get("warnings", [])