"""
LangGraph agent – medical report analyser
=========================================
Graph flow:
    extract_info → web_search → synthesise → END
"""

from __future__ import annotations

import json
from typing import TypedDict, Annotated
import operator

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
import os
import dotenv

dotenv.load_dotenv()

from guardrails import run_input_guardrails, run_output_guardrails
# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    raw_text: str                          # original OCR / file text
    extracted_info: dict                   # structured patient info
    search_results: Annotated[list, operator.add]  # web search hits
    final_summary: str                     # output paragraph


# ── LLM + tool ────────────────────────────────────────────────────────────────

llm = ChatAnthropic(
    model="claude-sonnet-4-5",
    temperature=0.3,
    max_tokens=2048,
)
# To install: pip install tavily-python
from tavily import TavilyClient
client = TavilyClient(os.getenv("TAVILY_API_KEY"))
    

#search_tool = TavilySearchResults(max_results=4)

print(os.getenv("ANTHROPIC_API_KEY"))

# ── Node 0 – input guardrails ─────────────────────────────────────────────────
 
def input_guard(state: AgentState) -> dict:
    """
    Runs before extraction:
      • Classifies document as medical or not
      • Redacts PII (phones, emails, Aadhaar, PAN)
    Returns cleaned text + any warnings. Pipeline always continues.
    """
    cleaned_text, warns = run_input_guardrails(state["raw_text"])
    return {"raw_text": cleaned_text, "warnings": warns}
 
# ── Node 1 – extract structured patient info from raw text ───────────────────

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

    # strip accidental markdown fences
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    try:
        info = json.loads(text)
    except json.JSONDecodeError:
        info = {"raw_extraction": text}

    return {"extracted_info": info}


# ── Node 2 – web search for context ──────────────────────────────────────────

def web_search(state: AgentState) -> dict:
    info = state["extracted_info"]
    response = client.search(
    query=state["extracted_info"].get("impression", "") or state["extracted_info"].get("findings", "") or "",
    search_depth="advanced"
    )

    results = []

    # Tavily returns results in 'results' key
    for r in response.get("results", [])[:3]:  # limit to 3
        results.append({
            "content": r.get("content", "")
        })

    return {
        "search_results": results
    }

    


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
      • Trims summary if too long (anti-hallucination sprawl)
    Returns (possibly modified) summary + any new warnings.
    """
    final, warns = run_output_guardrails(state["final_summary"])
    return {"final_summary": final, "warnings": warns}

# ── Build the graph ───────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    g.add_node("extract_info", extract_info)
    g.add_node("web_search", web_search)
    g.add_node("synthesise", synthesise)

    g.set_entry_point("extract_info")
    g.add_edge("extract_info", "web_search")
    g.add_edge("web_search", "synthesise")
    g.add_edge("synthesise", END)

    return g.compile()


_graph = build_graph()

#print(_graph.visualize())  # for debugging – view the graph structure in the console
# ── Public entry-point ────────────────────────────────────────────────────────
raw_text = """The liver is normal in size and contour, 
                demonstrating a homogeneous echotexture without evidence of focal 
                masses or intrahepatic biliary ductal dilation. The gallbladder is well-distended 
                with a thin, smooth wall; no gallstones, sludge, or pericholecystic fluid are 
                visualized. The common bile duct (CBD) is within normal limits, measuring 4 mm 
                in diameter. The pancreas is partially obscured by overlying bowel gas, 
                but the visualized head and body appear unremarkable. Both kidneys are normal in 
                size and position, showing maintained corticomedullary differentiation without 
                signs of hydronephrosis or renal calculi. The spleen is of normal size and 
                echogenicity. No free fluid (ascites) or enlarged lymph nodes are identified 
                within the visualized abdominal cavity. The abdominal aorta is of normal caliber 
                throughout its visualized course."""
def run_agent(raw_text: str) -> str:
    """Run the full LangGraph pipeline and return the final summary."""
    result = _graph.invoke({"raw_text": raw_text, "search_results": []})
    return(result["final_summary"])

#run_agent(raw_text)