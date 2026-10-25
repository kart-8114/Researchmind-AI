"""OpenAI integration for ResearchMind AI.

OpenAI is used for language understanding only: summaries, structured
information extraction, research-gap analysis, project candidates, and the
paper Q&A assistant. Classical AI algorithms in this project remain local:
Forward Chaining, Best-First Search, and HTN planning.
"""

import json
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
try:
    from openai import OpenAI
except ImportError:  # graceful fallback until `pip install -r requirements.txt` is run
    OpenAI = None

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna").strip()

_client: Optional[OpenAI] = None
if OPENAI_API_KEY and OpenAI is not None:
    _client = OpenAI(api_key=OPENAI_API_KEY)


ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "domain": {"type": "string"},
        "problem": {"type": "string"},
        "algorithms": {"type": "array", "items": {"type": "string"}},
        "datasets": {"type": "array", "items": {"type": "string"}},
        "metrics": {"type": "array", "items": {"type": "string"}},
        "limitations": {"type": "array", "items": {"type": "string"}},
        "research_gaps": {"type": "array", "items": {"type": "string"}},
        "future_work": {"type": "array", "items": {"type": "string"}},
        "key_contributions": {"type": "array", "items": {"type": "string"}}
    },
    "required": [
        "summary", "domain", "problem", "algorithms", "datasets", "metrics",
        "limitations", "research_gaps", "future_work", "key_contributions"
    ],
    "additionalProperties": False
}


PROJECT_SCHEMA = {
    "type": "object",
    "properties": {
        "candidates": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "problem_statement": {"type": "string"},
                    "algorithms": {"type": "array", "items": {"type": "string"}},
                    "datasets": {"type": "array", "items": {"type": "string"}},
                    "metrics": {"type": "array", "items": {"type": "string"}},
                    "future_scope": {"type": "array", "items": {"type": "string"}},
                    "feasibility": {"type": "number"},
                    "novelty": {"type": "number"},
                    "resource_suitability": {"type": "number"},
                    "rationale": {"type": "string"}
                },
                "required": [
                    "title", "problem_statement", "algorithms", "datasets", "metrics",
                    "future_scope", "feasibility", "novelty", "resource_suitability", "rationale"
                ],
                "additionalProperties": False
            }
        }
    },
    "required": ["candidates"],
    "additionalProperties": False
}


def is_available() -> bool:
    return _client is not None


def _response_json(instructions: str, input_text: str, schema: Dict[str, Any], name: str) -> Dict[str, Any]:
    if _client is None:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    response = _client.responses.create(
        model=OPENAI_MODEL,
        instructions=instructions,
        input=input_text,
        text={
            "format": {
                "type": "json_schema",
                "name": name,
                "strict": True,
                "schema": schema,
            }
        },
    )
    return json.loads(response.output_text)


def analyze_paper(text: str) -> Dict[str, Any]:
    """Analyze a research paper and return structured evidence."""
    excerpt = text[:16000]
    instructions = (
        "You are the ResearchMind AI literature-analysis engine. Analyze the supplied "
        "research-paper text. Extract only information supported by the text. Do not "
        "invent datasets, algorithms, metrics, results, or gaps. If an item is not "
        "present, return an empty array or a short 'Not explicitly stated' statement. "
        "The domain should be a concise research domain. Research gaps should be "
        "potential gaps supported by limitations, missing comparisons, scalability, "
        "generalization, or future-work statements in the paper."
    )
    return _response_json(instructions, excerpt, ANALYSIS_SCHEMA, "research_analysis")


def generate_project_candidates(domain: str, algorithms: List[str], datasets: List[str],
                                metrics: List[str], gaps: List[str], limitations: List[str]) -> Dict[str, Any]:
    """Generate candidate projects that will subsequently be selected by Forward Chaining."""
    context = {
        "domain": domain,
        "algorithms": algorithms,
        "datasets": datasets,
        "metrics": metrics,
        "research_gaps": gaps,
        "limitations": limitations,
    }
    instructions = (
        "You are the project-idea generator inside ResearchMind AI. Generate exactly "
        "three realistic student/research projects from the supplied literature facts. "
        "Keep the projects implementable. Do not claim a project is novel unless the "
        "provided gaps support it. Scores must be numbers from 0.0 to 1.0. Prefer the "
        "algorithms and datasets found in the literature, but you may suggest a clearly "
        "labelled alternative when necessary. These candidates will be ranked by a "
        "separate classical Forward Chaining + Best-First Search layer."
    )
    return _response_json(instructions, json.dumps(context, ensure_ascii=False), PROJECT_SCHEMA, "project_candidates")


def answer_question(question: str, context: str) -> str:
    """Answer a question using only retrieved paper context."""
    if _client is None:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    response = _client.responses.create(
        model=OPENAI_MODEL,
        instructions=(
            "You are ResearchMind AI's paper assistant. Answer using only the supplied "
            "paper context. If the answer is not supported by the context, say that it "
            "is not available in the uploaded paper. Be concise and do not fabricate citations."
        ),
        input=f"PAPER CONTEXT:\n{context[:12000]}\n\nQUESTION:\n{question}",
    )
    return response.output_text.strip()


def extract_research_gaps(text: str) -> List[str]:
    """Extract precise research gaps from paper text using OpenAI."""
    if _client is None:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    
    schema = {
        "type": "object",
        "properties": {
            "research_gaps": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of 3 to 5 precise, technical research gaps identified in the literature."
            }
        },
        "required": ["research_gaps"],
        "additionalProperties": False
    }
    
    response = _response_json(
        "You are an expert AI research assistant. Analyze the given research paper text and extract 3 to 5 precise, distinct research gaps, unaddressed limitations, or open challenges that motivate future work.",
        text[:16000],
        schema,
        "research_gaps_extraction"
    )
    return response.get("research_gaps", [])
