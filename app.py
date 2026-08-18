"""
Placement-Ready AI Career Agent
================================
PROJECT 2 — College AI Agents Assignment

A LangGraph-orchestrated career analysis agent for engineering students.
Given a resume PDF, a target job role, and an optional GitHub username,
it produces a structured Placement Readiness Report: skill-gap analysis,
prioritized learning path, project recommendations, GitHub evidence
analysis, an explainable readiness score, and a week-by-week roadmap.

This is NOT a RAG / document Q&A agent (see Project 1). There is no
vector store and no embedding step — the LLM reasons directly over the
extracted resume text, and a genuine multi-node LangGraph StateGraph
with a conditional branch drives the workflow.

Run locally:
    uvicorn app:app --reload

Deploy on Render:
    Start command: uvicorn app:app --host 0.0.0.0 --port $PORT
    Env vars:       GOOGLE_API_KEY   (required)
                     GOOGLE_LLM_MODEL (optional, default: gemini-flash-latest)
"""

import json
import os
import re
from typing import Optional, TypedDict

import requests
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from pypdf import PdfReader

# --------------------------------------------------------------------------
# LLM setup
# --------------------------------------------------------------------------

GOOGLE_LLM_MODEL = os.environ.get("GOOGLE_LLM_MODEL", "gemini-flash-latest")


def get_llm(temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    """
    Lazily construct the Gemini chat model. Constructed per-call (cheap)
    rather than at import time so a missing GOOGLE_API_KEY only breaks
    /analyze requests, not app startup (e.g. GET / still works for health
    checks even before the key is configured).
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY environment variable is not set. "
            "Set it before calling /analyze."
        )
    return ChatGoogleGenerativeAI(
        model=GOOGLE_LLM_MODEL,
        google_api_key=api_key,
        temperature=temperature,
    )


class LLMResponseParsingError(Exception):
    """
    Raised when Gemini's response text cannot be parsed as JSON after all
    recovery attempts. Carries a short, user-safe message — the raw model
    output is logged server-side (stderr) only, never sent to the client.
    """


def _extract_text_from_response(response) -> str:
    """
    ChatGoogleGenerativeAI (langchain-google-genai) can return response.content
    as either:
      - a plain string, or
      - a list of content blocks, e.g. [{"type": "text", "text": "..."}]
    Naively doing str(response.content) on the list form stringifies the
    Python list/dict repr itself rather than pulling out the real text,
    which is what was breaking JSON parsing. This extracts the actual text
    in both cases.
    """
    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # Standard shape: {"type": "text", "text": "..."}
                if "text" in block:
                    parts.append(block["text"])
        return "\n".join(parts)

    # Fallback for any other shape.
    return str(content)


def _extract_json_from_text(text: str) -> dict:
    """
    Robustly pull a JSON object out of raw LLM text that may:
      - be plain JSON already
      - be wrapped in ```json ... ``` or ``` ... ``` fences
      - have extra commentary before/after the JSON block
    """
    stripped = text.strip()

    # 1) Try parsing as-is first (the ideal case).
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # 2) Look for a fenced code block (```json ... ``` or ``` ... ```)
    #    anywhere in the text, not just at the very start/end.
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", stripped, re.DOTALL | re.IGNORECASE)
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    # 3) Fall back to grabbing the first "{" through the last "}" in the text,
    #    in case there's stray commentary with no fences at all.
    brace_match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    raise LLMResponseParsingError(
        "The AI model returned a response that could not be parsed as JSON."
    )


def call_llm_json(system_prompt: str, user_prompt: str, temperature: float = 0.2) -> dict:
    """
    Call Gemini and parse a strict-JSON response, handling both the
    plain-string and content-block response shapes, and stripping markdown
    code fences the model may add despite being told not to.
    """
    llm = get_llm(temperature=temperature)
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    response = llm.invoke(messages)
    text = _extract_text_from_response(response)

    try:
        return _extract_json_from_text(text)
    except LLMResponseParsingError:
        # Log full raw output server-side for debugging; never expose it
        # to the client (requirement: no raw dump on the webpage).
        print(f"[call_llm_json] Failed to parse model output as JSON. Raw text:\n{text[:2000]}")
        raise



# --------------------------------------------------------------------------
# State definition
# --------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    # Inputs
    resume_text: str
    target_role: str
    github_username: Optional[str]

    # Guardrail
    guardrail_passed: bool
    rejection_reason: Optional[str]

    # Step 1 - resume analysis
    resume_skills: dict

    # Step 2 - job requirement analysis
    job_skills: dict

    # Step 3 - skill gap
    matched_skills: list
    partial_skills: list
    missing_skills: list

    # Step 4 - priority
    priority_skills: dict

    # Step 5 - project recommendations
    projects: list

    # Step 6 - GitHub analysis
    github_analysis: Optional[dict]

    # Step 7 - readiness
    readiness_score: int
    readiness_breakdown: dict

    # Step 8 - roadmap
    roadmap: dict

    # Final
    final_report: dict


# --------------------------------------------------------------------------
# Node: Guardrail
# --------------------------------------------------------------------------
# NOTE ON THE GUARDRAIL REDESIGN:
# The guardrail previously made its own Gemini call to classify the input as
# SAFE/UNSAFE. On the Google AI Studio free tier (20 requests/day) that alone
# burned 1 of every request's calls before any real analysis happened. It is
# replaced here with a fast, zero-cost local heuristic pre-filter — no LLM
# call — that catches the same classes of abuse (prompt injection, requests
# to reveal system instructions, jailbreak phrasing, and attempts to
# manipulate the readiness score). This keeps a real guardrail in place while
# spending 0 Gemini calls on it, so a rejected request costs nothing and a
# legitimate request has its full quota available for the single analysis
# call below.

_INJECTION_PATTERNS = [
    r"ignore\s+(all|previous|prior|above)\s+instructions",
    r"disregard\s+(all|previous|prior|above)\s+instructions",
    r"forget\s+(your|all)\s+(instructions|rules)",
    r"system\s*prompt",
    r"reveal\s+.*(prompt|instructions|hidden)",
    r"you\s+are\s+now\s+(dan|jailbroken|unrestricted)",
    r"pretend\s+(you|to)\s+(are|be)\s+.*(unrestricted|no\s+rules|without\s+restrictions)",
    r"act\s+as\s+.*(unrestricted|without\s+restrictions|no\s+rules)",
    r"bypass\s+(the\s+)?(rules|restrictions|guardrails|safety)",
    r"override\s+(the\s+)?(rules|instructions|system)",
    r"give\s+me\s+(a\s+)?100\s*/\s*100",
    r"always\s+(give|say|return)\s+.*(100|perfect|maximum)\s*(score|readiness)?",
    r"regardless\s+of\s+(my|the)\s+(skills|resume|qualifications)",
]


def guardrail_check(state: AgentState) -> AgentState:
    combined_text = f"{state.get('target_role', '')} {state.get('github_username', '')}".lower()
    for pattern in _INJECTION_PATTERNS:
        if re.search(pattern, combined_text):
            return {
                **state,
                "guardrail_passed": False,
                "rejection_reason": "Input rejected by safety filter.",
            }
    return {**state, "guardrail_passed": True}


def route_after_guardrail(state: AgentState) -> str:
    return "validate_input" if state.get("guardrail_passed") else "END"


# --------------------------------------------------------------------------
# Node: Validate input
# --------------------------------------------------------------------------

def validate_input(state: AgentState) -> AgentState:
    if not state.get("resume_text", "").strip():
        return {
            **state,
            "guardrail_passed": False,
            "rejection_reason": (
                "No extractable text found in the uploaded PDF. "
                "This resume may be a scanned image — OCR would be required."
            ),
        }
    if not state.get("target_role", "").strip():
        return {
            **state,
            "guardrail_passed": False,
            "rejection_reason": "No target job role was provided.",
        }
    return state


# --------------------------------------------------------------------------
# Node: Main analysis (THE single Gemini call)
# --------------------------------------------------------------------------
# This one node replaces what used to be five separate Gemini calls (resume
# extraction, job requirement analysis, skill-gap classification, priority
# ranking, and project recommendation). Combining them into one prompt/one
# call is the core quota fix: it takes the workflow from ~6 Gemini calls
# down to exactly 1 per successful analysis.

MAIN_ANALYSIS_SYSTEM_PROMPT = """You are an expert technical recruiter and career coach analyzing a
student's resume against a target job role for campus placement preparation.

Perform ALL of the following in a single pass and return ONE structured JSON object:

1. RESUME EXTRACTION — extract ONLY what is explicitly present in the resume text.
   Do NOT invent, assume, or infer skills, projects, education, experience, or
   certifications that are not stated or clearly evidenced.

2. JOB REQUIREMENT ANALYSIS — identify the realistic technical requirements a
   B.Tech-level candidate would be evaluated on for the target role (use a
   pasted job description if one is supplied, otherwise infer standard
   requirements for that role).

3. SKILL GAP ANALYSIS — classify every required/preferred job skill as:
   - matched: the student clearly has it (exact or close synonym match)
   - partial: related/adjacent evidence but not a direct, confident match
   - missing: no evidence at all in the resume

4. PRIORITY RANKING — rank the missing skills into HIGH / MEDIUM / LOW
   importance specifically for this target role.

5. PROJECT RECOMMENDATIONS — recommend EXACTLY 3 realistic, resume/GitHub-worthy
   projects (each buildable in 1-4 weeks by a student) that close the
   priority skill gaps.

6. ROADMAP — build a practical 5-week preparation roadmap based on the
   priority gaps and the recommended projects.

Return ONLY valid JSON (no markdown fences, no commentary) with EXACTLY this shape:
{
  "resume_skills": {
    "languages": [...],
    "frameworks": [...],
    "libraries": [...],
    "databases": [...],
    "tools_cloud": [...],
    "projects": [{"name": "...", "description": "...", "technologies": [...]}],
    "education": [{"degree": "...", "institution": "...", "detail": "..."}],
    "experience": [{"role": "...", "organization": "...", "detail": "..."}],
    "certifications": [...]
  },
  "job_required_skills": {
    "required": {"languages": [...], "frameworks": [...], "tools_cloud": [...], "databases": [...], "concepts": [...]},
    "preferred": {"languages": [...], "frameworks": [...], "tools_cloud": [...], "databases": [...], "concepts": [...]}
  },
  "matched_skills": [...],
  "partial_skills": [...],
  "missing_skills": [...],
  "priority_skills": {"HIGH": [...], "MEDIUM": [...], "LOW": [...]},
  "recommended_projects": [
    {"title": "...", "problem_solved": "...", "technologies": [...], "skills_developed": [...], "difficulty": "Easy | Medium | Hard", "why_it_helps": "..."}
  ],
  "roadmap": {"Week 1": "...", "Week 2": "...", "Week 3": "...", "Week 4": "...", "Week 5": "..."}
}

"recommended_projects" must contain EXACTLY 3 entries.
Every skill listed in "missing_skills" must appear in exactly one of HIGH/MEDIUM/LOW
inside "priority_skills".
"""


def main_analysis(state: AgentState) -> AgentState:
    user_prompt = (
        f"Target role: {state['target_role']}\n\n"
        f"Resume text:\n{state['resume_text'][:15000]}"
    )
    result = call_llm_json(MAIN_ANALYSIS_SYSTEM_PROMPT, user_prompt, temperature=0.2)

    return {
        **state,
        "resume_skills": result.get("resume_skills", {}),
        "job_skills": result.get("job_required_skills", {}),
        "matched_skills": result.get("matched_skills", []),
        "partial_skills": result.get("partial_skills", []),
        "missing_skills": result.get("missing_skills", []),
        "priority_skills": result.get("priority_skills", {"HIGH": [], "MEDIUM": [], "LOW": []}),
        "projects": result.get("recommended_projects", []),
        "roadmap": result.get("roadmap", {}),
    }


def route_after_main(state: AgentState) -> str:
    return "github_analyzer" if state.get("github_username", "").strip() else "placement_readiness"


# --------------------------------------------------------------------------
# Node: GitHub analyzer (public REST API, no token)
# --------------------------------------------------------------------------

def github_analyzer(state: AgentState) -> AgentState:
    username = state.get("github_username", "").strip()
    headers = {"Accept": "application/vnd.github+json"}
    analysis = {
        "username": username,
        "found": False,
        "public_repos": 0,
        "languages": [],
        "repos_summary": [],
        "error": None,
    }

    try:
        user_resp = requests.get(
            f"https://api.github.com/users/{username}", headers=headers, timeout=10
        )
        if user_resp.status_code == 404:
            analysis["error"] = "GitHub username not found."
            return {**state, "github_analysis": analysis}
        if user_resp.status_code == 403:
            analysis["error"] = "GitHub API rate limit reached. Try again later."
            return {**state, "github_analysis": analysis}
        user_resp.raise_for_status()
        user_data = user_resp.json()
        analysis["found"] = True
        analysis["public_repos"] = user_data.get("public_repos", 0)

        repos_resp = requests.get(
            f"https://api.github.com/users/{username}/repos",
            headers=headers,
            params={"sort": "updated", "per_page": 20},
            timeout=10,
        )
        repos_resp.raise_for_status()
        repos = repos_resp.json()

        if not repos:
            analysis["error"] = "No public repositories found for this user."
            return {**state, "github_analysis": analysis}

        language_counts: dict = {}
        repos_summary = []
        for repo in repos[:10]:
            lang = repo.get("language")
            if lang:
                language_counts[lang] = language_counts.get(lang, 0) + 1
            repos_summary.append(
                {
                    "name": repo.get("name"),
                    "description": repo.get("description"),
                    "language": lang,
                    "stars": repo.get("stargazers_count", 0),
                    "updated_at": repo.get("updated_at"),
                    "fork": repo.get("fork", False),
                }
            )

        analysis["languages"] = sorted(
            language_counts, key=lambda k: language_counts[k], reverse=True
        )
        analysis["repos_summary"] = repos_summary

    except requests.exceptions.RequestException as exc:
        analysis["error"] = f"GitHub API request failed: {exc}"

    return {**state, "github_analysis": analysis}


# --------------------------------------------------------------------------
# Node: Placement readiness (deterministic, explainable scoring)
# --------------------------------------------------------------------------

def placement_readiness(state: AgentState) -> AgentState:
    matched = len(state.get("matched_skills", []))
    partial = len(state.get("partial_skills", []))
    missing = len(state.get("missing_skills", []))
    total_required = matched + partial + missing

    # 1. Skill coverage (0-50 points): matched counts full, partial counts half.
    if total_required > 0:
        coverage_ratio = (matched + 0.5 * partial) / total_required
    else:
        coverage_ratio = 0.5  # neutral if nothing to compare
    coverage_points = round(coverage_ratio * 50)

    # 2. Project relevance (0-20 points): based on number of resume projects,
    #    capped, as a proxy for demonstrated applied experience.
    resume_projects = state.get("resume_skills", {}).get("projects", []) or []
    project_points = min(len(resume_projects) * 5, 20)

    # 3. Experience (0-15 points): based on presence of listed experience entries.
    experience_entries = state.get("resume_skills", {}).get("experience", []) or []
    experience_points = min(len(experience_entries) * 7, 15)

    # 4. GitHub / project evidence (0-15 points).
    github_analysis = state.get("github_analysis")
    if github_analysis and github_analysis.get("found"):
        repo_count = github_analysis.get("public_repos", 0)
        github_points = min(round(repo_count * 1.5), 15)
    elif github_analysis and not github_analysis.get("found"):
        github_points = 0
    else:
        # No GitHub provided at all — treat as neutral-low rather than zero,
        # since resume-only evaluation shouldn't be unfairly penalized.
        github_points = 7

    total_score = min(coverage_points + project_points + experience_points + github_points, 100)

    breakdown = {
        "skill_coverage": {"points": coverage_points, "max": 50},
        "project_relevance": {"points": project_points, "max": 20},
        "experience": {"points": experience_points, "max": 15},
        "github_evidence": {"points": github_points, "max": 15},
    }

    return {**state, "readiness_score": int(total_score), "readiness_breakdown": breakdown}


# --------------------------------------------------------------------------
# Node: Final report builder
# --------------------------------------------------------------------------

def final_report_builder(state: AgentState) -> AgentState:
    report = {
        "target_role": state.get("target_role"),
        "estimated_placement_readiness": state.get("readiness_score"),
        "readiness_breakdown": state.get("readiness_breakdown"),
        "matched_skills": state.get("matched_skills", []),
        "partial_skills": state.get("partial_skills", []),
        "missing_skills": state.get("missing_skills", []),
        "priority_skills": state.get("priority_skills", {}),
        "recommended_projects": state.get("projects", []),
        "github_analysis": state.get("github_analysis"),
        "roadmap": state.get("roadmap", {}),
        "next_steps": _build_next_steps(state),
    }
    return {**state, "final_report": report}


def _build_next_steps(state: AgentState) -> list:
    steps = []
    high_priority = state.get("priority_skills", {}).get("HIGH", [])
    if high_priority:
        steps.append(f"Start closing your highest-priority gap: {high_priority[0]}.")
    projects = state.get("projects", [])
    if projects:
        steps.append(f"Begin building: \"{projects[0].get('title', 'your first recommended project')}\".")
    if not state.get("github_username", "").strip():
        steps.append("Create/update a GitHub profile with your project work — it strengthens your evidence.")
    steps.append("Revisit this analysis after 2-3 weeks of focused work to track improvement.")
    return steps[:5]


# --------------------------------------------------------------------------
# Graph construction
# --------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("guardrail_check", guardrail_check)
    graph.add_node("validate_input", validate_input)
    graph.add_node("main_analysis", main_analysis)
    graph.add_node("github_analyzer", github_analyzer)
    graph.add_node("placement_readiness", placement_readiness)
    graph.add_node("final_report_builder", final_report_builder)

    graph.set_entry_point("guardrail_check")

    graph.add_conditional_edges(
        "guardrail_check",
        route_after_guardrail,
        {"validate_input": "validate_input", "END": END},
    )

    # validate_input can also reject (e.g. empty PDF text) — same pattern.
    graph.add_conditional_edges(
        "validate_input",
        route_after_guardrail,
        {"validate_input": "main_analysis", "END": END},
    )

    graph.add_conditional_edges(
        "main_analysis",
        route_after_main,
        {"github_analyzer": "github_analyzer", "placement_readiness": "placement_readiness"},
    )

    graph.add_edge("github_analyzer", "placement_readiness")
    graph.add_edge("placement_readiness", "final_report_builder")
    graph.add_edge("final_report_builder", END)

    return graph.compile()


AGENT_GRAPH = build_graph()


# --------------------------------------------------------------------------
# PDF text extraction
# --------------------------------------------------------------------------

def extract_resume_text(file_bytes: bytes) -> str:
    import io

    reader = PdfReader(io.BytesIO(file_bytes))
    text_parts = []
    for page in reader.pages:
        extracted = page.extract_text() or ""
        text_parts.append(extracted)
    return "\n".join(text_parts).strip()


# --------------------------------------------------------------------------
# FastAPI app
# --------------------------------------------------------------------------

app = FastAPI(title="Placement-Ready AI Career Agent", version="1.0.0")


@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse(content=UI_HTML)


@app.get("/health")
def health():
    return JSONResponse({"status": "ok", "service": "placement-ready-ai-career-agent"})


@app.post("/analyze")
async def analyze(
    resume: UploadFile = File(...),
    target_role: str = Form(...),
    github_username: str = Form(default=""),
):
    file_bytes = await resume.read()

    try:
        resume_text = extract_resume_text(file_bytes)
    except Exception as exc:
        return JSONResponse(
            status_code=400,
            content={"error": f"Could not read PDF: {exc}"},
        )

    if not resume_text.strip():
        return JSONResponse(
            status_code=400,
            content={
                "error": (
                    "No extractable text found in this PDF. "
                    "It may be a scanned image — OCR would be required, "
                    "which this agent does not perform."
                )
            },
        )

    initial_state: AgentState = {
        "resume_text": resume_text,
        "target_role": target_role.strip(),
        "github_username": github_username.strip(),
    }

    try:
        result = AGENT_GRAPH.invoke(initial_state)
    except RuntimeError as exc:
        # e.g. missing GOOGLE_API_KEY
        return JSONResponse(status_code=500, content={"error": str(exc)})
    except LLMResponseParsingError:
        # Full raw model output was already logged server-side in call_llm_json.
        return JSONResponse(
            status_code=502,
            content={
                "error": (
                    "The AI model returned an unexpected response and the analysis "
                    "could not be completed. Please try again."
                )
            },
        )
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": f"Analysis failed: {exc}"})

    if not result.get("guardrail_passed", True):
        return JSONResponse(
            status_code=200,
            content={"rejected": True, "message": "I am not authorized to answer this."},
        )

    return JSONResponse(status_code=200, content={"rejected": False, "report": result["final_report"]})


# --------------------------------------------------------------------------
# Minimal custom UI (single inline HTML string — no templates/ directory)
# --------------------------------------------------------------------------

UI_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<title>Placement-Ready AI Career Agent</title>
<style>
  :root { --accent:#4f46e5; --bg:#0f1220; --card:#171a2b; --text:#e8e9f3; --muted:#9a9db8; }
  * { box-sizing: border-box; }
  body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; background: var(--bg); color: var(--text); margin:0; padding: 32px 16px; }
  .wrap { max-width: 760px; margin: 0 auto; }
  h1 { font-size: 22px; margin-bottom: 4px; }
  p.sub { color: var(--muted); margin-top: 0; margin-bottom: 24px; }
  .card { background: var(--card); border-radius: 12px; padding: 24px; margin-bottom: 20px; }
  label { display:block; font-size: 13px; color: var(--muted); margin-bottom: 6px; margin-top: 16px; }
  input[type=text], input[type=file] { width:100%; padding: 10px 12px; border-radius: 8px; border: 1px solid #2a2e45; background:#12142200; background:#0f1120; color: var(--text); }
  button { margin-top: 22px; width:100%; padding: 12px; border:none; border-radius: 8px; background: var(--accent); color:white; font-weight:600; cursor:pointer; font-size:15px; }
  button:disabled { opacity:0.6; cursor:default; }
  .section-title { font-weight:700; margin-top: 20px; margin-bottom:8px; color:var(--accent); }
  .skill-tag { display:inline-block; padding:3px 9px; border-radius: 999px; font-size:12px; margin:2px; }
  .matched { background:#14532d; color:#bbf7d0; }
  .partial { background:#713f12; color:#fde68a; }
  .missing { background:#7f1d1d; color:#fecaca; }
  .score { font-size: 40px; font-weight:800; color: var(--accent); }
  .project { border:1px solid #2a2e45; border-radius:10px; padding:14px; margin-bottom:10px; }
  .muted { color: var(--muted); font-size: 13px; }
  #error { color:#fca5a5; margin-top:10px; }
</style>
</head>
<body>
<div class="wrap">
  <h1>PLACEMENT-READY AI CAREER AGENT</h1>
  <p class="sub">Upload your resume, tell it your target role, and get a structured readiness report.</p>

  <div class="card">
    <label>1. Resume PDF</label>
    <input type="file" id="resume" accept="application/pdf" />

    <label>2. Target Job Role</label>
    <input type="text" id="target_role" placeholder="Machine Learning Engineer" />

    <label>3. GitHub Username (optional)</label>
    <input type="text" id="github_username" placeholder="username" />

    <button id="analyzeBtn" onclick="analyze()">ANALYZE RESUME</button>
    <div id="error"></div>
  </div>

  <div id="result"></div>
</div>

<script>
async function analyze() {
  const btn = document.getElementById('analyzeBtn');
  const errEl = document.getElementById('error');
  const resultEl = document.getElementById('result');
  errEl.textContent = '';
  resultEl.innerHTML = '';

  const resumeFile = document.getElementById('resume').files[0];
  const targetRole = document.getElementById('target_role').value.trim();
  const githubUsername = document.getElementById('github_username').value.trim();

  if (!resumeFile) { errEl.textContent = 'Please upload a resume PDF.'; return; }
  if (!targetRole) { errEl.textContent = 'Please enter a target job role.'; return; }

  const formData = new FormData();
  formData.append('resume', resumeFile);
  formData.append('target_role', targetRole);
  formData.append('github_username', githubUsername);

  btn.disabled = true;
  btn.textContent = 'Analyzing...';

  try {
    const resp = await fetch('/analyze', { method: 'POST', body: formData });
    const data = await resp.json();

    if (!resp.ok) {
      errEl.textContent = data.error || 'Something went wrong.';
      return;
    }
    if (data.rejected) {
      errEl.textContent = data.message;
      return;
    }
    renderReport(data.report);
  } catch (e) {
    errEl.textContent = 'Request failed: ' + e.message;
  } finally {
    btn.disabled = false;
    btn.textContent = 'ANALYZE';
  }
}

function tag(text, cls) {
  return `<span class="skill-tag ${cls}">${text}</span>`;
}

function renderReport(r) {
  const el = document.getElementById('result');
  let html = '';

  html += `<h2 style="margin-top:32px;">PLACEMENT READINESS REPORT</h2>`;

  html += `<div class="card">
    <div class="muted">Target Role:</div>
    <div style="font-size:18px;font-weight:700;">${r.target_role}</div>
    <div class="muted" style="margin-top:16px;">Estimated Placement Readiness:</div>
    <div class="score">${r.estimated_placement_readiness} / 100</div>
  </div>`;

  html += `<div class="card">
    <div class="section-title">Matched Skills:</div>
    ${(r.matched_skills || []).map(s => tag('✓ ' + s, 'matched')).join(' ') || '<span class="muted">None</span>'}
    <div class="section-title">Partial Skills:</div>
    ${(r.partial_skills || []).map(s => tag('◐ ' + s, 'partial')).join(' ') || '<span class="muted">None</span>'}
    <div class="section-title">Missing Skills:</div>
    ${(r.missing_skills || []).map(s => tag('✗ ' + s, 'missing')).join(' ') || '<span class="muted">None</span>'}
  </div>`;

  const pr = r.priority_skills || {};
  const orderedPriority = [...(pr.HIGH || []), ...(pr.MEDIUM || []), ...(pr.LOW || [])];
  html += `<div class="card">
    <div class="section-title">Priority Skills:</div>
    <ol>${orderedPriority.map(s => `<li>${s}</li>`).join('') || '<li class="muted">None</li>'}</ol>
  </div>`;

  html += `<div class="card"><div class="section-title">Recommended Projects:</div>`;
  (r.recommended_projects || []).forEach((p, i) => {
    html += `<div class="project">
      <div style="font-weight:700;">${i+1}. ${p.title || ''}</div>
      <div class="muted">Difficulty: ${p.difficulty || '—'}</div>
      <div style="margin-top:6px;">${p.problem_solved || ''}</div>
      <div class="muted" style="margin-top:6px;">Technologies: ${(p.technologies || []).join(', ')}</div>
      <div class="muted">Why: ${p.why_it_helps || ''}</div>
    </div>`;
  });
  html += `</div>`;

  if (r.github_analysis) {
    const g = r.github_analysis;
    html += `<div class="card"><div class="section-title">GitHub Analysis:</div>`;
    if (g.error) {
      html += `<div class="muted">${g.error}</div>`;
    } else {
      html += `<div>Repositories: ${g.public_repos}</div>
        <div>Main languages: ${(g.languages || []).join(', ') || '—'}</div>`;
    }
    html += `</div>`;
  }

  html += `<div class="card"><div class="section-title">Preparation Roadmap:</div>`;
  const roadmap = r.roadmap || {};
  Object.keys(roadmap).forEach(week => {
    html += `<div><strong>${week}:</strong> ${roadmap[week]}</div>`;
  });
  html += `</div>`;

  html += `<div class="card"><div class="section-title">Next Steps:</div>
    <ol>${(r.next_steps || []).map(s => `<li>${s}</li>`).join('')}</ol>
  </div>`;

  el.innerHTML = html;
}
</script>
</body>
</html>
"""
