# Placement-Ready AI Career Agent

**PROJECT 2 — College AI Agents Assignment**

A LangGraph-orchestrated career analysis agent for engineering students preparing
for campus placements. Upload a resume PDF, name a target job role, and
(optionally) a GitHub username — the agent returns a structured Placement
Readiness Report: skill-gap analysis, a prioritized learning path, three
tailored project recommendations, GitHub evidence analysis, an explainable
0–100 readiness score, and a week-by-week preparation roadmap.

---

## 1. Project Overview

This agent answers one question for a student: *"Given my resume and this
target role, what specifically should I do next?"* It does not answer
open-ended questions about the resume's contents (that is Project 1's job) —
it runs a fixed, multi-step evaluation pipeline and returns one structured
report per request.

## 2. Problem Statement

Students preparing for placements struggle to translate a job title ("Machine
Learning Engineer") into a concrete list of skill gaps and next actions. This
agent automates that translation: resume in, actionable report out.

## 3. Features

- PDF resume parsing (text extraction only — no OCR, no embeddings)
- LLM-driven extraction of skills, projects, education, experience, certifications
- LLM-driven job-role requirement modeling
- Skill-gap classification: **matched / partial / missing**
- Priority ranking of missing skills: **HIGH / MEDIUM / LOW**
- Exactly 3 tailored, realistic project recommendations
- Optional GitHub public-API analysis (repos, languages, activity)
- Explainable 0–100 **Estimated Placement Readiness** score
- 5-week personalized roadmap
- Guardrails against prompt injection, jailbreaks, off-topic requests, and
  score-manipulation attempts
- Single-page custom upload UI (no JSON playground needed for file upload)

## 4. Architecture

```
Browser (custom HTML/CSS/JS, served at /ui)
        │  multipart/form-data (resume PDF + role + github username)
        ▼
FastAPI (app.py)
    ├── GET  /         → health/status
    ├── GET  /ui        → custom upload interface
    ├── GET  /docs       → FastAPI auto-generated Swagger UI
    └── POST /analyze    → runs the LangGraph workflow, returns JSON report
                                  │
                                  ▼
                     LangGraph StateGraph (compiled once at import,
                     invoked fresh per request)
                                  │
              ┌───────────────────┼───────────────────────┐
              ▼                   ▼                        ▼
        Gemini LLM          GitHub public REST API     pypdf
   (langchain-google-genai)   (requests, no token)   (text extraction)
```

Everything — routes, state, nodes, graph wiring, guardrails, and the inline UI
— lives in a single `app.py`, per the project's simplicity constraint.

## 5. LangGraph Workflow

```
START
  ↓
guardrail_check ──(UNSAFE)──────────────────────────────────► END
  │ (SAFE)
  ▼
validate_input ──(empty PDF text / no role)─────────────────► END
  │ (valid)
  ▼
resume_analyzer
  ↓
job_requirement_analyzer
  ↓
skill_gap_analyzer
  ↓
priority_analyzer
  ↓
project_recommender
  │
  ├──(github_username provided)──► github_analyzer ──┐
  └──(no github_username)─────────────────────────────┤
                                                        ▼
                                            placement_readiness
                                                        ↓
                                              roadmap_generator
                                                        ↓
                                            final_report_builder
                                                        ↓
                                                       END
```

Built with `langgraph.graph.StateGraph` — no `AgentExecutor`, no legacy
LangChain agent, no `create_agent()`. Two genuine conditional edges are used:
one for the guardrail/validation short-circuit, one for the GitHub branch.

## 6. State

A single `TypedDict` (`AgentState`) threads through every node:

| Field | Description |
|---|---|
| `resume_text`, `target_role`, `github_username` | Raw inputs |
| `guardrail_passed`, `rejection_reason` | Guardrail outcome |
| `resume_skills` | Structured extraction from the resume |
| `job_skills` | Structured requirement profile for the target role |
| `matched_skills`, `partial_skills`, `missing_skills` | Gap classification |
| `priority_skills` | HIGH/MEDIUM/LOW ranked missing skills |
| `projects` | 3 recommended projects |
| `github_analysis` | GitHub API summary (or `None`) |
| `readiness_score`, `readiness_breakdown` | Score + explainable factor breakdown |
| `roadmap` | Week-by-week plan |
| `final_report` | Everything above, assembled for the API response |

Internal state is never returned to the client — `/analyze` returns only
`final_report`.

## 7. Guardrails

Before any resume/role data is processed, `guardrail_check` sends the
`target_role` field to Gemini with a strict classifier prompt that flags:
prompt injection, requests to reveal system instructions, jailbreak attempts,
off-topic requests, and attempts to manipulate the readiness score. An
`UNSAFE` verdict short-circuits the graph straight to `END`, and the API
returns exactly:

```
"I am not authorized to answer this."
```

Separately, `resume_analyzer`'s prompt explicitly instructs the model to
extract only what is present in the resume text — never to invent skills or
experience.

## 8. Resume Analysis

`resume_analyzer` sends the extracted resume text to Gemini with a prompt
constrained to explicit extraction (languages, frameworks, libraries,
databases, tools/cloud, projects, education, experience, certifications) and
returns strict JSON.

## 9. Skill-Gap Analysis

`job_requirement_analyzer` builds a required/preferred skill profile for the
target role. `skill_gap_analyzer` then classifies each required/preferred
skill as matched, partial, or missing against the resume's extracted skills.

## 10. GitHub Analysis

If a GitHub username is supplied, `github_analyzer` calls the public GitHub
REST API (`/users/{username}` and `/users/{username}/repos`) with no
authentication token. It reports repo count, the most-used languages across
recent repos, and a per-repo summary (name, description, language, stars,
last updated). Invalid usernames, empty repo lists, and rate-limit/API errors
are all handled gracefully and surfaced in the report rather than crashing
the run.

## 11. Project Recommendation

`project_recommender` generates exactly 3 projects sized for a B.Tech
student, targeted at the current missing/partial skills (weighted toward
HIGH-priority gaps), each with a title, problem solved, technologies, skills
developed, difficulty, and rationale.

## 12. Placement Readiness

`placement_readiness` computes a deterministic, explainable 0–100 score from
four weighted factors:

| Factor | Max points | Basis |
|---|---|---|
| Skill coverage | 50 | matched + 0.5 × partial, over total required/preferred skills |
| Project relevance | 20 | number of resume-listed projects (capped) |
| Experience | 15 | number of resume-listed experience entries (capped) |
| GitHub evidence | 15 | public repo count if a GitHub username was analyzed; neutral default if not provided |

This is clearly labeled **"Estimated Placement Readiness"** — it is not, and
does not claim to be, an official hiring score.

## 13. Roadmap

`roadmap_generator` produces a 5-week plan derived from the priority skill
gaps and recommended projects (e.g. Week 1: fundamentals, Week 2–3: tooling
and project build, Week 4: deployment, Week 5: interview prep).

## 14. API

### `POST /analyze`
`multipart/form-data`:
- `resume` (file, required) — resume PDF
- `target_role` (string, required)
- `github_username` (string, optional)

Response:
```json
{
  "rejected": false,
  "report": {
    "target_role": "...",
    "estimated_placement_readiness": 78,
    "readiness_breakdown": { "...": "..." },
    "matched_skills": ["..."],
    "partial_skills": ["..."],
    "missing_skills": ["..."],
    "priority_skills": {"HIGH": [], "MEDIUM": [], "LOW": []},
    "recommended_projects": [ {"...": "..."} ],
    "github_analysis": {"...": "..."},
    "roadmap": {"Week 1": "...", "...": "..."},
    "next_steps": ["..."]
  }
}
```

If the guardrail rejects the request:
```json
{ "rejected": true, "message": "I am not authorized to answer this." }
```

### `GET /`
Simple health check: `{"status": "ok", "service": "placement-ready-ai-career-agent"}`

### `GET /ui`
The custom upload interface (file input, role input, optional GitHub input,
Analyze button, rendered report).

### `GET /docs`
FastAPI's automatic Swagger UI.

## 15. Local Setup

```bash
git clone <this-repo>
cd placement-agent
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

export GOOGLE_API_KEY="your-google-ai-studio-key"
# optional: export GOOGLE_LLM_MODEL="gemini-flash-latest"

uvicorn app:app --reload
```

Visit `http://localhost:8000/ui` for the interface, or `http://localhost:8000/docs`
for the API docs.

## 16. Render Deployment

- **Start command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
- **Environment variables:**
  - `GOOGLE_API_KEY` (required)
  - `GOOGLE_LLM_MODEL` (optional — defaults to `gemini-flash-latest`, Google's
    self-updating alias for the current Gemini Flash release, chosen so the
    deployment doesn't silently break when a specific dated model is retired)

## 17. Example Input/Output

**Input:**
- Resume: `resume.pdf`
- Target role: `Machine Learning Engineer`
- GitHub: `nidhi123`

**Output (abridged):**
```
TARGET ROLE: Machine Learning Engineer
ESTIMATED PLACEMENT READINESS: 78 / 100

MATCHED SKILLS:      Python, SQL, Machine Learning
PARTIAL SKILLS:      Deep Learning
MISSING SKILLS:      Docker, AWS

PRIORITY SKILLS:
  HIGH:   Deep Learning, Docker
  MEDIUM: AWS
  LOW:    Kubernetes

RECOMMENDED PROJECTS:
  1. End-to-End ML Deployment (Medium) — Python, FastAPI, Docker
     Why: builds the deployment skills required for the target role.
  2. ...
  3. ...

GITHUB ANALYSIS:
  Repositories: 6
  Main languages: Python, Java

PREPARATION ROADMAP:
  Week 1 → Deep Learning fundamentals
  Week 2 → TensorFlow/PyTorch
  Week 3 → Build project
  Week 4 → Deploy project
  Week 5 → Interview preparation
```
All figures above are illustrative — actual output is generated from the real
resume, role, and GitHub data supplied per request.

## 18. Limitations

- Scanned/image-only PDFs are rejected (no OCR) — a clear error is returned.
- Skill classification and scoring depend on LLM judgment for the
  matched/partial/missing calls; results can vary slightly between runs.
- GitHub analysis only sees public data reachable without a token, and is
  capped to the 10 most recently updated repos for language/summary purposes.
- The readiness score is a heuristic teaching tool, not a validated hiring
  predictor.

## 19. Future Enhancements

- Support pasting a real job description to replace the LLM-inferred role profile
- Resume improvement suggestions (phrasing, missing sections)
- Multi-resume comparison mode
- Caching GitHub API responses to reduce rate-limit exposure
- Export report as PDF

---

## Why This Agent Is Different From Agent 1

| | **Agent 1 — PDF RAG Knowledge Agent** | **Agent 2 — Placement-Ready AI Career Agent** |
|---|---|---|
| Core task | Retrieval-augmented Q&A over one document | Multi-step comparative analysis, scoring, and planning |
| Data sources | Single uploaded PDF | Resume PDF **+** inferred job requirements **+** live GitHub API |
| Retrieval mechanism | FAISS vector search over embeddings | None — direct LLM reasoning + deterministic comparison/scoring logic |
| Graph shape | Linear retrieve → answer | Branching pipeline with two conditional edges (guardrail short-circuit, GitHub skip) |
| Output | A direct answer to a user's question | A fixed-schema structured report: gap analysis, priorities, projects, score, roadmap |
| External tool use | None beyond local embeddings | Live third-party REST API (GitHub) |

Agent 1 is fundamentally an information-retrieval system. Agent 2 is an
evaluation-and-planning system: it doesn't answer a question about a
document, it produces a judgment (skill gaps, a score, a plan) by combining
multiple structured LLM calls, deterministic scoring logic, and an external
API — genuinely exercising LangGraph's branching and multi-node
orchestration rather than a single retrieve-then-generate hop.
