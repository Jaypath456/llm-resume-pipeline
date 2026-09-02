#!/usr/bin/env python3
"""
Batch resume tailoring pipeline.

For each job description in jobs/*.txt, this script:
  1. Sends an ISOLATED API call (Prompt A logic) to tailor content -- no chat history, so nothing bleeds between jobs.
  2. Deterministically links Technical Skills to whichever academic projects
     got selected for this JD (merge_project_skills-equivalent logic lives in fill_template()).
  3. Fills your fixed LaTeX template with the tailored content (Prompt B logic).
  4. Compiles with pdflatex.
  5. Parses the REAL compile log for Overfull/Underfull hbox warnings -- ground truth, not character-count guessing.
  6. Measures the REAL rendered PDF text (pdftotext) to check section line counts and sparse last lines --
     this uses a best-match forward scan that (a) never silently freezes on a bad match and (b) never lets
     one unmatched chunk corrupt the measurement of every chunk after it.
  7. If there are layout failures -- INCLUDING real page-count overflow (>1 page), even when every section
     individually measured within its own line target -- sends a targeted correction call (Prompt C logic)
     naming only the failing lines/sections, recompiles, up to MAX_LAYOUT_RETRIES times.
  8. Generates a cover letter (Prompt E logic).
  9. Writes everything to output/<company>_<role>/, including content.json and jd_text.txt so a later
     standalone verify_output.py run can reconstruct what to fix without another Gemini call.
  10. Records a status per job in the manifest: "done" (layout fully verified clean, incl. 1-page + proofread
      clean) or "needs_review" (something couldn't be confirmed/fixed). needs_review jobs are automatically
      retried on the next run, up to NEEDS_REVIEW_RETRY_LIMIT times, instead of being silently skipped forever.

NOTE ON WHAT IS AND ISN'T AUTO-FIXED:
  Section line-count mismatches, sparse orphan last lines, overused verbs, and page-count overflow all get
  automatically corrected within this loop. A genuine Groq proofread finding (an actual typo/garbled word in
  the tailored content, as opposed to a pdftotext rendering artifact) is NOT auto-corrected -- it's flagged
  and marks the job needs_review, but nothing rewrites the bullet. That's deliberate: unlike a line-count
  target, "is this proofread flag a real typo or a false positive" isn't something to blindly hand back to
  another model pass without a human glancing at it first.

  ADDITIONALLY (see CHANGELOG below): explicitly-unsupported terms and a code-level narrative-coherence
  check are now enforced INSIDE the retry loop, not just once before it -- see CHANGELOG for why.

CHANGELOG (guardrail fixes):
  - FIXED: check_unsupported_terms() / flag_unverified_terms() were only ever called once, immediately
    after tailor_content() and BEFORE the layout-fix retry loop. Since fix_layout() (Pass C) can rewrite
    Technical Skills content during any retry, a banned term introduced during a retry was never re-checked
    against the final compiled content -- the reported "clean" status could silently disagree with what
    was actually in the PDF. Both checks (plus a new narrative-coherence check) now re-run on every retry
    iteration, feed into the "is this layout_clean" decision, and are also sent to fix_layout() so the model
    is told exactly what to remove/rewrite.
  - ADDED: strip_unsupported_skills() -- Technical Skills is a flat comma-separated list, so a banned term
    found there is auto-removed deterministically (safe: can't break grammar). A banned term found inside a
    bullet SENTENCE is not auto-edited (removing a substring from prose can break grammar) -- it's instead
    surfaced to fix_layout() as a required rewrite, and blocks layout_clean until it's gone.
  - ADDED: check_narrative_coherence() -- a regex-level backstop for the "one deliverable per bullet" prompt
    rule, since that rule is currently enforced only by the model reading a paragraph of instructions and
    already produced one violation ("...processing 10k+ records while maintaining strict software
    architecture...") that slipped through untouched.

Setup:
  pip install google-generativeai
  Set at least GEMINI_API_KEY_1 as an environment variable. GEMINI_API_KEY_2 through _5 are optional --
  if set, call_gemini() automatically rotates to the next account when one hits its daily free-tier quota
  (RESOURCE_EXHAUSTED / 429), instead of failing the whole run.

Folder layout expected:
  ./Master_Resume_Content.md      (your evidence doc)
  ./Resume_Template.tex           (your fixed template with [BULLET N] placeholders)
  ./jobs/company_role.txt         (one JD per file, filename becomes the output folder name)
  ./output/                       (created automatically)

Run:
  python tailor_resumes.py
"""

import os
import re
import glob
import json
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from verify_output import verify_job, check_page_count
from google import genai
import time

# Load API keys from a .env file in the project root, if one exists.
# This means you no longer need to `export` keys manually every terminal
# session -- create a .env file (see .env.example) and this picks it up
# automatically. If python-dotenv isn't installed, or no .env file exists,
# this silently does nothing and falls back to whatever's already in your
# real environment variables (e.g. if you still prefer manual export, or
# have them set in ~/.bashrc).
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
MODEL_NAME = "gemini-3.6-flash"
MASTER_CONTENT_PATH = "Master_Resume_Content.md"
TEMPLATE_PATH = "Resume_Template.tex"
JOBS_DIR = "jobs"
OUTPUT_DIR = "output"
MAX_LAYOUT_RETRIES = 5
MANIFEST_PATH = ".processed_jobs.json"  # tracks which input .txt files are already done / needs_review
NEEDS_REVIEW_RETRY_LIMIT = 2  # how many extra runs to re-attempt a needs_review job before leaving it alone

LINE_TARGETS = {
    "Graduate Student Developer": 5,
    "Software Engineer": 8,
    "Cloud Engineer Intern": 3,
    "Data Engineer Intern": 2,
    "Academic Projects": 14,
    "Technical Skills": 7,
}

# Full project registry (metadata only -- bullets are always regenerated per
# JD). The script picks the 3 most JD-relevant of these; header text is
# assembled deterministically here so the model never has to reproduce
# exact formatting, only choose relevance and write bullets.
PROJECTS = {
    "lms": {
        "title": "Decoupled Learning Management System (LMS)",
        "context": "Individual Project",
        "date": "July 2026",
        "sort_key": "2026-07",
        "complexity_rank": 1,
        "keywords": "Python, Django, DRF, ReactJS, WebSockets, Django Channels, Daphne, PostgreSQL, Redis, Docker Compose, JWT, Locust, backtracking algorithms",
    },
    "fraud": {
        "title": "Fraud Detection using Graph Neural Networks",
        "context": "Team of 2 $|$ University at Buffalo, NY",
        "date": "March 2026",
        "sort_key": "2026-03",
        "complexity_rank": 2,
        "keywords": "PyTorch, PyTorch Geometric, GNNs, GraphSAGE, GAT, NetworkX, Pandas, Scikit-Learn, Focal Loss, ML, fraud/anomaly detection",
    },
    "pintos": {
        "title": "Pintos Operating System (User Programs Layer)",
        "context": "Individual Project $|$ University at Buffalo, NY",
        "date": "May 2026",
        "sort_key": "2026-05",
        "complexity_rank": 3,
        "keywords": "C, x86, operating systems, kernel development, concurrency, thread synchronization, file systems, process lifecycle, low-level systems programming",
    },
    "temp": {
        "title": "Temperature Monitoring System",
        "context": "Team of 3 $|$ University at Buffalo, NY",
        "date": "November 2025",
        "sort_key": "2025-11",
        "complexity_rank": 4,
        "keywords": "Python, Django, DRF, ReactJS, PostgreSQL, IoT, Arduino, Auth0, REST APIs, hardware-to-software pipeline",
    },
    "music": {
        "title": "Music Genre Classification",
        "context": "Undergraduate Capstone / Published Research (IJRAR)",
        "date": "April 2023",
        "sort_key": "2023-04",
        "complexity_rank": 5,
        "keywords": "Python, CatBoost, Scikit-Learn, Librosa, Pandas, NumPy, ML, audio processing, feature extraction, gradient boosting, classification, published research",
    },
}

# Fixed print order for Technical Skills categories -- matches the .tex
# template's hardcoded item order. Used by BOTH fill_template() (to fill
# the PDF) and build_measurement_chunks() (to measure it), so the two can
# never disagree about where each category's text actually appears.
SKILL_CATEGORY_ORDER = [
    "Languages",
    "Frameworks",
    "Artificial Intelligence & Machine Learning",
    "Software Engineering",
    "Tools",
]

# JD-triggered mandatory skill insertions -- deterministic, not model
# discretion. Only fires if the JD literally asks for these terms, and only
# inserts terms that already exist in the master content's Technical
# Skills pool (added there separately) -- this function never invents new
# skill tokens, it only forces inclusion of ones already vetted as true.
JD_TRIGGERED_SKILLS = {
    "Software Engineering": [
        (r"\balgorithms?\b", "Algorithms"),
        (r"\bdata structures?\b", "Data Structures"),
    ],
}


def apply_jd_triggered_skills(jd_text: str, skills: dict) -> dict:
    """
    For each category in JD_TRIGGERED_SKILLS, if the JD text matches a
    trigger pattern and the corresponding skill isn't already present in
    the model's output for that category, append it. Runs after
    tailor_content() returns, so it's a deterministic post-processing step,
    not a prompt instruction the model could skip or paraphrase around.
    """
    for category, triggers in JD_TRIGGERED_SKILLS.items():
        if category not in skills:
            continue
        current = skills[category]
        for pattern, term in triggers:
            already_present = re.search(re.escape(term), current, re.IGNORECASE)
            jd_mentions_it = re.search(pattern, jd_text, re.IGNORECASE)
            if jd_mentions_it and not already_present:
                skills[category] = f"{current}, {term}" if current else term
    return skills

def get_project_display_order(selected_pids: list) -> list:
    """
    SINGLE SOURCE OF TRUTH for the order Academic Projects are printed in.
    fill_template() uses this to build the PDF; build_measurement_chunks()
    uses this to know what order to search for bullets in. These two must
    NEVER compute this sort independently -- academic_projects_selected is
    the model's RELEVANCE ranking, not the date-sorted order they're
    actually displayed in, and measure_fill()'s scan is forward-only with
    no ability to recover from a mismatch. A drift here doesn't raise an
    error: it makes the scan silently skip past whichever project got
    scanned over while chasing the next (out-of-order) chunk, and that
    project's chunks then fail to match on every single retry forever,
    since the true text is now behind the search pointer.
    """
    return sorted(selected_pids, key=lambda pid: PROJECTS[pid]["sort_key"], reverse=True)


# Common resume action verbs to check for document-wide overuse. A verb appearing more than MAX_VERB_REPEAT times across all bullets gets flagged for diversification -- this catches repetition SCATTERED across non-consecutive bullets, which the "avoid same verb in consecutive bullets" content rule alone does not catch.
ACTION_VERB_LIST = [
    "built", "automated", "engineered", "developed", "designed", "architected",
    "implemented", "optimized", "created", "streamlined", "collaborated",
    "conducted", "mentored", "containerized", "transformed", "evaluated",
    "processed", "benchmarked", "integrated", "reduced", "enabled",
    "delivered", "resolved", "deployed", "configured",
]
MAX_VERB_REPEAT = 2

# Regex-level backstop for the "one deliverable per bullet" narrative-coherence
# prompt rule. This is a heuristic, not a parser: it will have false positives
# (a genuine causal "X, which enabled Y" clause can trip the second pattern)
# and false negatives (a merge phrased without "while" or ", and Xed" slips
# through). It exists purely because the prompt rule alone is model-obeyed,
# not enforced, and a real run already produced a violation this catches:
# "...processing 10k+ records while maintaining strict software architecture
# and code quality standards through rigorous pull request reviews."
NARRATIVE_MERGE_PATTERNS = [
    r"\bwhile\s+\w+ing\b",     # "...while maintaining...", "...while building..."
    r",\s*and\s+\w+ed\b",      # ", and built...", ", and mentored..." (2nd past-tense clause)
]

GRAD_MONTH_YEAR = "December 2026"
START_MONTH_YEAR = "January 2027"

# ---------------------------------------------------------------------------
# API SETUP -- multi-account rotation for free-tier quota failover
# ---------------------------------------------------------------------------
GEMINI_KEY_VARS = [
    "GEMINI_API_KEY_1", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3",
    "GEMINI_API_KEY_4", "GEMINI_API_KEY_5",
]
gemini_keys = [os.environ[v] for v in GEMINI_KEY_VARS if os.environ.get(v)]
if not gemini_keys:
    raise SystemExit(
        "ERROR: set at least GEMINI_API_KEY_1 as an environment variable "
        "(GEMINI_API_KEY_2..5 optional, for automatic quota failover)."
    )
gemini_clients = [genai.Client(api_key=k) for k in gemini_keys]
_active_key_idx = 0  # module-level: which account call_gemini() is currently using

# ---------------------------------------------------------------------------
# GROQ SETUP -- used for proofread AND for the free-tier-friendly proofread
# auto-fix pass (see fix_proofread_issues_via_groq below). Falls back to
# Gemini only if Groq can't resolve an issue after MAX_GROQ_FIX_ATTEMPTS.
# ---------------------------------------------------------------------------
from groq import Groq
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"]) if os.environ.get("GROQ_API_KEY") else None


def call_groq(prompt: str, max_tokens: int = 1024) -> str:
    if groq_client is None:
        raise RuntimeError("GROQ_API_KEY not set -- cannot use Groq for proofread fixes.")
    resp = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()

def _is_quota_exhausted(exc: Exception) -> bool:
    msg = str(exc)
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower()

def _is_transient_server_error(exc: Exception) -> bool:
    msg = str(exc)
    return "503" in msg or "UNAVAILABLE" in msg or "500" in msg or "INTERNAL" in msg


def call_gemini(prompt: str, max_transient_retries: int = 3) -> str:
    global _active_key_idx
    last_exc = None
    for attempt in range(len(gemini_clients)):
        idx = (_active_key_idx + attempt) % len(gemini_clients)
        for transient_attempt in range(max_transient_retries):
            try:
                response = gemini_clients[idx].models.generate_content(
                    model=MODEL_NAME, contents=prompt
                )
                if idx != _active_key_idx:
                    print(f"  -> switched to Gemini account #{idx + 1} "
                          f"(account #{_active_key_idx + 1} exhausted)")
                    _active_key_idx = idx
                return response.text
            except Exception as e:
                if _is_quota_exhausted(e):
                    last_exc = e
                    break
                if _is_transient_server_error(e) and transient_attempt < max_transient_retries - 1:
                    wait = 2 ** transient_attempt
                    print(f"  -> transient error ({e}), retrying in {wait}s "
                          f"({transient_attempt + 1}/{max_transient_retries})...")
                    time.sleep(wait)
                    last_exc = e
                    continue
                if _is_transient_server_error(e):
                    last_exc = e
                    break
                raise
    raise RuntimeError(
        f"All {len(gemini_clients)} account(s) failed after transient retries. "
        f"Last error: {last_exc}"
    )


# ---------------------------------------------------------------------------
# PASS A -- ATS CONTENT TAILORING
# ---------------------------------------------------------------------------
def tailor_content(jd_text: str, master_content: str) -> dict:
    project_list = "\n".join(
        f'- id: "{pid}" | {p["title"]} | keywords: {p["keywords"]}'
        for pid, p in PROJECTS.items()
    )
    prompt = f"""You are tailoring resume CONTENT ONLY. Do not write LaTeX.
Do not count characters or lines -- layout happens later, separately.

JOB DESCRIPTION:
{jd_text}

MASTER RESUME CONTENT (your only source of facts -- never invent beyond this):
{master_content}

Rules:
- Never invent a keyword match or claim a technology not in the master content.
- Never relabel a technology with a more advanced-sounding synonym (e.g. don't call a monolith "microservices", don't call a REST endpoint "distributed" unless it factually was).
- Preserve every quantitative metric -- never delete one to make room for a keyword.
- No single action verb (Built, Automated, Engineered, Developed, etc.) may appear more than 2 times across the ENTIRE resume (all Experience + Academic Project bullets combined) -- not just avoiding repeats in consecutive bullets. Vary verbs deliberately across sections.
- BANNED weak/passive opening verbs -- never start a bullet with:
  Gathered, Formulated, Processed, Worked on, Helped, Responsible for, Assisted, Supported, Handled, Involved in. These read as administrative rather than as ownership of the outcome. Prefer a verb that names what was actually built/decided/fixed (e.g. "Collected requirements from stakeholders and translated them into..." instead of "Gathered technical specifications...").
- NO CROSS-POLLINATION OF TECHNOLOGIES (CRITICAL): You may NOT inject a technology, tool, or cloud provider (e.g., "AWS", "Redis", "C++") into an Experience or Project bullet unless the Master Content explicitly states you used that exact technology in THAT specific role. Never sprinkle global keywords from the "Technical Skills" section into a job just to satisfy the JD.
- No adverb bloat ("rigorously optimized" -> "optimized").
- BANNED cliche/buzzwords -- never use: dynamic, robust, seamless, cutting-edge, innovative, world-class, passionate, results-driven, proven track record, extensive experience, self-starter, leverage (as a verb), synergy, best-in-class.
- NARRATIVE COHERENCE (hard rule, not a style preference): a bullet may contain only ONE finite verb describing ONE deliverable. If you catch yourself writing a second "and [verb]ed" or "while [verb]ing" clause that introduces a NEW deliverable (a different system, a different metric, a different activity like interviewing/mentoring), STOP -- that is two bullets' worth of content forced into one. Pick whichever deliverable is more JD-relevant and drop the other; do not compress both into one sentence.
  The only exception: a second clause is allowed if it is the causal RESULT of the first action (X, which enabled Y), never a second, independent thing that was ALSO done.
- Ground every claim in the master content. Flag any JD requirement with zero factual backing instead of fabricating support for it.
- NEVER move a fact, achievement, or piece of terminology from one Experience entry or Academic Project into a DIFFERENT one.
- When multiple JD requirements are truly unsupported, prioritize surfacing them honestly in tier1_gaps over stretching a loosely-related bullet to imply coverage that doesn't exist.

PROJECT SELECTION (new step):
Below are all 5 available academic projects. Select exactly the 3 most relevant to this JD.
{project_list}

BULLET COUNT RULE (fixed, independent of relevance): LMS > Fraud Detection (GNN) > Pintos OS > Temperature Monitoring > Music Genre Classification. Whichever of your 3 selected projects ranks HIGHEST in that fixed order gets 3 bullets. The other two each get 2 bullets.

COMPANY NAME AND JOB TITLE:
Extract the hiring company's name and the job title as they appear in the JD.

Return ONLY valid JSON, no markdown fences, no commentary, matching exactly this schema:
{{
  "company_name": "...",
  "job_title": "...",
  "keyword_audit": [{{"requirement": "...", "status": "supported|unsupported|partial", "evidence": "..."}}],
  "experience": {{
    "Graduate Student Developer": ["bullet 1", "bullet 2", "bullet 3"],
    "Software Engineer": ["bullet 1", "bullet 2", "bullet 3", "bullet 4"],
    "Cloud Engineer Intern": ["bullet 1", "bullet 2"],
    "Data Engineer Intern": ["bullet 1"]
  }},
  "skills": {{
    "Languages": "...",
    "Frameworks": "...",
    "Artificial Intelligence & Machine Learning": "...",
    "Software Engineering": "...",
    "Tools": "..."
  }},
  "academic_projects_selected": ["project_id_rank1", "project_id_rank2", "project_id_rank3"],
  "academic_projects": {{
    "project_id_rank1": ["bullet 1", "bullet 2", "bullet 3"],
    "project_id_rank2": ["bullet 1", "bullet 2"],
    "project_id_rank3": ["bullet 1", "bullet 2"]
  }},
  "tier1_gaps": ["..."]
}}
"""
    raw = call_gemini(prompt)
    raw = re.sub(r"^```json\s*|\s*```$", "", raw.strip())
    return json.loads(raw)


# ---------------------------------------------------------------------------
# PASS B -- FILL TEMPLATE
# ---------------------------------------------------------------------------
def escape_latex(text: str) -> str:
    replacements = {
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"$\sim$",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def fill_template(template: str, content: dict) -> str:
    tex = template

    exp_map = {
        "Graduate Student Developer": content["experience"]["Graduate Student Developer"],
        "Software Engineer": content["experience"]["Software Engineer"],
        "Cloud Engineer Intern": content["experience"]["Cloud Engineer Intern"],
        "Data Engineer Intern": content["experience"]["Data Engineer Intern"],
    }
    for i, bullet in enumerate(exp_map["Graduate Student Developer"], 1):
        tex = tex.replace(f"[BULLET {i} -- INSERT TAILORED CONTENT]", escape_latex(bullet), 1)
    for i, bullet in enumerate(exp_map["Software Engineer"], 1):
        tex = tex.replace(f"[BULLET {i} -- INSERT TAILORED CONTENT]", escape_latex(bullet), 1)
    for i, bullet in enumerate(exp_map["Cloud Engineer Intern"], 1):
        tex = tex.replace(f"[BULLET {i} -- INSERT TAILORED CONTENT]", escape_latex(bullet), 1)
    for i, bullet in enumerate(exp_map["Data Engineer Intern"], 1):
        tex = tex.replace(f"[BULLET {i} -- INSERT TAILORED CONTENT]", escape_latex(bullet), 1)

    skill_key_map = {
        "Languages": "[INSERT:LANGUAGES]",
        "Frameworks": "[INSERT:FRAMEWORKS]",
        "Artificial Intelligence & Machine Learning": "[INSERT:AI_ML]",
        "Software Engineering": "[INSERT:SWE]",
        "Tools": "[INSERT:TOOLS]",
    }
    for label in SKILL_CATEGORY_ORDER:
        marker = skill_key_map[label]
        value = content["skills"].get(label, "")
        tex = tex.replace(marker, escape_latex(value))

    selected = content["academic_projects_selected"]
    best_pid = min(selected, key=lambda pid: PROJECTS[pid]["complexity_rank"])
    selected_display_order = get_project_display_order(selected)
    project_blocks = []
    for pid in selected_display_order:
        if pid not in PROJECTS:
            raise ValueError(
                f"Model selected unknown project id '{pid}' -- not in PROJECTS "
                f"registry ({list(PROJECTS.keys())}). Check the raw API response."
            )
        meta = PROJECTS[pid]
        if pid not in content["academic_projects"]:
            raise ValueError(
                f"Project '{pid}' was selected but has no bullets in "
                f"academic_projects -- model returned inconsistent JSON."
            )
        bullets = content["academic_projects"][pid]
        max_bullets = 3 if pid == best_pid else 2
        if len(bullets) > max_bullets:
            bullets = bullets[:max_bullets]
        bullet_lines = "\n".join(
            f"    \\item {escape_latex(b)}" for b in bullets
        )
        block = (
            f"\\noindent\\textbf{{{meta['title']}}} $|$ {meta['context']} "
            f"\\hfill \\textit{{{meta['date']}}}\n"
            f"\\begin{{itemize}}\n"
            f"    \\setlength{{\\itemsep}}{{0.3pt}}\n"
            f"{bullet_lines}\n"
            f"\\end{{itemize}}"
        )
        project_blocks.append(block)
    academic_projects_tex = "\n\n".join(project_blocks)
    tex = tex.replace("[AUTO:ACADEMIC_PROJECTS_BLOCK]", academic_projects_tex)

    return tex


# ---------------------------------------------------------------------------
# COMPILE + PARSE REAL HBOX WARNINGS
# ---------------------------------------------------------------------------
def compile_tex(tex_path: Path) -> str:
    for _ in range(2):
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=tex_path.parent, capture_output=True, text=True,
        )
    log_path = tex_path.with_suffix(".log")
    return log_path.read_text(errors="ignore") if log_path.exists() else ""


def parse_hbox_warnings(log_text: str) -> list:
    warnings = []
    pattern = re.compile(
        r"(Overfull|Underfull) \\hbox \(([\d.]+)pt too (wide|narrow)\) .*?lines? (\d+)(?:--(\d+))?"
    )
    for match in pattern.finditer(log_text):
        kind, amount, direction, line_start, line_end = match.groups()
        warnings.append({
            "type": kind,
            "amount_pt": float(amount),
            "direction": direction,
            "line_start": int(line_start),
            "line_end": int(line_end) if line_end else int(line_start),
        })
    return warnings


FILL_MIN_PERCENT = 45
SCAN_WINDOW = 40

def normalize_words(text: str) -> list:
    text = (text.replace("\\&", "&").replace("\\%", "%").replace("\\_", "_")
                .replace("\\$", "$").replace("\\#", "#"))
    text = re.sub(r"[^\w%$#&/.+-]", " ", text.lower())
    return [w for w in text.split() if w]


def run_pdftotext(pdf_path: Path) -> list:
    result = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"],
        capture_output=True, text=True,
    )
    return [l for l in result.stdout.split("\n") if l.strip()]


def build_measurement_chunks(content: dict) -> dict:
    exp = content["experience"]
    sections = {
        "Graduate Student Developer": [(f"gsd_{i}", b) for i, b in enumerate(exp["Graduate Student Developer"])],
        "Software Engineer": [(f"se_{i}", b) for i, b in enumerate(exp["Software Engineer"])],
        "Cloud Engineer Intern": [(f"ce_{i}", b) for i, b in enumerate(exp["Cloud Engineer Intern"])],
        "Data Engineer Intern": [(f"de_{i}", b) for i, b in enumerate(exp["Data Engineer Intern"])],
        "Technical Skills": [
            (f"skill_{k}", f"{k}: {content['skills'].get(k, '')}")
            for k in SKILL_CATEGORY_ORDER
        ],
        "Academic Projects": [],
    }
    for pid in get_project_display_order(content["academic_projects_selected"]):
        for i, b in enumerate(content["academic_projects"][pid]):
            sections["Academic Projects"].append((f"{pid}_{i}", b))
    return sections


def find_chunk_start(text_lines: list, line_ptr: int, first_words: list):
    min_needed = min(2, len(first_words))
    limit = min(len(text_lines), line_ptr + SCAN_WINDOW)
    best_idx, best_score = None, 0
    for idx in range(line_ptr, limit):
        candidate = normalize_words(text_lines[idx])
        overlap = sum(1 for w in first_words if w in candidate[:6])
        if overlap > best_score:
            best_score, best_idx = overlap, idx
        if overlap == len(first_words) and overlap > 0:
            break
    if best_score >= min_needed:
        return best_idx, best_score
    return None, 0


def measure_fill(pdf_path: Path, content: dict, verbose: bool = True) -> dict:
    text_lines = run_pdftotext(pdf_path)
    sections = build_measurement_chunks(content)

    line_ptr = 0
    all_chunk_lines = {}
    chunk_section = {}
    section_line_counts = {}
    unmatched_chunks = []

    for section_name, chunks in sections.items():
        total_lines_this_section = 0
        for chunk_id, chunk_text in chunks:
            chunk_section[chunk_id] = section_name
            target_words = normalize_words(chunk_text)
            if not target_words:
                continue
            first_words = target_words[:3]

            start_idx, score = find_chunk_start(text_lines, line_ptr, first_words)
            if start_idx is None:
                unmatched_chunks.append({
                    "chunk_id": chunk_id,
                    "section": section_name,
                    "expected_first_words": first_words,
                    "searched_from_line": line_ptr,
                })
                if verbose:
                    print(f"      !! no confident match for chunk '{chunk_id}' "
                          f"(section: {section_name}); searched lines "
                          f"{line_ptr}-{min(len(text_lines), line_ptr + SCAN_WINDOW) - 1}; "
                          f"expected first words: {first_words}")
                continue

            line_ptr = start_idx
            consumed = 0
            lines_used = []
            while consumed < len(target_words) and line_ptr < len(text_lines):
                line = text_lines[line_ptr]
                lw = normalize_words(line)
                consumed += len(lw)
                lines_used.append(line.strip())
                line_ptr += 1

            all_chunk_lines[chunk_id] = lines_used
            total_lines_this_section += len(lines_used)

        section_line_counts[section_name] = total_lines_this_section

    sections_with_unmatched = {u["section"] for u in unmatched_chunks}

    full_line_samples = [
        len(lines[i]) for lines in all_chunk_lines.values()
        for i in range(len(lines) - 1)
    ]
    reference_width = max(full_line_samples) if full_line_samples else 100

    section_report = {
        name: {
            "actual_lines": section_line_counts.get(name, 0),
            "target_lines": LINE_TARGETS.get(name),
            "reliable": name not in sections_with_unmatched,
        }
        for name in sections
    }

    sparse_bullets = []
    for chunk_id, lines in all_chunk_lines.items():
        if not lines:
            continue
        last_line_len = len(lines[-1])
        fill_pct = round(100 * last_line_len / reference_width, 1) if len(lines) > 1 else 100.0
        if len(lines) > 1 and fill_pct < FILL_MIN_PERCENT:
            section_name = chunk_section.get(chunk_id)
            info = section_report.get(section_name, {})
            target = info.get("target_lines")
            actual = info.get("actual_lines")
            if target is not None and actual is not None and actual < target:
                direction = "expand"
            elif target is not None and actual is not None and actual > target:
                direction = "do_not_expand_shrink_elsewhere_instead"
            else:
                direction = "expand"
            target_min_chars = round(0.5 * reference_width)
            target_max_chars = round(0.7 * reference_width)
            chars_to_add = max(0, target_min_chars - last_line_len)
            sparse_bullets.append({
                "chunk_id": chunk_id,
                "section": section_name,
                "last_line_text": lines[-1],
                "last_line_chars": last_line_len,
                "fill_pct": fill_pct,
                "target_chars_range": [target_min_chars, target_max_chars],
                "approx_chars_to_add": chars_to_add,
                "recommended_action": direction,
            })

    return {
        "sections": section_report,
        "sparse_bullets": sparse_bullets,
        "overflow_sections": build_overflow_report(sections, section_report, all_chunk_lines, reference_width),
        "unmatched_chunks": unmatched_chunks,
        "all_chunk_lines": all_chunk_lines,
        "chunk_section": chunk_section,
        "reference_width": reference_width,
    }


def build_overflow_report(sections: dict, section_report: dict, all_chunk_lines: dict, reference_width: int) -> list:
    overflow = []
    for section_name, chunks in sections.items():
        info = section_report.get(section_name, {})
        if not info.get("reliable", True):
            continue
        target = info.get("target_lines")
        actual = info.get("actual_lines")
        if target is None or actual is None or actual <= target:
            continue
        excess_lines = actual - target
        candidates = []
        for chunk_id, _chunk_text in chunks:
            lines = all_chunk_lines.get(chunk_id, [])
            if not lines:
                continue
            total_chars = sum(len(l) for l in lines)
            candidates.append({
                "chunk_id": chunk_id,
                "current_line_count": len(lines),
                "current_total_chars": total_chars,
            })
        candidates.sort(key=lambda c: -c["current_total_chars"])
        overflow.append({
            "section": section_name,
            "excess_lines": excess_lines,
            "approx_chars_to_remove_total": excess_lines * reference_width,
            "shrink_candidates_ranked": candidates[:3],
        })
    return overflow


def force_page_overflow_target(fill_report: dict) -> list:
    all_chunk_lines = fill_report["all_chunk_lines"]
    chunk_section = fill_report["chunk_section"]
    reference_width = fill_report["reference_width"]
    section_report = fill_report["sections"]

    reliable = {name: info for name, info in section_report.items() if info.get("reliable", True)}
    if not reliable:
        return []

    def section_total_chars(name):
        return sum(
            sum(len(l) for l in lines)
            for cid, lines in all_chunk_lines.items()
            if chunk_section.get(cid) == name
        )

    target_section = max(reliable, key=lambda name: (reliable[name]["actual_lines"], section_total_chars(name)))

    candidates = []
    for chunk_id, lines in all_chunk_lines.items():
        if chunk_section.get(chunk_id) != target_section:
            continue
        candidates.append({
            "chunk_id": chunk_id,
            "current_line_count": len(lines),
            "current_total_chars": sum(len(l) for l in lines),
        })
    candidates.sort(key=lambda c: -c["current_total_chars"])

    return [{
        "section": target_section,
        "excess_lines": 1,
        "approx_chars_to_remove_total": reference_width,
        "shrink_candidates_ranked": candidates[:3],
        "note": "synthetic: page count > 1 but no section individually over "
                "target -- targeting the longest reliable section as the "
                "likely source of the extra page.",
    }]


# ---------------------------------------------------------------------------
# GUARDRAILS: unsupported terms, unverified terms, verb repetition,
# narrative coherence
# ---------------------------------------------------------------------------
def check_unsupported_terms(content: dict, master_content: str) -> dict:
    """
    Hardcoded, code-level enforcement of the "## Explicitly Unsupported"
    block in Master_Resume_Content.md.

    Returns {"skills_hits": [{"term":..., "category":...}, ...],
             "bullet_hits": [{"term":..., "location":..., "text":...}, ...]}
    instead of a flat list, because the two are handled differently
    downstream: a hit inside Technical Skills (a flat comma-separated list)
    can be safely auto-removed by strip_unsupported_skills() without
    damaging grammar. A hit inside a bullet SENTENCE cannot be safely
    auto-edited by string removal -- it's surfaced as a required rewrite
    for fix_layout() / a fresh tailor_content() pass instead.

    IMPORTANT: this function itself was verified correct against the
    "CI/CD (no direct evidence...)" case -- the original bug was NOT here,
    it was that process_job() only ever called this once, before the
    layout-fix retry loop, so a term introduced by a later fix_layout()
    pass was never checked. See process_job() and the module CHANGELOG.
    """
    match = re.search(
        r"## Explicitly Unsupported(.*?)(?=\n## |\Z)", master_content, re.DOTALL
    )
    if not match:
        print("  !! DEBUG check_unsupported_terms: no '## Explicitly Unsupported' section found in master content")
        return {"skills_hits": [], "bullet_hits": []}
    block = match.group(1)

    block_no_parens = re.sub(r"\([^)]*\)", "", block)
    raw_terms = re.split(r"[,\n]", block_no_parens)
    blocked_terms = []
    for t in raw_terms:
        term = t.strip().rstrip(".").strip()
        if not term or len(term) <= 2:
            continue
        if term.lower().startswith(("note", "explicitly", "never", "unsupported")):
            continue
        # A blocked term is always a short token/phrase from a comma list.
        # Anything longer than ~4 words is a run-on sentence fragment that
        # leaked through the comma/newline split (e.g. a wrapped NOTE
        # sentence, or the "---" divider) -- never a real blocked term, and
        # letting it through just wastes a regex search or, worse, could
        # accidentally match unrelated prose.
        if len(term.split()) > 4:
            continue
        blocked_terms.append(term)

    print(f"  -> DEBUG check_unsupported_terms: blocked_terms = {blocked_terms}")

    def make_pattern(term: str) -> str:
        pattern = re.escape(term)
        if term[0].isalnum():
            pattern = r"\b" + pattern
        if term[-1].isalnum():
            pattern = pattern + r"\b"
        return pattern

    skills_hits = []
    for category, value in content.get("skills", {}).items():
        for term in blocked_terms:
            if re.search(make_pattern(term), value, re.IGNORECASE):
                skills_hits.append({"term": term, "category": category})

    bullet_sources = []
    for role, bullets in content.get("experience", {}).items():
        for i, b in enumerate(bullets):
            bullet_sources.append((f"experience:{role}:{i}", b))
    for pid, bullets in content.get("academic_projects", {}).items():
        for i, b in enumerate(bullets):
            bullet_sources.append((f"academic_projects:{pid}:{i}", b))

    bullet_hits = []
    for location, text in bullet_sources:
        for term in blocked_terms:
            if re.search(make_pattern(term), text, re.IGNORECASE):
                bullet_hits.append({"term": term, "location": location, "text": text})

    return {"skills_hits": skills_hits, "bullet_hits": bullet_hits}


def strip_unsupported_skills(content: dict, skills_hits: list) -> dict:
    """
    Deterministically removes banned terms found in Technical Skills.
    Safe to do mechanically because skills is a flat comma-separated list --
    unlike a bullet sentence, dropping one item can't break grammar or leave
    a dangling clause. Runs IN-PLACE on content["skills"] and also returns
    it for convenience. Bullet-level hits are intentionally NOT auto-edited
    here (see check_unsupported_terms docstring) -- the caller must route
    those to fix_layout() or treat them as a hard needs_review instead.
    """
    for hit in skills_hits:
        category, term = hit["category"], hit["term"]
        value = content["skills"].get(category, "")
        pattern = r"\b" + re.escape(term) + r"\b"
        parts = split_skill_terms(value)
        kept = [p for p in parts if p and not re.search(pattern, p, re.IGNORECASE)]
        content["skills"][category] = ", ".join(kept)
        print(f"  -> auto-stripped banned term '{term}' from skills[{category}]")
    return content


def check_narrative_coherence(content: dict) -> list:
    """
    Regex-level backstop for the "one deliverable per bullet" prompt rule
    (see NARRATIVE_MERGE_PATTERNS module constant for why this is a
    heuristic, not a parser, and will have both false positives and false
    negatives). Returns a list of hits to feed into fix_layout() and into
    the layout_clean determination -- never auto-edits a bullet itself,
    since splitting a merged sentence back into "pick one deliverable"
    is a judgment call, not a mechanical string operation.
    """
    hits = []
    bullet_sources = []
    for role, bullets in content.get("experience", {}).items():
        for i, b in enumerate(bullets):
            bullet_sources.append((f"experience:{role}:{i}", b))
    for pid, bullets in content.get("academic_projects", {}).items():
        for i, b in enumerate(bullets):
            bullet_sources.append((f"academic_projects:{pid}:{i}", b))

    for location, text in bullet_sources:
        for pattern in NARRATIVE_MERGE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                hits.append({"location": location, "text": text, "matched_pattern": pattern})
                break
    return hits

def locate_proofread_snippet(snippet: str, content: dict):
    """
    Maps a Groq-flagged snippet back to the exact bullet it came from in
    content.json, so a fix call can target just that one bullet instead of
    guessing. Uses substring/word-overlap matching, not exact match,
    because pdftotext line-wrapping can shift whitespace slightly from the
    source JSON text. Returns None if no bullet can be confidently
    matched -- that issue is then left for manual review instead of guessed at.
    """
    def norm(s):
        return re.sub(r"\s+", " ", s.lower()).strip()

    snippet_norm = norm(snippet)
    sources = []
    for role, bullets in content.get("experience", {}).items():
        for i, b in enumerate(bullets):
            sources.append({"location": f"experience:{role}:{i}", "kind": "experience",
                             "key": role, "index": i, "text": b})
    for category, value in content.get("skills", {}).items():
        sources.append({"location": f"skills:{category}", "kind": "skills",
                         "key": category, "index": None, "text": value})

    for src in sources:
        text_norm = norm(src["text"])
        if snippet_norm[:25] in text_norm or text_norm[:25] in snippet_norm:
            return src

    best, best_overlap = None, 0
    snippet_words = set(snippet_norm.split())
    for src in sources:
        overlap = len(snippet_words & set(norm(src["text"]).split()))
        if overlap > best_overlap:
            best_overlap, best = overlap, src
    return best if best_overlap >= 3 else None
    
def split_skill_terms(value: str) -> list:
    """
    Splits a Technical Skills comma-separated string into individual terms,
    while treating a parenthetical group like "Graph Neural Networks
    (GraphSAGE, GAT)" as ONE term rather than splitting on the comma
    inside the parentheses. A naive value.split(",") mangles that into two
    broken fragments -- "Graph Neural Networks (GraphSAGE" and " GAT)" --
    and the first fragment then fails the unverified-term check (it's not
    a real phrase, it's a parsing artifact) and gets silently stripped,
    deleting legitimate, verified content along with it. Confirmed real
    case: this exact fragment appeared in a live run and cost 2 lines of
    Technical Skills that the retry loop then couldn't recover.
    """
    parts = re.split(r',\s*(?![^(]*\))', value)
    return [p.strip() for p in parts if p.strip()]
    
def flag_unverified_terms(content: dict, master_content: str) -> list:
    """
    Heuristic: flags Technical Skills terms with no basis anywhere in
    master_content. Checks each comma-separated sub-term individually and
    normalizes case/whitespace/punctuation before comparing. Also does a
    light suffix-fold (plurals, "-ing"/"-ed"/"-ical"/"-ization" endings) on
    the WORD-LEVEL fallback check, so a phrasing difference like "Dynamic
    SQL Querying" (output) vs. "dynamic queries" (master) or "Relational
    Database Design" vs. "relational databases" doesn't get flagged as
    unverified just because the exact inflection differs -- a genuinely new
    term like "CI/CD Pipelines" still won't have all its stemmed words
    present, so it still gets flagged correctly.
    """
    def normalize(s: str) -> str:
        s = s.lower()
        s = re.sub(r"[^\w\s]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def stem(word: str) -> str:
        for suffix in ("ization", "ications", "ication", "ical", "ing", "ies", "es", "ed", "s"):
            if len(word) > len(suffix) + 2 and word.endswith(suffix):
                return word[: -len(suffix)]
        return word

    master_norm = normalize(master_content)
    # Sub-phrase scoped, not just line-scoped. Master content lines are
    # frequently comma-separated tech lists, and a bag-of-words match
    # across the WHOLE line lets two unrelated list items combine into a
    # claim that was never made. Splitting on commas/semicolons too, and
    # requiring all term words to co-occur within the SAME sub-phrase,
    # closes that gap.
    raw_subphrases = re.split(r"[\n,;]", master_content)
    master_lines_norm = [normalize(p) for p in raw_subphrases if p.strip()]
    master_line_stems = [set(line.split()) for line in master_lines_norm]
    master_stems = {stem(w) for w in master_norm.split()}

    flagged = []
    for category, value in content.get("skills", {}).items():
        terms = split_skill_terms(value)
        for term in terms:
            paren_match = re.search(r"\(([^)]*)\)", term)
            bare = re.sub(r"\s*\([^)]*\)", "", term).strip()
            if len(bare) < 3:
                continue
            bare_norm = normalize(bare)
            if bare_norm in master_norm:
                continue
            # NEW: if the output expands an abbreviation that master content
            # only ever states in short form (e.g. output "Object-Oriented
            # Programming (OOP)" where master content just says "OOP"),
            # the parenthetical abbreviation IS the verified claim -- don't
            # flag the expanded label as unverified.
            if paren_match:
                abbrev_norm = normalize(paren_match.group(1))
                if abbrev_norm and re.search(r"\b" + re.escape(abbrev_norm) + r"\b", master_norm):
                    continue
            words = bare_norm.split()
            if words and any(
                all(re.search(r"\b" + re.escape(w) + r"\b", line) for w in words)
                for line in master_lines_norm
            ):
                continue
            if words and any(
                all(stem(w) in {stem(x) for x in line_words} for w in words)
                for line_words in master_line_stems
            ):
                continue
            flagged.append({"term": bare, "category": category})
    return flagged


def check_verb_repetition(content: dict) -> list:
    all_bullets = []
    for bullets in content["experience"].values():
        all_bullets.extend(bullets)
    for pid in content["academic_projects_selected"]:
        all_bullets.extend(content["academic_projects"].get(pid, []))

    counts = {}
    samples = {}
    for text in all_bullets:
        words_in_bullet = set(re.findall(r"[a-zA-Z]+", text.lower()))
        for verb in ACTION_VERB_LIST:
            if verb in words_in_bullet:
                counts[verb] = counts.get(verb, 0) + 1
                samples.setdefault(verb, []).append(text)

    return [
        {"verb": verb, "count": count, "bullets_using_it": samples[verb]}
        for verb, count in counts.items() if count > MAX_VERB_REPEAT
    ]


# ---------------------------------------------------------------------------
# PASS C -- TARGETED LAYOUT CORRECTION
# ---------------------------------------------------------------------------
def fix_layout(content: dict, hbox_warnings: list, fill_report: dict, verb_report: list,
                jd_text: str, unsupported_bullet_hits: list = None,
                narrative_issues: list = None) -> dict:
    unsupported_bullet_hits = unsupported_bullet_hits or []
    narrative_issues = narrative_issues or []

    section_issues = {
        name: info for name, info in fill_report["sections"].items()
        if info["target_lines"] is not None
        and info["actual_lines"] != info["target_lines"]
        and info.get("reliable", True)
    }
    prompt = f"""The compiled PDF has REAL, measured layout problems (from
actually parsing the rendered PDF text -- this is ground truth, not an
estimate).

SECTION LINE-COUNT MISMATCHES (actual vs target, measured from the real PDF):
{json.dumps(section_issues, indent=2)}

UNMATCHED CHUNKS:
{json.dumps(fill_report["unmatched_chunks"], indent=2)}

SPARSE ORPHAN LAST LINES:
{json.dumps(fill_report["sparse_bullets"], indent=2)}

SECTIONS OVER THEIR LINE-COUNT TARGET:
{json.dumps(fill_report["overflow_sections"], indent=2)}

OVERUSED ACTION VERBS (appearing more than {MAX_VERB_REPEAT} times):
{json.dumps(verb_report, indent=2)}

EXPLICITLY-UNSUPPORTED TERMS FOUND INSIDE BULLET SENTENCES (CRITICAL, MUST FIX):
{json.dumps(unsupported_bullet_hits, indent=2)}
Each entry names a banned technology/term that appears inside a bullet's
prose (not the Technical Skills list -- that gets auto-stripped
separately). You MUST rewrite each implicated bullet to remove the banned
term entirely, using ONLY facts already present in the master content for
that same role/project. Do not substitute a different unverified
technology in its place. If the bullet cannot be rewritten to remove the
term without fabricating a replacement fact, strip it down to only the
facts you can verify and add the location to "flagged".

NARRATIVE-COHERENCE VIOLATIONS (two deliverables merged into one bullet --
MUST FIX):
{json.dumps(narrative_issues, indent=2)}
Each entry is a bullet that stitches two separate accomplishments together
with "while doing X" or ", and did Y". Rewrite it to keep ONLY the single
most JD-relevant deliverable; drop the other clause entirely rather than
compressing both into one sentence. If genuinely unsure whether a flagged
clause is a real second deliverable versus a legitimate causal result (X,
which enabled Y), leave it as-is and add it to "flagged" for a human to
judge -- don't guess.

RAW LATEX OVERFULL/UNDERFULL WARNINGS:
{json.dumps(hbox_warnings, indent=2)}

Current tailored content (JSON):
{json.dumps(content, indent=2)}

Job description (for keyword priority if trimming):
{jd_text}

Fix ONLY the bullets implicated above. Do not touch any other bullet.
NEVER solve a character-count/line-count problem by merging two
deliverables into one bullet with "and"/"while". Never delete a
quantitative metric or a previously-included Tier 1 keyword to fix layout.
Do not change "academic_projects_selected". Only edit bullet text (and
Technical Skills text if a skills-related issue is listed above).

TECHNICAL SKILLS -- SPECIAL RULE (CRITICAL): if a sparse-orphan or
line-count issue names a "skill_*" chunk, this is a LIST-ITEM-ORDERING
problem, not a content gap. Do NOT fix it by inventing or adding a new
skill term to that category -- any term not already verbatim in the
master content gets automatically stripped back out on the next pass,
which just recreates the same sparse line again. Instead: reorder the
EXISTING comma-separated items so a longer one lands last, or reword an
EXISTING item using a form already used elsewhere in the master content.
If neither gets you to the target length, leave that category's line
count as-is and add the chunk_id to "flagged" instead of adding an
unverified term.

Return the FULL corrected JSON in the exact same schema as before (including
academic_projects_selected unchanged), with one added top-level key:
"flagged": ["..."] (empty list if nothing flagged).
Return ONLY valid JSON, no markdown fences, no commentary.
"""

    raw = call_gemini(prompt)
    raw = re.sub(r"^```json\s*|\s*```$", "", raw.strip())
    return json.loads(raw)


# ---------------------------------------------------------------------------
# PROOFREAD AUTO-FIX -- Groq first (free), Gemini as last-resort fallback
# ---------------------------------------------------------------------------
CORE_WRITING_RULES = """
Rules for this edit:
- Never invent a fact, technology, or number not present in the master content below.
- Preserve every quantitative metric exactly as written -- never delete or alter one.
- A bullet may describe only ONE deliverable with ONE finite verb -- never merge two
  accomplishments with "while X-ing" or ", and X-ed".
- No banned buzzwords: dynamic, robust, seamless, cutting-edge, innovative, world-class,
  passionate, results-driven, proven track record, extensive experience, self-starter,
  leverage (as a verb), synergy, best-in-class.
- This is a CORRECTION, not a rewrite: keep the original meaning, structure, and length
  as close to the original as possible. Fix only what's broken.
"""


def groq_fix_bullet(bullet_text: str, problem: str, master_content: str) -> str:
    prompt = f"""A resume bullet has a proofreading issue.

BULLET TEXT:
{bullet_text}

ISSUE FOUND:
{problem}

MASTER RESUME CONTENT (ground truth):
{master_content}

{CORE_WRITING_RULES}

Return ONLY the corrected bullet text as plain text. No JSON, no quotes,
no markdown, no leading bullet character, no commentary.
"""
    return call_groq(prompt).strip().strip('"').strip()


def gemini_fix_bullet(bullet_text: str, problem: str, master_content: str) -> str:
    prompt = f"""A resume bullet has a proofreading issue flagged by an independent proofreader.

BULLET TEXT:
{bullet_text}

ISSUE FOUND:
{problem}

MASTER RESUME CONTENT (ground truth -- never add a fact not present here):
{master_content}

{CORE_WRITING_RULES}

Return ONLY the corrected bullet text as plain text. No JSON, no quotes,
no markdown, no leading bullet character, no commentary.
"""
    raw = call_gemini(prompt)
    return raw.strip().strip('"').strip()


def fix_proofread_issues(content: dict, issues: list, master_content: str, use_groq: bool = True):
    """
    Applies a surgical single-bullet fix to every located Groq finding, in
    place. use_groq=True (default) uses the free-tier Groq call;
    use_groq=False uses Gemini, reserved as the fallback when Groq's fix
    doesn't hold up after MAX_GROQ_FIX_ATTEMPTS. Returns (content,
    unresolved) -- unresolved holds any issue whose source bullet couldn't
    be confidently located, left untouched rather than guessed at.
    """
    unresolved = []
    fixer = groq_fix_bullet if use_groq else gemini_fix_bullet
    engine_label = "Groq" if use_groq else "Gemini"
    for issue in issues:
        snippet = issue.get("snippet", "")
        problem = issue.get("problem", "")
        located = locate_proofread_snippet(snippet, content)
        if not located:
            print(f"  !! could not locate source bullet for proofread issue: "
                  f"\"{snippet}\" -- leaving for manual review.")
            unresolved.append(issue)
            continue

        print(f"  -> [{engine_label}] fixing proofread issue in {located['location']}: {problem}")
        fixed_text = fixer(located["text"], problem, master_content)
        if located["kind"] == "skills":
            # Skills categories are a flat string, not a list of bullets --
            # no index to assign into, just overwrite the whole category.
            content["skills"][located["key"]] = fixed_text
        else:
            content[located["kind"]][located["key"]][located["index"]] = fixed_text

    return content, unresolved


# ---------------------------------------------------------------------------
# PASS E -- COVER LETTER
# ---------------------------------------------------------------------------

def detect_employment_type(jd_text: str) -> str:
    signals = re.findall(r"\b(intern|internship|co-?op)\b", jd_text, re.IGNORECASE)
    return "internship" if signals else "full_time"


def write_cover_letter(jd_text: str, content: dict) -> str:
    employment_type = detect_employment_type(jd_text)

    if employment_type == "internship":
        close_instruction = f"""
- Mandatory close: state current standing as a Master's student in Computer
  Science at University at Buffalo, expected graduation {GRAD_MONTH_YEAR},
  and availability for the internship's stated dates. Do NOT claim
  full-time availability or mention OPT -- this is an internship, not a
  full-time hire.
- If the JD states a target graduation year or class year that conflicts
  with {GRAD_MONTH_YEAR}, do not paper over it or imply alignment that
  isn't there.
"""
    else:
        close_instruction = f"""
- Mandatory close: state {GRAD_MONTH_YEAR} graduation and availability for
  full-time employment beginning {START_MONTH_YEAR} under OPT (Optional
  Practical Training). Do not mention CPT.
"""

    prompt = f"""Using the tailored resume content and job description below,
write a cover letter.

TAILORED CONTENT:
{json.dumps(content, indent=2)}

JOB DESCRIPTION:
{jd_text}

Rules:
- 3-4 paragraphs, under 350 words.
- Confident, technically direct, warm, mission-driven. No "synergy"/"dynamic" buzzwords.
- No em dashes or double/triple hyphens as sentence connectors anywhere in the letter.
- Connect the company's biggest technical need to 1-2 specific projects/roles from the tailored content.
{close_instruction}
- Mention the Master's in Computer Science; only if the JD is AI/ML-focused, note relevant coursework/project work in agentic AI -- do not call it a "capstone."
- Plain text output only, no markdown formatting, no subject line.
- If the tailored content's keyword_audit shows major unsupported Tier-1 requirements (e.g. a large required-years-of-experience gap, or a core required skill marked unsupported), do not write around this with confident generalities -- keep claims scoped to what's actually backed by the tailored content, and do not imply seniority or domain experience beyond what's in the master content.
"""
    raw = call_gemini(prompt)

    has_close = bool(re.search(
        r"(graduat|OPT|Optional Practical Training|internship|available)",
        raw, re.IGNORECASE
    ))
    if not has_close:
        raw += ("\n\n[GENERATION WARNING: mandatory close appears to be "
                "missing -- review before sending.]")
    return raw


# ---------------------------------------------------------------------------
# OUTPUT NAMING + SKIP-ALREADY-DONE TRACKING
# ---------------------------------------------------------------------------
def sanitize_for_filename(text: str) -> str:
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "_", text.strip())
    return text[:60] or "Unknown"


def unique_output_name(base_name: str) -> str:
    out_dir = Path(OUTPUT_DIR) / base_name
    if not out_dir.exists():
        return base_name
    suffix = 2
    while (Path(OUTPUT_DIR) / f"{base_name}_{suffix}").exists():
        suffix += 1
    return f"{base_name}_{suffix}"


def load_manifest() -> dict:
    path = Path(MANIFEST_PATH)
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_manifest(manifest: dict):
    Path(MANIFEST_PATH).write_text(json.dumps(manifest, indent=2))


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------
def process_job(jd_path: Path, master_content: str, template: str):
    """
    Returns (output_folder_name, layout_clean) on success, or (None, False)
    if the job could not be completed at all.
    """
    input_name = jd_path.name
    print(f"\n=== {input_name} ===")
    jd_text = jd_path.read_text()

    print("  -> tailoring content (isolated call)...")

    content = tailor_content(jd_text, master_content)
    content["skills"] = apply_jd_triggered_skills(jd_text, content["skills"])

    # Early guardrail pass, right after generation and before the first
    # compile: strip any banned term that landed in Technical Skills so the
    # VERY FIRST compiled PDF is already clean.
    unsupported = check_unsupported_terms(content, master_content)
    if unsupported["skills_hits"]:
        content = strip_unsupported_skills(content, unsupported["skills_hits"])
    if unsupported["bullet_hits"]:
        print(f"  !! VIOLATION: explicitly-unsupported term(s) present in bullet text: "
              f"{[h['term'] for h in unsupported['bullet_hits']]} -- will be sent to the "
              f"layout-fix pass for a mandatory rewrite.")

    unverified_terms = flag_unverified_terms(content, master_content)
    if unverified_terms:
        print(f"  -> auto-stripping unverified skill term(s) not found in master content: "
              f"{[t['term'] for t in unverified_terms]}")
        content = strip_unsupported_skills(content, unverified_terms)

    company = sanitize_for_filename(content.get("company_name", "Unknown_Company"))
    title = sanitize_for_filename(content.get("job_title", "Unknown_Role"))
    base_name = unique_output_name(f"{title}_{company}")
    print(f"  -> output will be named: {base_name}")

    out_dir = Path(OUTPUT_DIR) / base_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "jd_text.txt").write_text(jd_text)
    if content.get("tier1_gaps"):
        print(f"  !! Tier 1 gaps (unsupported by master content): {content['tier1_gaps']}")

    tex_out = out_dir / f"{base_name}.tex"
    layout_clean = False

    for attempt in range(MAX_LAYOUT_RETRIES + 1):
        skills_were_stripped = False

        tex_filled = fill_template(template, content)

        leftover = re.findall(r"\[(?:INSERT[^\]]*|BULLET \d+[^\]]*)\]", tex_filled)
        if leftover:
            print(f"  !! UNFILLED PLACEHOLDER(S) DETECTED: {set(leftover)}")
            print("  !! This job's PDF will contain literal placeholder text if compiled. Skipping compile.")
            (out_dir / f"{base_name}_INCOMPLETE.tex").write_text(tex_filled)
            print(f"  !! Partial file saved for inspection: {out_dir}/{base_name}_INCOMPLETE.tex")
            return None, False

        tex_out.write_text(tex_filled)
        (out_dir / "content.json").write_text(json.dumps(content, indent=2))
        print(f"  -> compiling (attempt {attempt + 1})...")
        log_text = compile_tex(tex_out)
        hbox_warnings = parse_hbox_warnings(log_text)

        pdf_path = tex_out.with_suffix(".pdf")
        page_report = check_page_count(pdf_path)
        fill_report = measure_fill(pdf_path, content)

        section_mismatches = [
            name for name, info in fill_report["sections"].items()
            if info["target_lines"] is not None
            and info["actual_lines"] != info["target_lines"]
            and info.get("reliable", True)
        ]

        unmatched = fill_report["unmatched_chunks"]
        sparse = fill_report["sparse_bullets"]
        verb_report = check_verb_repetition(content)

        # Re-run unsupported/unverified/narrative checks on EVERY retry,
        # because fix_layout() can rewrite Technical Skills and bullets.
        unsupported = check_unsupported_terms(content, master_content)
        if unsupported["skills_hits"]:
            print(f"  -> auto-stripping banned term(s) from Technical Skills: "
                  f"{[h['term'] for h in unsupported['skills_hits']]}")
            content = strip_unsupported_skills(content, unsupported["skills_hits"])
            unsupported["skills_hits"] = []
            skills_were_stripped = True

        unverified_terms = flag_unverified_terms(content, master_content)
        if unverified_terms:
            print(f"  -> auto-stripping unverified skill term(s) still present: "
                  f"{[t['term'] for t in unverified_terms]}")
            content = strip_unsupported_skills(content, unverified_terms)
            skills_were_stripped = True

        narrative_issues = check_narrative_coherence(content)

        # The checks above can modify Technical Skills. If they did, the
        # current PDF no longer exactly matches content.json, so force a
        # recompile on this iteration rather than declaring the PDF clean.
        if skills_were_stripped:
            (out_dir / "content.json").write_text(json.dumps(content, indent=2))

        print(f"  -> section line counts: {fill_report['sections']}")
        print(f"  -> page count: {page_report['page_count']}")
        if unmatched:
            print(f"  -> {len(unmatched)} unmatched chunk(s) -- line counts for "
                  f"{sorted({u['section'] for u in unmatched})} are unreliable this pass: "
                  f"{[u['chunk_id'] for u in unmatched]}")
        if sparse:
            print(f"  -> {len(sparse)} sparse orphan last-line(s): {[s['chunk_id'] for s in sparse]}")
        if unsupported["bullet_hits"]:
            print(f"  -> {len(unsupported['bullet_hits'])} unsupported term(s) still present in bullet text: "
                  f"{[h['term'] for h in unsupported['bullet_hits']]}")
        if narrative_issues:
            print(f"  -> {len(narrative_issues)} narrative-coherence violation(s): "
                  f"{[n['location'] for n in narrative_issues]}")

        if not page_report["one_page"] and not fill_report["overflow_sections"]:
            print(f"  -> PDF is {page_report['page_count']} pages but no section individually "
                  f"over target -- synthesizing a shrink target for the longest section.")
            fill_report["overflow_sections"] = force_page_overflow_target(fill_report)

        if fill_report["overflow_sections"]:
            print(f"  -> overflow sections needing trim: "
                  f"{[(o['section'], o['excess_lines']) for o in fill_report['overflow_sections']]}")
        if verb_report:
            print(f"  -> overused action verb(s): "
                  f"{[(v['verb'], v['count']) for v in verb_report]}")

        if (not hbox_warnings and not section_mismatches and not sparse and not verb_report
                and not unmatched and page_report["one_page"]
                and not unsupported["bullet_hits"] and not narrative_issues
                and not skills_were_stripped):
            print("  -> layout clean: no hbox warnings, all sections hit target line counts, "
                  "no sparse orphans, no overused verbs, every chunk matched, PDF is 1 page, "
                  "no unsupported/unverified terms, no narrative-coherence violations.")
            layout_clean = True
            break

        if attempt == MAX_LAYOUT_RETRIES:
            print(f"  !! layout/verb/page/guardrail issues remain after {MAX_LAYOUT_RETRIES} retries -- review manually.")
            break

        only_issue_was_skill_strip = (
            skills_were_stripped and not hbox_warnings and not section_mismatches
            and not sparse and not verb_report and not unmatched and page_report["one_page"]
            and not unsupported["bullet_hits"] and not narrative_issues
        )
        if only_issue_was_skill_strip:
            print("  -> only issue this iteration was a deterministic skills-list strip -- "
                  "recompiling with corrected skills, skipping Gemini fix_layout() call.")
            continue

        # QUOTA GUARD: if fix_layout() already told us on a PRIOR attempt
        # that a chunk can't be fixed without inventing an unverified term
        # or cutting a real metric ("flagged"), and that same chunk is
        # STILL the only thing wrong, asking again just spends another
        # Gemini call to get the same honest "can't do this" answer. Stop
        # here and accept the minor cosmetic imperfection instead --
        # confirmed real case: 'skill_Artificial Intelligence & Machine
        # Learning' got flagged on attempt 3 and then re-asked (for free,
        # on the free tier) on attempts 4, 5, AND 6 with zero change.
        previously_flagged = set(content.get("flagged", []))
        remaining_issue_ids = (
            {s["chunk_id"] for s in sparse}
            | {u["chunk_id"] for u in unmatched}
            | {o["section"] for o in fill_report["overflow_sections"]}
        )
        still_stuck_on_flagged = previously_flagged & remaining_issue_ids

        # Only skip the Gemini call if the previously-flagged chunk(s) are
        # the ENTIRE remaining problem -- not just A problem. Confirmed
        # regression: this used to fire even while OTHER unrelated, fully
        # fixable issues were still open (a real 'built' x3 verb-repeat
        # violation, and 3 sections genuinely short of their line target),
        # silently discarding fixable problems along with the one
        # legitimately-unfixable chunk. Now requires the flagged chunk(s)
        # to account for ALL open issues before skipping.
        only_issue_is_previously_flagged = (
            still_stuck_on_flagged
            and remaining_issue_ids <= previously_flagged
            and not section_mismatches
            and not verb_report
            and not hbox_warnings
            and page_report["one_page"]
        )
        if only_issue_is_previously_flagged:
            print(f"  -> {still_stuck_on_flagged} already flagged unfixable on a prior attempt, "
                  f"and it's the ONLY remaining issue -- accepting minor cosmetic imperfection "
                  f"instead of re-spending a Gemini call on an already-declined fix.")
            break

        print("  -> requesting targeted fix based on measured PDF...")
        content = fix_layout(
            content, hbox_warnings, fill_report, verb_report, jd_text,
            unsupported_bullet_hits=unsupported["bullet_hits"],
            narrative_issues=narrative_issues,
        )
        if content.get("flagged"):
            print(f"  !! flagged (could not fix without cutting a metric/keyword): {content['flagged']}")

    pdf_path = tex_out.with_suffix(".pdf")
    content_path = out_dir / "content.json"
    content_path.write_text(json.dumps(content, indent=2))

    # FINAL, AUTHORITATIVE guardrail check against the actual final content.
    final_unsupported = check_unsupported_terms(content, master_content)
    final_unverified_terms = flag_unverified_terms(content, master_content)
    final_narrative_issues = check_narrative_coherence(content)

    final_strip_needed = bool(final_unsupported["skills_hits"]) or bool(final_unverified_terms)
    if final_unsupported["skills_hits"]:
        content = strip_unsupported_skills(content, final_unsupported["skills_hits"])
    if final_unverified_terms:
        content = strip_unsupported_skills(content, final_unverified_terms)

    if final_strip_needed:
        print("  -> FINAL CHECK forced a Technical Skills strip -- recompiling once more "
              "so the shipped PDF actually reflects it (no Gemini call).")
        content_path.write_text(json.dumps(content, indent=2))
        tex_out.write_text(fill_template(template, content))
        compile_tex(tex_out)
        final_unsupported = check_unsupported_terms(content, master_content)
        final_unverified_terms = flag_unverified_terms(content, master_content)
        final_narrative_issues = check_narrative_coherence(content)

    if final_unsupported["skills_hits"] or final_unsupported["bullet_hits"]:
        print(f"  !! FINAL CHECK: unsupported term(s) still present -- "
              f"skills: {[h['term'] for h in final_unsupported['skills_hits']]}, "
              f"bullets: {[h['term'] for h in final_unsupported['bullet_hits']]}")
        layout_clean = False
    if final_unverified_terms:
        print(f"  !! FINAL CHECK: unverified skill term(s) still present -- "
              f"{[t['term'] for t in final_unverified_terms]}")
        layout_clean = False
    if final_narrative_issues:
        print(f"  !! FINAL CHECK: narrative-coherence violation(s) still present: "
              f"{[n['location'] for n in final_narrative_issues]}")
        layout_clean = False

    if pdf_path.exists():
        print(f"  -> PDF ready: {pdf_path}")
        print("  -> running final verification (page count + proofread)...")
        verification = verify_job(out_dir, pdf_path)
        if not verification["page_check"]["one_page"]:
            print(f"  !! FAIL: PDF is {verification['page_check']['page_count']} pages, "
                  f"not 1 -- marking needs_review.")

        if verification.get("proofread") and not verification["proofread"].get("clean", True):
            issues = verification["proofread"]["issues"]

            # CIRCUIT BREAKER: if most/all flagged issues share near-identical
            # wording about a garbled/garbage/bullet character, this is almost
            # certainly a systemic PDF-extraction artifact (a new, still-
            # unstripped font-rendering codepoint), not per-bullet content
            # damage. Rewriting bullet text can NEVER fix a character inserted
            # at compile time by \item itself -- confirmed by a real run that
            # burned 3 Gemini accounts trying anyway. Skip the auto-fix loop
            # entirely and flag it for a human to add the new codepoint to
            # extract_text()'s strip regex instead of wasting API calls.
            rendering_artifact_signal = sum(
                1 for i in issues
                if re.search(r"garbage|bullet point|replacement symbol|non-standard",
                              i.get("problem", ""), re.IGNORECASE)
            )
            if len(issues) >= 3 and rendering_artifact_signal >= len(issues) * 0.7:
                print(f"  !! {len(issues)} proofread issue(s) found, but {rendering_artifact_signal} "
                      f"share near-identical wording about a garbled/bullet character -- this pattern "
                      f"means a PDF-extraction artifact, not real content typos. Skipping the auto-fix "
                      f"loop (it cannot fix a rendering-time character by editing bullet text) and "
                      f"marking needs_review for a human to check extract_text()'s strip regex.")
                verification["needs_review"] = True
            else:
                print(f"  !! {len(issues)} proofread issue(s) found -- attempting Groq-based "
                      f"auto-fix (free tier) before escalating to Gemini:")
                for issue in issues:
                    print(f"      - \"{issue['snippet']}\" -- {issue['problem']}")

            MAX_GROQ_FIX_ATTEMPTS = 2
            resolved = False
            for groq_attempt in range(1, MAX_GROQ_FIX_ATTEMPTS + 1):
                content, unresolved = fix_proofread_issues(content, issues, master_content, use_groq=True)

                post_fix_unsupported = check_unsupported_terms(content, master_content)
                post_fix_narrative = check_narrative_coherence(content)
                if post_fix_unsupported["bullet_hits"] or post_fix_narrative:
                    print(f"  !! Groq's fix introduced a new guardrail violation "
                          f"(attempt {groq_attempt}/{MAX_GROQ_FIX_ATTEMPTS}) -- retrying.")
                    continue

                content_path.write_text(json.dumps(content, indent=2))
                tex_out.write_text(fill_template(template, content))
                compile_tex(tex_out)

                print(f"  -> re-running Groq proofread to confirm (attempt {groq_attempt}/{MAX_GROQ_FIX_ATTEMPTS})...")
                verification = verify_job(out_dir, pdf_path)
                still_broken = verification.get("proofread") and not verification["proofread"].get("clean", True)
                if not still_broken and not unresolved:
                    print("  -> proofread confirmed clean via Groq -- no Gemini call needed.")
                    resolved = True
                    break
                issues = verification.get("proofread", {}).get("issues", issues)

            if not resolved:
                print(f"  !! Groq couldn't resolve it after {MAX_GROQ_FIX_ATTEMPTS} free attempts "
                      f"-- escalating to ONE Gemini fix as last resort.")
                content, unresolved = fix_proofread_issues(content, issues, master_content, use_groq=False)
                content_path.write_text(json.dumps(content, indent=2))
                tex_out.write_text(fill_template(template, content))
                compile_tex(tex_out)
                verification = verify_job(out_dir, pdf_path)
                still_broken = verification.get("proofread") and not verification["proofread"].get("clean", True)
                if still_broken or unresolved:
                    print("  !! still not clean after Gemini escalation -- marking needs_review for manual review.")
                    verification["needs_review"] = True
                else:
                    print("  -> proofread confirmed clean after Gemini escalation.")

        if verification["needs_review"]:
            layout_clean = False
    else:
        print("  !! PDF was not produced -- check the .log file for a LaTeX error.")
        layout_clean = False

    print("  -> writing cover letter...")
    cover_letter = write_cover_letter(jd_text, content)
    (out_dir / "cover_letter.txt").write_text(cover_letter)

    (out_dir / "keyword_audit.json").write_text(
        json.dumps(content.get("keyword_audit", []), indent=2)
    )

    if final_unverified_terms:
        (out_dir / "unverified_terms_flagged.json").write_text(
            json.dumps(final_unverified_terms, indent=2)
        )

    all_final_unsupported_terms = (
        [h["term"] for h in final_unsupported["skills_hits"]]
        + [h["term"] for h in final_unsupported["bullet_hits"]]
    )
    if all_final_unsupported_terms:
        (out_dir / "unsupported_terms_flagged.json").write_text(
            json.dumps(final_unsupported, indent=2)
        )

    status_label = "clean" if layout_clean else "needs_review"
    print(f"  -> done: {out_dir}/ (status: {status_label})")
    return base_name, layout_clean

def main():
    master_content = Path(MASTER_CONTENT_PATH).read_text()
    template = Path(TEMPLATE_PATH).read_text()

    job_files = sorted(glob.glob(f"{JOBS_DIR}/*.txt"))
    if not job_files:
        raise SystemExit(f"No JD files found in {JOBS_DIR}/. Add one .txt file per job (filename = company_role).")

    manifest = load_manifest()
    to_process = []
    for jd_path_str in job_files:
        jd_path = Path(jd_path_str)
        entry = manifest.get(jd_path.name)
        if entry is None:
            to_process.append(jd_path)
        elif entry.get("status") == "needs_review" and entry.get("retry_count", 0) < NEEDS_REVIEW_RETRY_LIMIT:
            print(f"Retrying {jd_path.name} -- previously needs_review "
                  f"(attempt {entry.get('retry_count', 0) + 1}/{NEEDS_REVIEW_RETRY_LIMIT})")
            to_process.append(jd_path)
        else:
            status = entry.get("status", "done")
            print(f"Skipping {jd_path.name} -- {status} -> "
                  f"{entry['output_folder']} (done {entry['processed_at']})")

    if not to_process:
        print("Nothing new to process -- every .txt file in jobs/ is already done or out of review retries.")
    else:
        print(f"Found {len(to_process)} job(s) to process "
              f"({len(job_files) - len(to_process)} already done/exhausted, skipped).")
        for jd_path in to_process:
            try:
                output_folder, layout_clean = process_job(jd_path, master_content, template)
                if output_folder:
                    prior = manifest.get(jd_path.name, {})
                    was_needs_review = prior.get("status") == "needs_review"
                    manifest[jd_path.name] = {
                        "output_folder": output_folder,
                        "processed_at": datetime.now().isoformat(timespec="seconds"),
                        "status": "done" if layout_clean else "needs_review",
                        "retry_count": 0 if layout_clean else (prior.get("retry_count", 0) + (1 if was_needs_review else 0)),
                    }
                    save_manifest(manifest)
                else:
                    print(f"  !! {jd_path.name} did not complete successfully -- "
                          f"NOT marked done, will retry on next run.")
            except Exception as e:
                print(f"  !! FAILED on {jd_path}: {e}")
                prior = manifest.get(jd_path.name, {})
                manifest[jd_path.name] = {
                    "output_folder": prior.get("output_folder"),
                    "processed_at": datetime.now().isoformat(timespec="seconds"),
                    "status": "needs_review",
                    "retry_count": prior.get("retry_count", 0) + 1,
                    "last_error": str(e),
                }
                save_manifest(manifest)
                continue

        print("\nAll jobs processed. Check the output/ folder.")

    all_needs_review = [
        (name, info) for name, info in manifest.items()
        if info.get("status") == "needs_review"
    ]
    if all_needs_review:
        print(f"\n!! {len(all_needs_review)} job(s) currently marked needs_review "
              f"(layout/verb/page/guardrail issues unresolved, measurement unreliable, or a real proofread "
              f"finding needs manual review):")
        for name, info in all_needs_review:
            retries_left = max(0, NEEDS_REVIEW_RETRY_LIMIT - info.get("retry_count", 0))
            print(f"   - {name} -> output/{info['output_folder']}/ "
                  f"(retry {info.get('retry_count', 0)}/{NEEDS_REVIEW_RETRY_LIMIT}, "
                  f"{retries_left} auto-retr{'y' if retries_left == 1 else 'ies'} left)")


if __name__ == "__main__":
    main()
