"""The only place this pipeline talks to a language model.

Three transports share one request path: Gemini (multi-account rotation),
Groq (bounded retries, Gemini as fallback) and a mock used by `--mock` runs.
The mock replaces provider RESPONSES only - prompt construction, response
validation, grounding and the retry loop are the same code in every mode.

What a model is allowed to decide:
  * which 3 projects to select, and their rank;
  * the wording of project bullets (inside that project's own evidence);
  * the cover letter draft, and the closing assessment.

What a model never touches: Professional Experience, Technical Skills,
section order, project headers, layout. Those are deterministic.

Secrets: a key is read from `Credentials` at call time and never logged.
Logs identify an account as `#1`, `#2`, and never carry key material.
"""
from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import grounding
import resume_engine as engine
from resume_engine import (BACKOFF_BASE_SECONDS, BACKOFF_MAX_SECONDS, Credentials,
                           GEMINI_MODEL, GROQ_MAX_ATTEMPTS, GROQ_MAX_OUTPUT_TOKENS,
                           GROQ_MODEL, MasterFacts, Policy, Project,
                           RATE_LIMIT_MAX_ATTEMPTS, SERVER_ERROR_MAX_ATTEMPTS, Signals)



BULLET_ATTEMPTS = 3
LETTER_ATTEMPTS = 3
ASSESSMENT_ATTEMPTS = 3


# ================================================================= requests


@dataclass
class Request:
    """One model call. `context` is for the mock; real transports ignore it."""

    purpose: str
    prompt: str
    context: dict[str, Any] = field(default_factory=dict)
    temperature: float = 0.4
    max_tokens: int = GROQ_MAX_OUTPUT_TOKENS
    json: bool = True          # False for purposes whose output is plain prose


@dataclass
class Reply:
    """Output plus the provider that actually produced it."""

    text: str
    provider: str
    detail: str = "ok"


@dataclass
class CallRecord:
    """One provider generation, audited on two independent axes.

    `transport_ok` says the provider answered. `accepted` says the pipeline
    could actually use what came back: a schema violation or a grounding
    rejection is a successful transport with an unusable payload.
    """

    purpose: str
    provider: str
    transport_ok: bool
    accepted: bool
    detail: str = "ok"

    def as_dict(self) -> dict:
        return {"purpose": self.purpose, "provider": self.provider,
                "transport_ok": self.transport_ok, "accepted": self.accepted,
                # `ok` is retained for compatibility and means `accepted`:
                # strategy.json audits usable outputs, not HTTP status.
                "ok": self.accepted, "detail": self.detail}


class ProviderError(RuntimeError):
    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category


class FailFast(ProviderError):
    """A malformed or unserviceable request. Retrying cannot help."""


def classify_http(status: int, body: str) -> str:
    lowered = body.lower()
    if status == 429:
        return "quota_exhausted" if "quota" in lowered or "exhausted" in lowered else "rate_limited"
    if status in (401, 403):
        return "auth_permission"
    if status == 404:
        return "bad_request"          # retired model or wrong path
    if status == 400:
        return "bad_request"
    if status >= 500:
        return "server_error"
    return "transport"


def classify_exception(error: Exception) -> str:
    """Classify an SDK exception without ever touching credential material."""
    status = getattr(error, "code", None) or getattr(error, "status_code", None)
    text = str(error).lower()
    if isinstance(status, int):
        return classify_http(status, text)
    if "model_not_found" in text or "not found" in text or "is not supported" in text:
        return "bad_request"
    if "quota" in text or "exhausted" in text:
        return "quota_exhausted"
    if "rate limit" in text or "429" in text:
        return "rate_limited"
    if "api key" in text or "permission" in text or "unauthorized" in text:
        return "auth_permission"
    if "invalid" in text or "400" in text:
        return "bad_request"
    if "internal" in text or "unavailable" in text or "500" in text or "503" in text:
        return "server_error"
    return "transport"


def _trim(text: str, limit: int = 240) -> str:
    flat = re.sub(r"\s+", " ", text).strip()
    return flat[:limit] + ("..." if len(flat) > limit else "")


def _backoff(attempt: int) -> float:
    return min(BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)), BACKOFF_MAX_SECONDS)


# =============================================================== transports


class GeminiTransport:
    """Gemini with account rotation. Quota/auth failures move to the next key."""

    name = "gemini"

    def __init__(self, credentials: Credentials, log, model: str = GEMINI_MODEL):
        self.credentials = credentials
        self.log = log
        self.model = model
        if not credentials.gemini_accounts:
            raise ProviderError("auth_permission", "no Gemini API key is configured")

    def generate(self, request: Request) -> Reply:
        last: ProviderError | None = None
        for account in self.credentials.gemini_accounts:
            for attempt in range(1, RATE_LIMIT_MAX_ATTEMPTS + 1):
                self.log.info("gemini account #%d attempt %d model=%s purpose=%s",
                              account, attempt, self.model, request.purpose)
                try:
                    return Reply(self._generate(self.credentials.gemini_key(account), request),
                                 self.name)
                except FailFast:
                    raise
                except ProviderError as error:
                    last = error
                    self.log.warning("gemini account #%d attempt %d failed: category=%s %s",
                                     account, attempt, error.category, error)
                    if error.category == "bad_request":
                        raise FailFast(error.category,
                                       f"Gemini rejected the request ({error}); not retrying")
                    if error.category in ("quota_exhausted", "auth_permission"):
                        break                            # rotate to the next account
                    limit = (SERVER_ERROR_MAX_ATTEMPTS if error.category == "server_error"
                             else RATE_LIMIT_MAX_ATTEMPTS)
                    if attempt >= limit:
                        break
                    time.sleep(_backoff(attempt))
        raise ProviderError(last.category if last else "transport",
                            f"every Gemini account failed; last category="
                            f"{last.category if last else 'unknown'}: {last}")

    def _generate(self, api_key: str, request: Request) -> str:   # pragma: no cover - live path
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        settings: dict[str, Any] = {"temperature": request.temperature}
        if request.json:
            settings["response_mime_type"] = "application/json"
        config = types.GenerateContentConfig(**settings)
        try:
            response = client.models.generate_content(
                model=self.model, contents=request.prompt, config=config)
        except Exception as error:
            raise ProviderError(classify_exception(error), _trim(str(error))) from None
        text = response.text or ""
        if not text.strip():
            raise ProviderError("malformed_response", "Gemini returned an empty response")
        return text


class GroqTransport:
    """Groq with bounded retries, falling back to Gemini when configured."""

    name = "groq"

    def __init__(self, credentials: Credentials, log, model: str = GROQ_MODEL,
                 fallback: "GeminiTransport | None" = None):
        self.credentials = credentials
        self.log = log
        self.model = model
        self.fallback = fallback

    def generate(self, request: Request) -> Reply:
        if not self.credentials.has_groq:
            if self.fallback:
                self.log.warning("no Groq key configured; using the Gemini fallback for purpose=%s",
                                 request.purpose)
                return self._fallback(request)
            raise ProviderError("auth_permission", "no Groq API key is configured")

        last: ProviderError | None = None
        for attempt in range(1, GROQ_MAX_ATTEMPTS + 1):
            self.log.info("groq attempt %d model=%s purpose=%s",
                          attempt, self.model, request.purpose)
            try:
                return Reply(self._generate(request), self.name)
            except ProviderError as error:
                last = error
                self.log.warning("groq attempt %d failed: category=%s %s",
                                 attempt, error.category, error)
                if error.category == "bad_request":
                    break
                if attempt < GROQ_MAX_ATTEMPTS:
                    time.sleep(_backoff(attempt))
        if self.fallback:
            self.log.warning("groq unavailable (category=%s); falling back to Gemini for purpose=%s",
                             last.category if last else "unknown", request.purpose)
            return self._fallback(request)
        raise ProviderError(last.category if last else "transport", f"Groq failed: {last}")

    def _fallback(self, request: Request) -> Reply:
        """Gemini answered, so the reply must say Gemini - not the wrapper."""
        reply = self.fallback.generate(request)
        return Reply(reply.text, reply.provider, "fallback_from_groq")

    def _generate(self, request: Request) -> str:                 # pragma: no cover - live path
        from groq import Groq

        client = Groq(api_key=self.credentials.groq_key)
        # Only ask for a JSON object when the parser needs one. This model
        # rejects its own generation (400 json_validate_failed) when a prose
        # answer like a cover letter is forced through JSON mode.
        settings: dict[str, Any] = {}
        if request.json:
            settings["response_format"] = {"type": "json_object"}
        try:
            completion = client.chat.completions.create(
                model=self.model, temperature=request.temperature,
                max_tokens=request.max_tokens,
                messages=[{"role": "user", "content": request.prompt}], **settings)
        except Exception as error:
            raise ProviderError(classify_exception(error), _trim(str(error))) from None
        choice = completion.choices[0]
        # A reasoning model can spend the whole budget before answering; a
        # truncated cover letter is worse than a retry or a fallback.
        if choice.finish_reason == "length":
            raise ProviderError("malformed_response",
                                f"Groq hit the {request.max_tokens}-token output cap "
                                f"(finish_reason=length)")
        return choice.message.content or ""


# ==================================================================== mock

# Curated, evidence-grounded bullet candidates. Every number here is copied
# from that project's own evidence with its exact structure (`45+`, `sub-500 ms`,
# `approximately 26%`). Candidates are offered per project so the mock can
# respect the resume-wide action-verb budget, exactly as a real model is asked to.
MOCK_BULLETS: dict[str, list[str]] = {
    "lms": [
        "Engineered a decoupled learning management system with a Django REST Framework backend and a React SPA frontend, using PostgreSQL for durable persistence and Redis for ephemeral real-time state and ranking.",
        "Implemented the real-time subsystem with Django Channels, Daphne, Redis, and WebSockets, serving live quizzes, dynamic leaderboards, and rate-limited chat to 45+ concurrent users with sub-500 ms broadcast latency.",
        "Optimized REST and submission paths under Locust and asyncio load tests of up to 100 concurrent users, cutting p95 REST latency by approximately 26% and quiz-submission processing from approximately 2000 ms to 1500 ms.",
        "Designed stateless JWT authentication with a custom JWT-over-WebSocket handshake, and added server-side grading with idempotent quiz submissions and a backtracking scheduler for conflict-free schedules.",
    ],
    "temp": [
        "Developed an end-to-end IoT environmental monitoring system during a weekend hackathon, with a Python, Django, and Django REST Framework backend that parsed and validated incoming sensor data.",
        "Integrated Arduino microcontrollers and MPI 3118A sensors to transmit localized ambient-temperature data, storing history and location metadata in PostgreSQL for an interactive ReactJS dashboard.",
        "Delivered a ReactJS dashboard that polled the REST API for real-time building temperatures, with Auth0 protecting access to the system.",
    ],
    "pintos": [
        "Implemented the user-programs layer of Pintos, an x86 teaching operating-system kernel written in C, covering process execution, stack setup for argument passing, parent-child synchronization, waiting, and clean termination.",
        "Engineered a system-call handler covering 13 process-control and file I/O operations, with thread-safe file-descriptor management for up to 128 open files per process and executable write-denial protection.",
        "Hardened the kernel using synchronization primitives and byte-wise memory validation to prevent deadlocks, kernel panics, and invalid-memory failures, passing 100% of 80 concurrency, memory-fault, and edge-case tests.",
    ],
    "fraud": [
        "Engineered a graph-based fraud detection pipeline over the IEEE-CIS dataset of 590,540 transactions with an approximately 3.5% fraud rate, converting tabular records into a graph across shared card, address, and email-domain identifiers.",
        "Implemented Focal Loss, class weighting, and threshold optimization to handle an approximately 27.6:1 class imbalance, capping large groups to control memory use during graph construction.",
        "Benchmarked a baseline MLP, GraphSAGE, and GAT in PyTorch Geometric, where GraphSAGE reached 0.9259 AUC-ROC and 0.5641 fraud F1 while GAT underperformed with hub-collapse behavior.",
    ],
    "tailor_pipeline": [
        "Engineered an automated resume-tailoring pipeline that treats model output as untrusted until independently verified, compiling generated LaTeX with pdflatex and re-extracting the rendered PDF with pdftotext so checks read the real artifact.",
        "Implemented deterministic factual-grounding and narrative-coherence checks that reject unsupported technology claims and bullets merging unrelated accomplishments into inflated claims.",
        "Integrated an independent Groq model to proofread compiled text, with targeted repair logic and cost-aware retries that send only genuinely failing sections back for revision.",
    ],
    "music": [
        "Developed a machine learning pipeline classifying raw audio into music genres, extracting MFCC, Spectral Centroid, Spectral Rolloff, Zero-Crossing Rate, and Chroma features with Librosa.",
        "Evaluated CatBoost and KNN models on the extracted feature set, reaching 97.68% classification accuracy and publishing the results in IJRAR.",
        "Implemented the feature-extraction and evaluation workflow in Python with Pandas, NumPy, and scikit-learn.",
    ],
}

MOCK_LETTER = """Dear Hiring Manager,

{paragraph_one}

{paragraph_two}

{project_paragraph}

I would be glad to talk through any of this in more detail.

Sincerely,
Jay Niketan Pathare"""

MOCK_OPENING = (
    "I am applying for the {job_title} role at {company}. My experience building Python "
    "backend systems, REST APIs and real-time services lines up closely with what this "
    "role asks for.{eligibility}")

MOCK_EXPERIENCE_PARAGRAPH = (
    "At Thesis Mumbai Tech I led platform module delivery end to end, from gathering "
    "requirements in client meetings through prototyping to validate scope and reviewing "
    "and merging the team's work into production. I built healthcare modules supporting "
    "10k+ records and designed a real-time WebSocket and Redis pipeline that fed IoT "
    "sensor data into PostgreSQL with ~2s live monitoring, and I architected the "
    "relational schemas and SQL queries behind those services.")


_MONTH_NAMES = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
                7: "July", 8: "August", 9: "September", 10: "October", 11: "November",
                12: "December"}

MOCK_PROJECT_PARAGRAPHS = {
    "lms": ("Outside work I built a decoupled learning management system whose real-time "
            "layer runs on Django Channels, Daphne, Redis and WebSockets, serving 45+ "
            "concurrent users with sub-500 ms broadcast latency. Load testing it with "
            "Locust and asyncio at up to 100 concurrent users is where I learned most "
            "about keeping latency predictable under concurrency."),
    "pintos": ("I also implemented the user-programs layer of Pintos, an x86 teaching "
               "kernel written in C, covering process execution, parent-child "
               "synchronization and a system-call handler, and passed 100% of 80 "
               "concurrency and memory-fault tests. That work is the closest thing I have "
               "to low-level debugging under real correctness pressure."),
    "tailor_pipeline": ("I also built an automated resume-tailoring pipeline that treats "
                        "model output as untrusted until independently verified, compiling "
                        "LaTeX with pdflatex and re-extracting the rendered PDF with "
                        "pdftotext so every check reads the real artifact."),
    "temp": ("I also built an end-to-end IoT monitoring system in a weekend hackathon, with "
             "a Python, Django and Django REST Framework backend that parsed and validated "
             "incoming sensor data and stored history in PostgreSQL."),
    "fraud": ("I also built a graph-based fraud detection pipeline over the IEEE-CIS dataset "
              "of 590,540 transactions, where GraphSAGE reached 0.9259 AUC-ROC against an "
              "MLP baseline."),
    "music": ("I also built an audio classification pipeline using Librosa features with "
              "CatBoost and KNN models, reaching 97.68% accuracy and a published paper."),
}



class MockTransport:
    """Deterministic, realistic structured output. Makes zero network calls."""

    name = "mock"

    def __init__(self, master: MasterFacts, log, seed: int = 7):
        self.master = master
        self.log = log
        self.random = random.Random(seed)

    def generate(self, request: Request) -> Reply:
        self.log.info("mock transport serving purpose=%s (no network call)", request.purpose)
        if not request.json:
            return Reply(self._letter(request.context), self.name)
        handler: Callable[[dict], dict] = {
            "project_selection": self._select,
            "project_bullets": self._bullets,
            "bullet_repair": self._repair,
            "assessment": self._assess,
        }[request.purpose]
        return Reply(json.dumps(handler(request.context)), self.name)

    def _select(self, context: dict) -> dict:
        ranked = engine.rank_projects(self.master, context["jd_text"], context["role_family"])
        count = context["count"]
        chosen = ranked[:count]
        return {
            "role_family": context["role_family"],
            "career_stage": "early_career_masters_student",
            "selected": [
                {"project_id": pid, "llm_rank": index + 1,
                 "reason": f"strongest overlap with the job description ({matched})"}
                for index, (pid, _score, matched) in enumerate(chosen)],
            "considered": [
                {"project_id": pid, "selected": pid in {c[0] for c in chosen},
                 "reason": matched} for pid, _score, matched in ranked],
            "skill_priority": context.get("skill_hints", []),
        }

    def _bullets(self, context: dict) -> dict:
        project_id = context["project_id"]
        candidates = list(MOCK_BULLETS.get(project_id, []))
        wanted = context["count"]
        budget: dict[str, int] = dict(context.get("verb_budget", {}))
        limit = context.get("verb_limit", 2)
        chosen: list[str] = []
        for bullet in candidates:
            if len(chosen) == wanted:
                break
            verb = engine.opening_verb(bullet)
            if budget.get(verb, 0) >= limit:
                continue
            budget[verb] = budget.get(verb, 0) + 1
            chosen.append(bullet)
        for bullet in candidates:                       # budget could not be met
            if len(chosen) == wanted:
                break
            if bullet not in chosen:
                chosen.append(bullet)
        return {"project_id": project_id, "bullets": chosen[:wanted]}

    def _repair(self, context: dict) -> dict:
        """Repair one bullet without touching facts, as the real prompt asks."""
        bullet = context["bullet"]
        goal = context.get("goal", "shorten")
        if goal == "reword_verb":
            # Swap only the opening verb; every fact stays exactly as it was.
            options = context.get("available_verbs") or []
            head, _, rest = bullet.partition(" ")
            if options and rest:
                return {"bullet": f"{options[0]} {rest}"}
            return {"bullet": bullet}
        if goal == "shorten":
            # Drop the trailing subordinate clause, keeping the main claim and
            # its metrics rather than truncating mid-thought.
            best = None
            for connector in (" so ", " while ", ", plus ", ", and ", " alongside ",
                              ", with ", " for "):
                head, sep, tail = bullet.rpartition(connector)
                if sep and len(head) > 110 and (best is None or len(head) > len(best)):
                    best = head
            if best:
                return {"bullet": best.rstrip(" ,") + "."}
            trimmed = re.sub(r"\s*\([^()]*\)", "", bullet, count=1)
            return {"bullet": trimmed if trimmed != bullet else bullet}
        alternatives = [b for b in MOCK_BULLETS.get(context["project_id"], [])
                        if b not in context.get("siblings", []) and b != bullet]
        return {"bullet": alternatives[0] if alternatives else bullet}

    def _letter(self, context: dict) -> str:
        """Assemble a letter from the current approved facts, per JD."""
        jd_text = (context.get("jd_text") or "").lower()
        ranked = list(context.get("project_ids") or ["lms"])
        # Choose the project that adds evidence this JD actually cares about.
        priorities = [("concurrency", "pintos"), ("kernel", "pintos"), ("systems", "pintos"),
                      ("real-time", "lms"), ("websocket", "lms"), ("latency", "lms"),
                      ("machine learning", "fraud"), ("model", "fraud"),
                      ("verification", "tailor_pipeline"), ("llm", "tailor_pipeline")]
        chosen = next((pid for needle, pid in priorities
                       if needle in jd_text and pid in ranked), ranked[0])
        eligibility = ""
        if context.get("needs_eligibility"):
            eligibility = (" I finish my Master of Science in Computer Science at the "
                           "University at Buffalo in December 2026 and can start full time "
                           "in January 2027 under OPT.")
        return MOCK_LETTER.format(
            paragraph_one=MOCK_OPENING.format(
                job_title=context.get("job_title") or "Software Engineer",
                company=context.get("company") or "your team",
                eligibility=eligibility),
            paragraph_two=MOCK_EXPERIENCE_PARAGRAPH,
            project_paragraph=MOCK_PROJECT_PARAGRAPHS.get(
                chosen, MOCK_PROJECT_PARAGRAPHS["lms"]))

    def _assess(self, context: dict) -> dict:
        """Classify Python's authoritative requirements against the final resume.

        Every entry cites a requirement_id, so a requirement identity can never
        be invented here either. Only explicit capability/qualification
        requirements reach the fit score; logistics, conditions and eligibility
        are routed to manual review.
        """
        all_requirements = list(context.get("requirements") or [])
        skills = list(context.get("skills") or [])
        tailoring = context.get("tailoring") or {}
        evidence = context.get("evidence") or engine.ResumeEvidence(skills=tuple(skills))

        scored = engine.scored_requirements(all_requirements)
        # One authority for both paths: the same deterministic verdicts the live
        # assessment is forced to adopt.
        verdicts = engine.deterministic_assessment(all_requirements, self.master, evidence,
                                                   context.get("jd_text") or "")
        strong = verdicts["strong_matches"][:12]
        partial = verdicts["partial_matches"][:8]
        gaps = verdicts["gaps"][:8]
        manual = verdicts["manual_review"]

        # Role signals the posting only describes: colour, not requirements.
        complementary = []
        for requirement in engine.role_signals(all_requirements):
            for term in requirement.terms:
                if self.master.canonical_skill(term):
                    complementary.append(f"{term} (role signal {requirement.requirement_id})")
        for extra in ("AWS (EC2/RDS)", "PostgreSQL", "WebSockets"):
            if extra in skills and not any(extra in c for c in complementary):
                complementary.append(f"{extra} (supporting evidence the posting does not ask "
                                     f"for)")

        required_total = sum(1 for r in scored if r.importance == "required") or 1
        strong_ids = {m["requirement_id"] for m in strong}
        required_strong = sum(1 for r in scored
                              if r.importance == "required"
                              and r.requirement_id in strong_ids)
        coverage = required_strong / required_total
        required_gaps = [g for g in gaps if g["importance"] == "required"]
        score = round(min(9.6, 6.0 + 3.6 * coverage) - 0.4 * len(required_gaps), 1)
        score = max(3.0, score)
        recommendation = ("strong_apply" if score >= 9 else "apply" if score >= 7.5
                          else "borderline" if score >= 6 else "skip")

        # Eligibility is computed once, by deterministic_assessment, with the
        # posting text in hand. Recomputing it here risked two answers.
        status = verdicts["eligibility"]["status"]
        details = list(verdicts["eligibility"]["details"])
        flags = list(verdicts["eligibility_flags"])

        notes = []
        if tailoring.get("skills_rendered_lines"):
            notes.append(f"Technical Skills renders "
                         f"{tailoring['skills_rendered_lines']} lines as required")
        if tailoring.get("experience_exact"):
            notes.append("Professional Experience matches approved wording exactly")
        if tailoring.get("projects"):
            notes.append(f"projects selected for this posting: {tailoring['projects']}")
        notes.append(f"{len(strong)} of {len(scored)} scored requirement(s) are directly "
                     f"evidenced on the resume")
        if manual:
            notes.append(f"{len(manual)} requirement(s) routed to manual review")
        tailoring_score = 9.5 if not tailoring.get("issues") else 7.0

        verdict = {"strong_apply": "a strong apply", "apply": "worth applying to",
                   "borderline": "borderline", "skip": "a poor match"}[recommendation]
        summary = (f"{len(strong)} of {len(scored)} scored requirement(s) are directly "
                   f"evidenced and {len(gaps)} are unsupported, so this posting is "
                   f"{verdict}.")
        return {
            "fit_score": score, "recommendation": recommendation, "summary": summary,
            "eligibility": {"status": status, "details": details},
            "strong_matches": strong, "partial_matches": partial, "gaps": gaps,
            "manual_review": manual, "complementary_strengths": complementary[:6],
            "tailoring_quality": {"score": tailoring_score, "notes": notes},
            "risk_flags": flags,
        }


# ================================================================== prompts

# ================================================================== prompts

_STYLE_RULES = """WRITING RULES (follow exactly):
- Tone: direct, technical, professional, concise, evidence first. Plain past tense.
- Preferred opening verbs: Built, Developed, Designed, Engineered, Implemented, Integrated,
  Architected, Parallelized, Delivered, Optimized, Automated, Scaled, Reduced.
- Never use: leveraged, spearheaded, cutting-edge, seamless, innovative, transformative,
  world-class, results-driven, utilize. Do not write "production-grade" unless it is meaningful.
- No em dashes anywhere. No LaTeX. No markdown. No bullet characters. Plain sentences.
- One sentence per bullet, ending with a period.
- Lead with the JD-relevant concept and use the job description's own supported vocabulary,
  but never keyword stuff.

FACTUAL RULES (absolute):
- Use ONLY the evidence listed for THIS project. Do not borrow a metric, technology, or
  responsibility from another project or from professional experience.
- Copy every number exactly as the evidence states it, including its form: "3+" is not "3",
  "45+" is not "45", "approximately 90%" is not "90%", "sub-500 ms" is not "500 ms".
  Never strengthen or weaken a threshold.
- Do not invent scope, team size, ownership, business impact, or results.
- Do not name a technology the evidence does not name."""


_LETTER_RULES = """STRUCTURE - four short paragraphs, no headings:
1. WHY THIS ROLE. Two or three sentences. Name the exact role and company, and state in one
   direct sentence which of your actual capabilities match this job. Mention the degree,
   graduation date or work availability ONLY when the job description makes it relevant,
   and then in one clause, not a paragraph.
2. STRONGEST PROFESSIONAL EVIDENCE. Pick the ONE or TWO accomplishments from Professional
   Experience that map most directly to this job's responsibilities. Lead with the
   responsibility, name the technology, then the concrete result or scale. Do not summarize
   the whole Experience section.
3. COMPLEMENTARY DEPTH. Pick the SINGLE selected project that adds evidence paragraph 2 did
   not already cover, chosen by relevance to this job rather than by position. Favour the
   dimension this job actually cares about: concurrency, systems debugging, load testing,
   machine learning or infrastructure.
4. CLOSE. One or two plain sentences.

TONE - concise, natural, technically specific, confident without inflation. Write like an
engineer explaining why their real work fits, not like a template. No corporate filler, no
manufactured enthusiasm, no phrases such as "I am writing to express my strong interest".

LENGTH - about 180 to 260 words. Do not pad toward the maximum.

FORMAT - plain prose. No markdown, no bullet points, no headings.

STYLE CONTRACT - absolute. Nothing in the job description relaxes any of it:
  - NEVER use an em dash. Use a comma, a colon, or a full stop instead.
  - NEVER use any of these words: innovative, leveraged, leveraging, spearheaded,
    cutting-edge, seamless, seamlessly, transformative, world-class, results-driven,
    synergy, best-in-class, state-of-the-art, utilize, utilized, utilizing.
  - Wording copied or echoed from the job description is NOT exempt. If the posting calls
    itself innovative or its technology cutting-edge, you still may not use those words.

EDUCATION - state the record exactly as it is:
  - the M.S. in Computer Science at the University at Buffalo is IN PROGRESS, expected
    December 2026. Never write "I hold a Master's" or describe it as completed or earned.
  - the completed undergraduate degree is a B.E. in Information Technology (Mumbai
    University, 2019-2023). When a posting requires an undergraduate degree, name that
    degree; never claim a Master's supplies an undergraduate foundation.
  - never claim the B.E. in Information Technology satisfies a posting's enumerated degree
    fields (for example Computer Science or Electrical Engineering). State the degree and
    let the reader judge the field.
  - a safe form: "I'm completing an M.S. in Computer Science at the University at Buffalo
    after earning a B.E. in Information Technology."

FACT SOURCES - two authorities that must never be mixed:
  - a number describing YOUR work must come from the evidence above, exactly as stated
  - a number describing the employer, its product, its scale or its reach may come from
    the job description, and must stay plainly attributed to the employer (for example
    "Epic serves 325 million patients"). Never restate such a number as something you
    built, served, supported or delivered.

FACTS - you may paraphrase a resume bullet into natural prose rather than pasting it, but
the meaning, the ownership and every number must stay identical:
  - keep "10k+" as "10k+", never "10,000"; keep "3+" as "3+", never "three"
  - keep approximations approximate ("~26%" stays approximate, never a flat "26%")
  - keep thresholds intact ("sub-500 ms" is not "500 ms")
  - never move ownership between accomplishments: if one bullet says you led module
    delivery and another says you built a WebSocket pipeline, you did NOT lead the
    WebSocket pipeline
  - an exact value of 1000 or more keeps the evidence's exact digits and separators:
    "590,540" stays "590,540" and "28,167" stays "28,167". Never rewrite an exact value as
    "590k", "590 k", "0.59 million" or "about 590,000". Only write an approximate form when
    the evidence itself is approximate, and then keep its authorized wording ("~90%",
    "10k+", "45+")
  - claim no technology that the evidence below does not name
  - do not claim "profiling" unless the evidence says profiling. Supported adjacent
    wording: testing concurrent code, concurrency testing, load testing, optimizing
    systems, performance optimization\""""


def _project_evidence_block(project: Project) -> str:
    lines = [f"PROJECT: {project.name}", f"project_id: {project.project_id}",
             f"context: {project.context}", f"date: {project.date}",
             f"technologies named by the evidence: {', '.join(project.tech)}",
             "SUPPORTED EVIDENCE (the only facts you may use):"]
    lines += [f"  - {item}" for item in project.evidence]
    if project.tags:
        lines.append(f"relevance tags: {', '.join(project.tags)}")
    return "\n".join(lines)


def _json_instruction(shape: str) -> str:
    return (f"Return ONLY valid JSON, no prose and no code fences, shaped exactly like:\n{shape}")


def parse_json(text: str, purpose: str) -> dict:
    """Parse a model response, tolerating fences and surrounding prose."""
    cleaned = re.sub(r"^\s*```(?:json)?|```\s*$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            pass
    raise ProviderError("malformed_response",
                        f"{purpose} response was not JSON: {_trim(text)}")


# =================================================================== client


@dataclass
class Selection:
    selected: list[str]
    ranks: dict[str, int]
    reasons: dict[str, str]
    considered: list[tuple[str, bool, str]]
    career_stage: str | None = None


class LLMClient:
    """Prompt construction, response validation and grounding for every call."""

    def __init__(self, gemini, groq, master: MasterFacts, policy: Policy, log):
        self.gemini = gemini
        self.groq = groq
        self.master = master
        self.policy = policy
        self.log = log
        self.calls: list[CallRecord] = []

    # -- plumbing --------------------------------------------------------
    def _invoke(self, transport, request: Request) -> Reply:
        try:
            reply = transport.generate(request)
        except ProviderError as error:
            self.calls.append(CallRecord(request.purpose, transport.name, False, False,
                                         f"category={error.category}: {error}"))
            self.log.error("[%s] %s call failed: category=%s %s", request.purpose.upper(),
                           transport.name, error.category, error)
            raise
        # reply.provider is whoever answered, which is not the wrapper when a
        # fallback handled the call.
        self.calls.append(CallRecord(request.purpose, reply.provider, True, True, reply.detail))
        return reply

    def reject_last(self, detail: str) -> None:
        """Mark the most recent generation unusable, keeping its history."""
        if self.calls:
            self.calls[-1].accepted = False
            self.calls[-1].detail = detail

    def _call(self, transport, request: Request) -> dict:
        return parse_json(self._invoke(transport, request).text, request.purpose)

    # -- project selection ----------------------------------------------
    def select_projects(self, jd_text: str, signals: Signals, count: int) -> Selection:
        catalogue = "\n\n".join(_project_evidence_block(p) for p in self.master.projects)
        prompt = f"""You are selecting which academic projects belong on a tailored resume.

JOB DESCRIPTION:
{jd_text}

DETECTED ROLE FAMILY: {signals.role_family}

ALL AVAILABLE PROJECTS (every one is factually supported):
{catalogue}

Evaluate EVERY project above and select exactly {count}, ranked 1 to {count} (1 is the most
relevant, and it receives the most resume space). Weigh: relevance to this job description,
ATS keyword overlap with the posting, technical depth, strength of the supporting evidence,
complementary coverage across the {count} picks, and whether specialized work helps or
distracts for THIS role. Give a short concrete reason for every project, selected or not.

{_json_instruction('''{"role_family": "...", "career_stage": "...",
 "selected": [{"project_id": "...", "llm_rank": 1, "reason": "..."}],
 "considered": [{"project_id": "...", "selected": true, "reason": "..."}]}''')}"""

        data = self._call(self.gemini, Request(
            "project_selection", prompt, temperature=0.2,
            context={"jd_text": jd_text, "role_family": signals.role_family, "count": count}))

        valid = {p.project_id for p in self.master.projects}
        selected, ranks, reasons = [], {}, {}
        for entry in data.get("selected", []):
            pid = str(entry.get("project_id", "")).strip()
            if pid in valid and pid not in ranks:
                ranks[pid] = int(entry.get("llm_rank", len(selected) + 1))
                reasons[pid] = str(entry.get("reason", "")).strip() or "no reason given"
                selected.append(pid)
        selected.sort(key=lambda pid: ranks[pid])
        if len(selected) != count:
            raise ProviderError("malformed_response",
                                f"project selection returned {len(selected)} valid project(s), "
                                f"expected exactly {count}")
        considered = [(str(e.get("project_id", "")), bool(e.get("selected")),
                       str(e.get("reason", "")).strip())
                      for e in data.get("considered", [])
                      if str(e.get("project_id", "")) in valid]
        return Selection(selected, ranks, reasons, considered,
                         str(data.get("career_stage", "")).strip() or None)

    # -- project bullets -------------------------------------------------
    def write_bullets(self, project: Project, count: int, jd_text: str,
                      verb_counts: dict[str, int], target_chars: tuple[int, int],
                      ) -> tuple[list[str], list[grounding.Problem]]:
        """Write `count` grounded bullets, re-prompting with the failures found."""
        exhausted = engine.exhausted_verbs(verb_counts, self.policy.verb_max_uses)
        low, high = target_chars
        base = f"""Write exactly {count} resume bullet(s) for one academic project.

JOB DESCRIPTION (for relevance and vocabulary only, never for facts):
{jd_text}

{_project_evidence_block(project)}

{_STYLE_RULES}

LENGTH: each bullet should be about {low} to {high} characters so it renders as two full
lines. Do not pad with filler and do not add facts to reach the length.
ACTION VERBS ALREADY USED ELSEWHERE ON THIS RESUME (do not start a bullet with these):
{', '.join(exhausted) or 'none'}
Do not start two bullets with the same verb.

{_json_instruction('{"project_id": "...", "bullets": ["...", "..."]}')}"""

        problems: list[grounding.Problem] = []
        bullets: list[str] = []
        for attempt in range(1, BULLET_ATTEMPTS + 1):
            prompt = base
            if problems:
                prompt += ("\n\nYour previous attempt was rejected by deterministic validation. "
                           "Fix exactly these problems and change nothing else:\n"
                           + "\n".join(f"  - {p.message}" for p in problems))
            data = self._call(self.gemini, Request(
                "project_bullets", prompt, temperature=0.35,
                context={"project_id": project.project_id, "count": count,
                         "verb_budget": dict(verb_counts),
                         "verb_limit": self.policy.verb_max_uses}))
            bullets = [re.sub(r"\s+", " ", str(b)).strip()
                       for b in data.get("bullets", []) if str(b).strip()]
            if len(bullets) != count:
                problems = [grounding.Problem("count", "error",
                                              f"returned {len(bullets)} bullets, expected {count}")]
                self.reject_last(f"{project.project_id}: {problems[0].message}")
                self.log.warning("[PROJECT WRITING] %s attempt %d: %s",
                                 project.project_id, attempt, problems[0].message)
                continue
            problems = []
            for bullet in bullets:
                problems.extend(grounding.check_bullet_grounding(
                    bullet, project, self.master, self.policy.banned_phrases))
            seen_verbs: dict[str, int] = {}
            for bullet in bullets:
                verb = engine.opening_verb(bullet)
                seen_verbs[verb] = seen_verbs.get(verb, 0) + 1
                if verb in exhausted:
                    problems.append(grounding.Problem(
                        "verb_budget", "error",
                        f"bullet starts with {verb!r}, already used "
                        f"{self.policy.verb_max_uses} times on this resume"))
            for verb, uses in seen_verbs.items():
                if uses > 1:
                    problems.append(grounding.Problem(
                        "verb_budget", "error",
                        f"{uses} bullets in this project start with {verb!r}"))
            blocking = grounding.errors(problems)
            for problem in problems:
                self.log.info("[PROJECT WRITING] %s attempt %d: %s",
                              project.project_id, attempt, problem)
            if not blocking:
                return bullets, problems
            self.reject_last(f"{project.project_id}: " + "; ".join(
                p.message for p in blocking[:2]))
            problems = blocking
        return bullets, problems

    def repair_bullet(self, project: Project, bullet: str, goal: str, reason: str,
                      siblings: list[str], available: list[str] | None = None
                      ) -> tuple[str, list[grounding.Problem]]:
        """Bounded, truthful repair of ONE bullet for layout or verb reasons."""
        options = list(available or [])
        directions = {
            "shorten": "Make it shorter so it fits on exactly two rendered lines.",
            "lengthen": ("Make it longer, using only unused evidence from the same project, so "
                         "it fills two rendered lines instead of leaving a short tail."),
            "reword_verb": (
                "Start the bullet with a DIFFERENT action verb and keep everything else "
                "intact. Every fact, number, technology and the JD-relevant idea must survive "
                f"unchanged. Natural choices still available: {', '.join(options) or 'none'}. "
                "Do not force an awkward synonym: if no natural verb fits, return the bullet "
                "unchanged."),
        }
        direction = directions[goal]
        prompt = f"""Revise ONE resume bullet for layout only. The facts must not change.

{_project_evidence_block(project)}

CURRENT BULLET:
{bullet}

OTHER BULLETS IN THIS PROJECT (do not duplicate their content):
{chr(10).join('  - ' + s for s in siblings) or '  (none)'}

WHY IT IS BEING REVISED: {reason}
WHAT TO DO: {direction}

{_STYLE_RULES}

Do not drop a number to save space; drop a clause instead. Do not add a new claim.

{_json_instruction('{"bullet": "..."}')}"""
        data = self._call(self.gemini, Request(
            "bullet_repair", prompt, temperature=0.2,
            context={"project_id": project.project_id, "bullet": bullet, "goal": goal,
                     "siblings": siblings, "available_verbs": options}))
        revised = re.sub(r"\s+", " ", str(data.get("bullet", ""))).strip()
        if not revised:
            return bullet, [grounding.Problem("repair", "error", "repair returned nothing")]
        problems = grounding.check_bullet_grounding(revised, project, self.master,
                                                    self.policy.banned_phrases)
        for sibling in siblings:
            if engine.normalize_plain(sibling).lower() == engine.normalize_plain(revised).lower():
                problems.append(grounding.Problem("repair", "error",
                                                  "repair duplicates a sibling bullet"))
        return revised, problems

    # -- cover letter ----------------------------------------------------
    def cover_letter(self, jd: engine.JobPosting, signals: Signals,
                     experience_plain: list[str], projects: list[Project],
                     themes: list[engine.Requirement] | None = None) -> tuple[str, list]:
        evidence = "\n".join(f"  - {item}" for item in experience_plain)
        project_facts = "\n\n".join(_project_evidence_block(p) for p in projects)
        facts = "\n".join(f"  - {f}" for f in self.master.standing_facts)
        themes = themes or []
        theme_block = "\n".join(
            f"  {index}. [{req.importance}] {req.headline}"
            + (f"  (names: {', '.join(req.terms)})" if req.terms else "")
            for index, req in enumerate(themes, start=1)) or "  (none extracted)"
        grad = engine.graduation_requirement(jd.text)
        base = f"""Write a cover letter for this job, grounded strictly in the evidence below.

JOB DESCRIPTION:
{jd.text}

RESUME EXPERIENCE BULLETS (exact approved wording; these are the professional facts):
{evidence}

SELECTED PROJECTS:
{project_facts}

STANDING FACTS (state these accurately if you mention them):
{facts}

THE REQUIREMENTS THAT MATTER MOST IN THIS POSTING:
{theme_block}

Cover at least TWO of those requirements with direct evidence, ideally one from
Professional Experience and one from a project. Depth beats keyword coverage: do not try
to mention every requirement.

{_LETTER_RULES}

{'ELIGIBILITY: this posting states a graduation requirement (' + grad['text'] + '). One concise sentence covering the Master of Science in Computer Science, December 2026 graduation and January 2027 availability under OPT is useful here.' if grad else 'ELIGIBILITY: this posting states no graduation or authorization requirement, so do not discuss immigration or logistics unless it strengthens the match.'}

Address it to "Dear Hiring Manager," and sign as Jay Niketan Pathare.
Do not claim HIPAA, FHIR, EHR, CI/CD, Kubernetes or microservices experience.

Return ONLY the letter itself as plain text: no JSON, no code fences, no commentary."""

        # Bounded repair, exactly like write_bullets(): one wording slip should
        # not cost the whole run. Only DETERMINISTIC VALIDATION failures retry
        # here; transport retry/failover stays owned by the transport layer.
        letter = ""
        problems: list[grounding.Problem] = []
        for attempt in range(1, LETTER_ATTEMPTS + 1):
            prompt = base
            if problems:
                prompt += ("\n\nPrevious attempt failed deterministic validation. Fix "
                           "exactly these problems and change nothing else:\n"
                           + "\n".join(f"  - {p.message}"
                                      for p in grounding.errors(problems)))
            reply = self._invoke(self.groq, Request(
                "cover_letter", prompt, temperature=0.5, json=False,
                context={"job_title": jd.job_title, "company": jd.company_name,
                         "project_ids": [p.project_id for p in projects],
                         "jd_text": jd.text, "needs_eligibility": bool(grad),
                         "themes": [r.headline for r in themes]}))
            letter = reply.text.strip()
            if len(letter) < 200:
                raise ProviderError("malformed_response",
                                    f"cover letter was only {len(letter)} characters")
            problems = grounding.validate_cover_letter(
                letter, self.master, masked_terms=(jd.job_id,) if jd.job_id else (),
                banned=self.policy.banned_phrases, jd_text=jd.text,
                company=jd.company_name)
            problems += grounding.validate_letter_quality(
                letter, company=jd.company_name, job_title=jd.job_title)
            problems = grounding.dedupe_problems(problems)
            blocking = grounding.errors(problems)
            if not blocking:
                return letter, problems
            # The provider answered, but an invalid letter is not a usable
            # artifact. Transport success and acceptance are audited separately.
            self.reject_last("cover-letter validation failed: " + "; ".join(
                p.message for p in blocking[:3]))
            self.log.warning("[COVER LETTER] attempt %d rejected: %s", attempt,
                             "; ".join(p.message for p in blocking[:2]))
        return letter, problems

    # -- assessment ------------------------------------------------------
    def assess(self, jd: engine.JobPosting, signals: Signals, selected: list[Project],
               skills: list[str], *, experience_plain: list[str],
               project_bullets: dict[str, list[str]],
               requirements: list[engine.Requirement],
               tailoring: dict | None = None,
               extra_experience: list[str] | None = None) -> dict:
        """Judge candidate-to-JD fit against the FINAL resume. Advisory only."""
        scored = engine.scored_requirements(requirements)
        manual = engine.manual_review_requirements(requirements)
        signal_reqs = engine.role_signals(requirements)
        requirement_block = "\n".join(
            f"  {req.requirement_id} [{req.kind}/{req.importance}] {req.original_text}"
            + (f"  (names: {', '.join(req.terms)})" if req.terms else "")
            for req in scored) or "  (none)"
        manual_block = "\n".join(
            f"  {req.requirement_id} [{req.kind}] {req.original_text}"
            for req in manual) or "  (none)"
        signal_block = "\n".join(
            f"  {req.requirement_id} names {', '.join(req.terms)}"
            for req in signal_reqs) or "  (none)"
        evidence = engine.ResumeEvidence(
            experience_text=" ".join(list(experience_plain) + list(extra_experience or [])),
            project_bullets=tuple((pid, " ".join(bullets))
                                  for pid, bullets in project_bullets.items()),
            skills=tuple(skills))
        # Python classifies first. These verdicts are immutable: the model is
        # asked to explain them, never to decide them.
        verdicts = engine.deterministic_assessment(requirements, self.master, evidence,
                                                   jd.text)
        table = engine.requirement_table(requirements)
        # Kept for the final integrity gate in the pipeline.
        self.last_verdicts = verdicts
        verdicts_text = engine.verdict_block(verdicts, table)
        resume_experience = "\n".join(f"  - {item}" for item in experience_plain)
        resume_projects = "\n".join(
            f"  {pid}:\n" + "\n".join(f"    - {b}" for b in bullets)
            for pid, bullets in project_bullets.items())
        grad = engine.graduation_requirement(jd.text)
        expected = engine.expected_graduation(self.master)
        facts = "\n".join(f"  - {f}" for f in self.master.standing_facts)

        prompt = f"""Assess how well this candidate fits this job. Advisory only: it must never
change the resume.

JOB DESCRIPTION:
{jd.text}

AUTHORITATIVE REQUIREMENTS - Python extracted these from the posting and owns their
identity. Cite them ONLY by requirement_id. You may never invent, rename, merge, split or
rephrase a requirement, and you may not add one that is absent from this table. These are
the only requirements that may affect fit_score:
{requirement_block}

MANUAL-REVIEW REQUIREMENTS - report these in `manual_review` by id with a short reason.
They are never scored and never gaps, because the recorded facts cannot settle them:
{manual_block}

ROLE SIGNALS - the posting only describes these; they are NOT requirements. Never score
them. If the candidate happens to be strong there, say so in `complementary_strengths`:
{signal_block}

THE FINAL TAILORED RESUME - Professional Experience:
{resume_experience}

THE FINAL TAILORED RESUME - Academic Projects:
{resume_projects}

THE FINAL TAILORED RESUME - Technical Skills:
  {', '.join(skills)}

EDUCATION AND STANDING FACTS:
{facts}
{f"  - the posting states a graduation requirement: {grad['text']}" if grad else "  - the posting states no graduation requirement"}

EXPLICITLY UNSUPPORTED - never credit the candidate with any of these:
  {', '.join(self.master.unsupported_heads())}

IMMUTABLE VERDICTS - Python has already classified every requirement against the final
resume. These are FACTS for you, decided deterministically from the shipped document:
{verdicts_text}

Your job is to EXPLAIN and SUMMARIZE these verdicts, never to change them. You may not
move a requirement to a different bucket, add a requirement, remove one, or contradict a
basis above. In particular you may NEVER write that a degree, qualification or skill is
missing, absent or undocumented when the verdict above shows it is present: a field or
scope mismatch is not an absent degree. Evidence you cite must be evidence the resume
actually contains; do not name a technology the resume does not list.

FOR REFERENCE, how those verdicts were reached:
  strong_match   the final resume shows direct evidence for it
  partial_match  related evidence exists but does not meet the requirement as written
  unsupported    no supporting evidence exists
Never infer an upgrade: Docker is NOT Kubernetes, C is NOT C++, AWS EC2/RDS is NOT general
cloud-platform expertise, Django is NOT FastAPI, and generic testing is NOT pytest.

FIT SCORE RUBRIC - this scores CANDIDATE-TO-JD FIT, not how polished the resume looks:
  9-10  nearly all important requirements evidenced
  8-8.9 core requirements strongly matched, only minor gaps
  7-7.9 meaningful match with several weaker or missing areas
  6-6.9 borderline but viable
  <6    substantial mismatch, or a hard requirement is not met
Do not penalize a technology the posting never asks for, and do not treat generic related
experience as an exact match.

RULES FOR GAPS - a gap cites a requirement_id from the authoritative table above and
explains in `detail` why it is unmet. If nothing important is missing, return an empty gaps
array rather than inventing one. A requirement_id may appear in exactly ONE of
strong_matches, partial_matches, gaps or manual_review.

SUPPORTING EVIDENCE THE POSTING DID NOT ASK FOR (for example AWS, PostgreSQL or WebSockets
when the posting never mentions them) belongs in `complementary_strengths` or the summary.
It must NOT become a requirement match and must NOT raise fit_score.

tailoring_quality is a SEPARATE judgement about how well the resume presents the candidate
for this posting. A well-tailored resume can still have a real gap.

ENUM CONTRACT - these fields accept EXACTLY ONE of the listed strings and no other
wording. Do not substitute a section title or a prose variant:
  strong_matches[].source and partial_matches[].source: "experience" | "project" |
      "education" | "skills"
  every requirement_id: exactly one of the ids listed in the authoritative table
  recommendation: "strong_apply" | "apply" | "borderline" | "skip"
  eligibility.status: "meets" | "uncertain" | "does_not_meet" | "not_applicable"
      (Python decides this deterministically and overwrites whatever you send)
  gaps[].importance: "required" | "preferred" | "unclear"
  gaps[].status: "unsupported" | "weak_evidence"

{_json_instruction('''{"fit_score": 8.2, "recommendation": "apply",
 "summary": "one to three concise sentences consistent with the verdicts above",
 "eligibility": {"status": "meets", "details": ["..."]},
 "strong_matches": [{"requirement_id": "REQ-001", "evidence": "one sentence explaining
                     this verdict", "source": "experience"}],
 "partial_matches": [{"requirement_id": "REQ-002", "evidence": "...",
                      "limitation": "..."}],
 "gaps": [{"requirement_id": "REQ-003", "importance": "required",
           "status": "unsupported", "detail": "...", "evidence": null}],
 "manual_review": [{"requirement_id": "REQ-004", "reason": "..."}],
 "complementary_strengths": ["..."],
 "tailoring_quality": {"score": 9.0, "notes": ["..."]},
 "risk_flags": ["..."]}''')}

Use exactly the requirement_ids and buckets shown in IMMUTABLE VERDICTS. Python overwrites
these fields with its own verdicts regardless, so any change you make is discarded."""

        known_ids = {req.requirement_id for req in requirements}
        # Bounded schema repair. Only STRUCTURAL provider-schema failures retry
        # here; the deterministic finalization below is untouched, and there is
        # no Python-only assessment fallback.
        data: dict = {}
        problems: list[grounding.Problem] = []
        for attempt in range(1, ASSESSMENT_ATTEMPTS + 1):
            attempt_prompt = prompt
            if problems:
                attempt_prompt += (
                    "\n\nYour previous assessment response failed deterministic schema "
                    "validation.\n\nFix exactly these structural problems:\n"
                    + "\n".join(f"  - {p.message}" for p in problems)
                    + "\n\nReturn the complete assessment JSON again. Do not alter "
                      "immutable Python verdict semantics.")
            reply = self._invoke(self.groq, Request(
                "assessment", attempt_prompt, temperature=0.3,
                context={"jd_text": jd.text, "requirements": requirements,
                         "requirement_ids": sorted(known_ids),
                         "skills": skills, "selected": selected, "evidence": evidence,
                         "tailoring": tailoring or {}, "grad": grad,
                         "expected": expected}))
            try:
                data = parse_json(reply.text, "assessment")
            except ProviderError as error:
                # A transport failure is not ours to retry; only unparseable
                # output counts as a rejected structural attempt.
                if error.category != "malformed_response":
                    raise
                problems = [grounding.Problem("json", "error", str(error))]
                self.reject_last(f"malformed JSON: {error}")
                self.log.warning("[ASSESSMENT] attempt %d rejected: malformed JSON",
                                 attempt)
                continue
            data, normalizations = grounding.canonicalize_assessment(data)
            for event in normalizations:
                self.log.info("[ASSESSMENT] normalized %s", event)
            problems = grounding.errors(
                grounding.validate_assessment(data, jd.text, known_ids))
            if not problems:
                break
            detail = ("schema validation failed: "
                      + "; ".join(p.message for p in problems[:4]))
            self.reject_last(detail)
            self.log.warning("[ASSESSMENT] attempt %d rejected: %s", attempt, detail)
        if problems:
            raise ProviderError(
                "malformed_response",
                f"assessment could not be validated after {ASSESSMENT_ATTEMPTS} "
                f"attempt(s): " + "; ".join(p.message for p in problems[:4]))

        # The model explained; Python decides. Verdicts, eligibility, the cap,
        # breadth and the user-facing summary are all applied here in order, so
        # assess() returns a fully finalized assessment.
        for note in grounding.finalize_assessment(
                data, verdicts=verdicts, table=table, requirements=requirements,
                scored=len(engine.scored_requirements(requirements)),
                signals=len(signal_reqs),
                manual=len(engine.manual_review_requirements(requirements))):
            self.log.info("[ASSESSMENT] %s", note)
        data.setdefault("eligibility_flags", verdicts.get("eligibility_flags") or [])
        return data


def build_client(master: MasterFacts, policy: Policy, log, *, mock: bool,
                 credentials: Credentials | None = None) -> LLMClient:
    if mock:
        transport = MockTransport(master, log)
        return LLMClient(transport, transport, master, policy, log)
    credentials = credentials or engine.load_credentials()
    gemini = GeminiTransport(credentials, log)
    groq = GroqTransport(credentials, log, fallback=gemini)
    return LLMClient(gemini, groq, master, policy, log)
