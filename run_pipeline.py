#!/usr/bin/env python3
"""The only entrypoint.

    python run_pipeline.py jobs/backend_swe.txt              # production run
    python run_pipeline.py jobs/backend_swe.txt --mock       # no API calls at all
    python run_pipeline.py jobs/backend_swe.txt --smoke-test # real providers, no tracking
    python run_pipeline.py                                   # every unprocessed JD in jobs/
    python run_pipeline.py --batch [jobs_dir]                # every JD in a folder
    python run_pipeline.py --revalidate output/<run_folder>  # re-verify, zero API calls

Every run writes a folder containing run.log, strategy.json, resume.tex,
resume.pdf, resume.txt, cover_letter.txt and assessment.json. run.log is meant
to be complete enough that run.log + strategy.json + resume.tex + resume.pdf
lets someone else reconstruct the whole run.

Statuses: success (every contract met), needs_review (something the pipeline
refuses to fix by inventing or rewriting protected content), failed (it could
not produce the artifacts at all). Protected content is never silently
repaired to turn needs_review into success.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
import traceback
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import grounding
import llm_client
import pdf_utils as pdf
import resume_engine as engine

RESUME_STEM = "resume"
REQUIRED_ARTIFACTS = ("run.log", "strategy.json", "resume.tex", "resume.pdf",
                      "resume.txt", "cover_letter.txt", "assessment.json")
CSV_COLUMNS = ["company_name", "job_title", "job_id", "applied_at_date"]

# Drafting heuristic only. The rendered PDF decides whether a bullet fits;
# character counts merely give the writer a sane starting length.
BULLET_CHAR_TARGET = (150, 215)
MAX_REPAIRS_PER_BULLET = 2
# A skill may only fill a short line if the JD itself ties it to the resume:
# tier 1-2 are JD keywords, tier 3 is a selected project's technology used in
# the same work as a JD keyword. Tier 4 is true but irrelevant, so it is padding.
FILL_MAX_TIER = 3

EXPERIENCE_HEADING = "EXPERIENCE"
PROJECTS_HEADING = "ACADEMIC PROJECTS"
SKILLS_HEADING = "TECHNICAL SKILLS"


# ================================================================== logging

_KEY_PATTERN = re.compile(r"\b(?:AIza[0-9A-Za-z_\-]{6,}|gsk_[0-9A-Za-z_\-]{6,})")


class SecretFilter(logging.Filter):
    """Keys, key prefixes and secrets never reach a log line."""

    def __init__(self, secrets: list[str]):
        super().__init__()
        self.secrets = sorted((s for s in secrets if s and len(s) > 8), key=len, reverse=True)

    def _clean(self, text: str) -> str:
        for secret in self.secrets:
            if secret in text:
                text = text.replace(secret, "[REDACTED]")
        return _KEY_PATTERN.sub("[REDACTED]", text)

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = self._clean(str(record.msg))
        if isinstance(record.args, tuple):
            record.args = tuple(self._clean(a) if isinstance(a, str) else a
                                for a in record.args)
        return True


class StageLog:
    """A logger view that tags every line with its pipeline stage."""

    def __init__(self, logger: logging.Logger, stage: str):
        self.logger = logger
        self.stage = stage

    def _emit(self, level: int, message: str, *args) -> None:
        self.logger.log(level, f"[{self.stage}] {message}", *args)

    def info(self, message: str, *args) -> None:
        self._emit(logging.INFO, message, *args)

    def warning(self, message: str, *args) -> None:
        self._emit(logging.WARNING, message, *args)

    def error(self, message: str, *args) -> None:
        self._emit(logging.ERROR, message, *args)

    def stage_log(self, stage: str) -> "StageLog":
        return StageLog(self.logger, stage)


def build_logger(secrets: list[str], console: bool = True) -> tuple[logging.Logger, list]:
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False
    logger.addFilter(SecretFilter(secrets))
    buffer = _Buffer()
    logger.addHandler(buffer)
    if console:
        stream = logging.StreamHandler(sys.stdout)
        stream.setFormatter(logging.Formatter("%(levelname)-7s %(message)s"))
        logger.addHandler(stream)
    return logger, [buffer]


class _Buffer(logging.Handler):
    """Holds records until the run folder exists, then replays them into it."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def attach_file(self, path: Path, logger: logging.Logger) -> logging.Handler:
        # One run per run.log: appending would mix several runs into the file
        # that is supposed to explain this one.
        handler = logging.FileHandler(path, mode="w", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
        for record in self.records:
            handler.emit(record)
        logger.addHandler(handler)
        self.records.clear()
        return handler


# ================================================================= tracking


class TrackingStateError(RuntimeError):
    """The tracking index exists but cannot be trusted."""


def load_index(index_path: Path) -> dict:
    """Read the tracking index, failing CLOSED on corruption.

    A missing file is a legitimately fresh state. A file that exists but is
    empty, unparseable, or not a JSON object means the skip source is damaged:
    treating that as "nothing processed yet" would re-apply to every posting
    already applied to, so it raises instead.
    """
    if not index_path.exists():
        return {}
    try:
        raw = index_path.read_text(encoding="utf-8")
    except OSError as error:
        raise TrackingStateError(f"{index_path} could not be read: {error}") from error
    if not raw.strip():
        raise TrackingStateError(
            f"{index_path} exists but is empty ({index_path.stat().st_size} bytes). The "
            f"tracking index is the authority for skip decisions, so this is treated as "
            f"corruption rather than a fresh state. Reconcile it before running.")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as error:
        raise TrackingStateError(
            f"{index_path} is not valid JSON ({error}). Reconcile it before running."
        ) from error
    if not isinstance(data, dict):
        raise TrackingStateError(
            f"{index_path} holds {type(data).__name__}, expected a JSON object. "
            f"Reconcile it before running.")
    return data


class Tracker:
    """Appends a production application to processed_jobs.csv, once."""

    def __init__(self, csv_path: Path, index_path: Path, enabled: bool = True):
        self.csv_path = csv_path
        self.index_path = index_path
        self.enabled = enabled

    def already_processed(self, fingerprint: str) -> str | None:
        if not self.enabled:
            return None
        index = load_index(self.index_path)      # raises on a corrupt index
        entry = index.get(fingerprint)
        return entry.get("run_folder") if isinstance(entry, dict) else None

    def missing_artifacts(self, run_dir: Path) -> list[str]:
        return [name for name in REQUIRED_ARTIFACTS if not (run_dir / name).exists()]

    def record(self, jd: engine.JobPosting, run_dir: Path, status: str, log: StageLog) -> bool:
        if not self.enabled:
            log.info("tracking skipped: this is a mock/smoke run, production tracking "
                     "files are never opened")
            return False
        if status != "success":
            log.info("tracking skipped: status=%s (only a verified success is recorded)", status)
            return False
        missing = self.missing_artifacts(run_dir)
        if missing:
            log.warning("tracking skipped: missing required artifact(s) %s", ", ".join(missing))
            return False

        try:
            index = load_index(self.index_path)
        except TrackingStateError as error:
            log.error("tracking aborted: %s", error)
            return False

        # Idempotence: an explicit rerun of an already-recorded fingerprint must
        # not append a second CSV row or repoint the index.
        existing = index.get(jd.fingerprint)
        if isinstance(existing, dict):
            folder = (existing.get("run_folder") or "").strip()
            if folder and (engine.OUTPUT_DIR / folder).exists():
                log.info("tracking idempotent: fingerprint already recorded against run "
                         "folder %s; no CSV row appended and the index is unchanged",
                         folder)
                return False
            log.error("tracking STOPPED: fingerprint %s is recorded against run folder "
                      "%r, which no longer exists. Not guessing a reconciliation; report "
                      "this state before recording.", jd.fingerprint[:12], folder)
            return False

        # The index is the authority for skip decisions, so commit it FIRST and
        # atomically. The CSV is a human-readable log written only afterwards.
        index[jd.fingerprint] = {"run_folder": run_dir.name, "source_file": jd.source_file,
                                 "recorded_at": datetime.now().isoformat(timespec="seconds")}
        temp_path = self.index_path.with_name(self.index_path.name + ".tmp")
        try:
            with temp_path.open("w", encoding="utf-8") as handle:
                json.dump(index, handle, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.index_path)
        except OSError as error:
            try:
                temp_path.unlink()
            except OSError:
                pass
            log.error("tracking aborted: could not commit %s (%s); %s was NOT touched",
                      self.index_path.name, error, self.csv_path.name)
            return False

        try:
            header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
            with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
                if header:
                    writer.writeheader()
                writer.writerow({"company_name": jd.company_name or "",
                                 "job_title": jd.job_title or "",
                                 "job_id": jd.job_id or "",
                                 "applied_at_date": date.today().isoformat()})
        except OSError as error:
            # The skip source is already committed; never roll it back.
            log.error("%s committed, but appending the human log %s failed (%s). The skip "
                      "source is safe; no rollback performed.",
                      self.index_path.name, self.csv_path.name, error)
            return True

        log.info("recorded in %s and %s (index committed first)",
                 self.index_path.name, self.csv_path.name)
        return True


# =================================================================== result


@dataclass
class RunResult:
    jd_path: Path
    status: str = "failed"
    run_dir: Path | None = None
    issues: list[str] = field(default_factory=list)
    strategy: dict = field(default_factory=dict)


def sanitize(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", (value or "").strip()).strip("_")
    return cleaned[:60] or "Unknown"


def run_folder_for(jd: engine.JobPosting, *, isolated: bool,
                   unique: bool = False) -> Path:
    """Where one run writes. Production folders are unique PER EXECUTION.

    A fingerprint cannot supply that uniqueness: the same posting has the same
    fingerprint, so a same-day rerun would collide and overwrite the earlier
    run's artifacts. The wall-clock time to microseconds does distinguish them.
    """
    parts = [sanitize(jd.company_name or "Unknown_Company"),
             sanitize(jd.job_title or "Unknown_Role")]
    if jd.job_id:
        parts.append(sanitize(jd.job_id))
    parts.append(date.today().isoformat())
    if unique:
        parts.append(datetime.now().strftime("%H%M%S%f"))
    root = engine.SMOKE_OUTPUT_DIR if isolated else engine.OUTPUT_DIR
    return root / "_".join(parts)


def make_run_dir(jd: engine.JobPosting, *, isolated: bool) -> Path:
    """Create the run directory, never reusing a production one.

    Mock/smoke runs keep their stable per-day folder. A production run gets a
    fresh directory or fails before anything is written, so no artifact from an
    earlier execution can be overwritten.
    """
    if isolated:
        run_dir = run_folder_for(jd, isolated=True)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    for _ in range(10):
        run_dir = run_folder_for(jd, isolated=False, unique=True)
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
        except FileExistsError:
            continue
    raise RuntimeError("could not allocate a unique production run directory; refusing "
                       "to write into an existing one")


# ============================================================= verification


def header_tech_leaks(header: str, master: engine.MasterFacts) -> list[str]:
    """Catch a technology stack that leaked into a project header."""
    vocabulary = {engine.fold_term(t) for project in master.projects for t in project.tech}
    vocabulary |= {engine.fold_term(s.name) for s in master.skills}
    vocabulary.discard("")
    leaks = []
    for segment in [s.strip() for s in header.split("|")][1:]:
        body = re.split(r"\s{2,}", segment)[0]
        parts = [p.strip() for p in body.split(",") if p.strip()]
        hits = [p for p in parts if engine.fold_term(p) in vocabulary]
        if len(hits) >= 2 or (len(parts) >= 2 and hits):
            leaks.append(f"{segment.strip()!r} (technology terms: {', '.join(hits)})")
    return leaks


@dataclass
class Verification:
    pages: int
    skills_rendered_lines: int
    experience_exact: bool
    project_headers_clean: bool
    issues: list[str] = field(default_factory=list)
    action_verbs: dict[str, int] = field(default_factory=dict)
    action_verbs_ok: bool = True

    def as_dict(self) -> dict:
        return {"pages": self.pages, "skills_rendered_lines": self.skills_rendered_lines,
                "experience_exact": self.experience_exact,
                "project_headers_clean": self.project_headers_clean,
                "action_verbs": self.action_verbs,
                "action_verbs_ok": self.action_verbs_ok, "issues": self.issues}


def _project_id_for_header(header: str, master: engine.MasterFacts) -> str | None:
    """Match a rendered project header back to its project id by title."""
    rendered = engine.normalize_plain(header)
    best: tuple[str, str] | None = None
    for project in master.projects:
        name = engine.normalize_plain(project.name)
        if name and rendered.startswith(name) and (best is None or len(name) > len(best[1])):
            best = (project.project_id, name)
    return best[0] if best else None


def verify(pdf_path: Path, decision: engine.ExperienceDecision, policy: engine.Policy,
           master: engine.MasterFacts, skills_labels: list[str], order: list[str],
           projects: dict, log: StageLog) -> Verification:
    """Independent verification of the FINAL compiled PDF. The PDF is the truth."""
    lines = pdf.extract_lines(pdf_path)
    right_edge = pdf.body_right_edge(lines)
    pages = pdf.page_count(pdf_path)
    issues: list[str] = []

    experience = pdf.extract_bullets(lines, EXPERIENCE_HEADING, right_edge)
    exact, problems = engine.verify_experience_in_pdf([b.text for b in experience], decision)
    log.info("experience bullets found in the PDF: %d", len(experience))
    for bullet in experience:
        log.info("  rendered %d line(s): %s", bullet.lines, bullet.text[:96])
    if exact:
        log.info("EXPERIENCE VERIFIED: every bullet matches approved wording exactly")
    else:
        for problem in problems:
            log.error("EXPERIENCE MISMATCH: %s", problem)
        issues.extend(f"experience wording not approved: {p.splitlines()[0]}" for p in problems)

    categories = pdf.measure_skill_categories(lines, skills_labels, right_edge)
    rendered = pdf.total_skill_lines(categories)
    for metric in categories:
        log.info("  skills category %-24s %d line(s), last line %.0f%% full",
                 metric.label, metric.lines, metric.last_line_fill)
    log.info("skills rendered lines: %d (target %d)", rendered, policy.skills_target_lines)
    if rendered != policy.skills_target_lines:
        issues.append(f"Technical Skills renders {rendered} lines, "
                      f"target is exactly {policy.skills_target_lines}")

    blocks = pdf.project_blocks(lines, PROJECTS_HEADING, right_edge)
    headers = [header for header, _ in blocks]
    clean = True
    for header in headers:
        leaks = header_tech_leaks(header, master)
        log.info("  project header: %s", header)
        if leaks:
            clean = False
            for leak in leaks:
                log.error("PROJECT HEADER contains a technology stack: %s", leak)
                issues.append(f"project header carries a tech stack: {leak}")
    if clean:
        log.info("PROJECT HEADERS VERIFIED: no technology stacks, titles and context only")
    if len(headers) != policy.project_count:
        issues.append(f"PDF shows {len(headers)} project headers, expected {policy.project_count}")

    # Display order is chronological; the bullet allocation belongs to the
    # project id, so both are checked against the rendered document.
    expected_display = list(projects.get("display_order") or [])
    allocation = dict(projects.get("allocation") or {})
    rendered_ids = [_project_id_for_header(header, master) for header in headers]
    if expected_display:
        if rendered_ids == expected_display:
            log.info("PROJECT ORDER VERIFIED: rendered order %s matches the chronological "
                     "display order", " -> ".join(expected_display))
        else:
            log.error("PROJECT ORDER MISMATCH: rendered %s, expected %s",
                      rendered_ids, expected_display)
            issues.append(f"PDF project order {rendered_ids} does not match the computed "
                          f"chronological display order {expected_display}")
    for position, (project_id, (header, block)) in enumerate(zip(rendered_ids, blocks), start=1):
        earned = allocation.get(project_id)
        log.info("  display_position=%d project=%s bullets_rendered=%d allocation=%s",
                 position, project_id, len(block), earned)
        if earned is not None and len(block) != earned:
            issues.append(f"project {project_id} renders {len(block)} bullet(s) but its "
                          f"relevance rank earned {earned}")
    if allocation:
        top = max(allocation, key=lambda pid: (allocation[pid], -expected_display.index(pid)
                                               if pid in expected_display else 0))
        rendered_top = dict(zip(rendered_ids, (len(b) for _h, b in blocks))).get(top)
        position = (expected_display.index(top) + 1) if top in expected_display else None
        log.info("most relevant project %s keeps %s bullet(s) at display position %s",
                 top, rendered_top, position)
        if rendered_top != allocation[top]:
            issues.append(f"the most relevant project {top} renders {rendered_top} bullet(s), "
                          f"expected {allocation[top]}")

    bullets = pdf.extract_bullets(lines, PROJECTS_HEADING, right_edge)
    log.info("project bullets: %d rendered, %d lines total",
             len(bullets), sum(b.lines for b in bullets))
    expected_total = sum(allocation.values()) if allocation else sum(policy.bullet_allocation)
    if len(bullets) != expected_total:
        issues.append(f"PDF shows {len(bullets)} project bullets, expected {expected_total}")

    if pages != policy.page_count:
        issues.append(f"resume is {pages} page(s), must be exactly {policy.page_count}")

    in_pdf = pdf.section_order_in_pdf(lines)
    expected = [name.upper() for name in order]
    normalized = [s.upper() for s in in_pdf]
    log.info("section order in the PDF: %s", " -> ".join(in_pdf))
    if normalized[:len(expected)] != expected:
        issues.append(f"PDF section order {normalized} does not match the decision {expected}")

    # The action-verb limit is a property of the finished document, so it is
    # counted from the extracted PDF across Experience and Projects together.
    verb_counts = engine.count_opening_verbs([b.text for b in experience]
                                             + [b.text for b in bullets])
    log.info("action verb counts: %s",
             ", ".join(f"{verb}x{uses}" for verb, uses
                       in sorted(verb_counts.items(), key=lambda kv: (-kv[1], kv[0]))))
    violations = engine.verb_violations(verb_counts, policy.verb_max_uses)
    for verb, uses in violations:
        log.error("ACTION VERB REPETITION: %r starts %d bullets, the limit is %d",
                  verb, uses, policy.verb_max_uses)
        issues.append(f"action verb {verb!r} starts {uses} bullets, the limit is "
                      f"{policy.verb_max_uses}")
    if not violations:
        log.info("action verb repetition verified: no verb starts more than %d bullet(s)",
                 policy.verb_max_uses)

    text = pdf.extract_text(pdf_path)
    artifacts = pdf.extraction_artifacts(text)
    if artifacts:
        issues.append(f"PDF extraction artifacts: {artifacts[:4]}")

    return Verification(pages, rendered, exact, clean, issues,
                        action_verbs=dict(sorted(verb_counts.items())),
                        action_verbs_ok=not violations)


# ============================================================ the main run


def run_one(jd_path: Path, *, mock: bool, smoke: bool, console: bool = True) -> RunResult:
    credentials = engine.load_credentials()
    logger, buffers = build_logger(credentials.secrets, console=console)
    buffer = buffers[0]
    log = StageLog(logger, "INPUT")
    result = RunResult(jd_path=jd_path)
    file_handler = None

    try:
        master = engine.load_master()
        policy = engine.load_policy()
        template = engine.load_template()

        jd = engine.read_jd(jd_path)
        isolated = mock or smoke
        if not isolated:
            # Fail closed before writing anything or calling any provider: a
            # damaged skip source would silently re-apply to recorded postings.
            load_index(engine.PROCESSED_INDEX)
        run_dir = make_run_dir(jd, isolated=isolated)
        result.run_dir = run_dir
        file_handler = buffer.attach_file(run_dir / "run.log", logger)

        job = log.stage_log("JOB")
        job.info("source file: %s", jd_path)
        job.info("company=%r title=%r job_id=%r", jd.company_name, jd.job_title, jd.job_id)
        job.info("jd fingerprint: %s", jd.fingerprint)
        job.info("mode: %s", "mock (no API calls)" if mock else
                 ("smoke test (real providers, isolated output)" if smoke else "production"))
        job.info("run folder: %s", run_dir)
        (run_dir / "job_description.txt").write_text(jd.text, encoding="utf-8")

        log.info("master: %d projects, %d claimable skills, %d evidence capsules",
                 len(master.projects), len(master.skills), len(master.capsules))
        log.info("template: %s with blocks %s", template.path.name, sorted(template.blocks))
        log.info("credentials: %s", credentials.describe())

        pol = log.stage_log("POLICY")
        pol.info("policy source: %s (read fresh from disk)", engine.POLICY_PATH.name)
        pol.info("swap rules: %d role rows, %d JD-signal rows",
                 len(policy.role_swaps), len(policy.signal_swaps))
        pol.info("mandatory skills block: %s",
                 "; ".join(f"{label} ({len(items)})" for label, items in policy.mandatory))
        pol.info("overflow removal priority: %s",
                 "; ".join(f"{cat} -> {frag}" for cat, frag in policy.overflow_removals))
        pol.info("targets from the spreadsheet: skills=%d rendered lines, projects=%d, "
                 "bullets=%s, pages=%d, action verb max=%d",
                 policy.skills_target_lines, policy.project_count, policy.bullet_allocation,
                 policy.page_count, policy.verb_max_uses)
        for fix in policy.data_fixes:
            pol.warning("SPREADSHEET DATA FIX: %s", fix)
        pol.info("section-order policy comes from the master (PART B); the current "
                 "spreadsheet does not define one")

        # The Experience id graph is validated before anything expensive: a bad
        # mapping must never be discovered after spending provider calls.
        pol.info("experience library: %d base bullet(s) across %d role(s), %d alternate(s), "
                 "%d swap rule(s)", len(policy.experience_bullet_ids),
                 sum(1 for e in policy.experience_library if e.is_role_header),
                 len(policy.alternate_library), len(policy.swap_rules))
        integrity = engine.verify_experience_ids(policy)
        if integrity:
            for problem in integrity:
                pol.error("ERROR %s", problem)
            raise engine.PolicyError(
                f"Experience id integrity failed with {len(integrity)} problem(s); "
                f"refusing to run before any provider call")
        pol.info("EXPERIENCE ID INTEGRITY VERIFIED")

        signals = engine.classify_jd(jd.text, master.section_order)
        pol.info("role family: %s (domain scores %s)", signals.role_family, signals.domain_scores)
        pol.info("healthcare: %s - %s", signals.healthcare, signals.healthcare_reason)
        pol.info("code-quality emphasis: %s - %s", signals.code_quality,
                 signals.code_quality_reason)
        pol.info("agentic AI emphasis: %s | scrum emphasis: %s",
                 signals.jd_emphasizes_agentic_ai, signals.jd_emphasizes_scrum)

        requirements = engine.extract_jd_requirements(jd.text, master)
        scored_reqs = engine.scored_requirements(requirements)
        manual_reqs = engine.manual_review_requirements(requirements)
        signal_reqs = engine.role_signals(requirements)
        themes = engine.jd_themes(requirements, 5)
        kinds: dict[str, int] = {}
        for req in requirements:
            kinds[req.kind] = kinds.get(req.kind, 0) + 1
        pol.info("extracted %d JD requirement(s): %d scored, %d manual review, %d role "
                 "signal(s) | kinds: %s", len(requirements), len(scored_reqs),
                 len(manual_reqs), len(signal_reqs),
                 ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))
        for req in requirements:
            pol.info("  %s [%s/%s/%s] %s", req.requirement_id, req.kind, req.authority,
                     req.importance, req.original_text[:104])
        for req in manual_reqs:
            pol.info("  %s routed to manual review (%s): never scored",
                     req.requirement_id, req.kind)

        # A substantive posting that yields nothing means deterministic
        # understanding failed. Stop before spending any provider call.
        # Volume of prose, not newline count: a real posting pasted as one long
        # paragraph is still substantive.
        words = len(re.findall(r"[A-Za-z][A-Za-z0-9'\-/+.]*", jd.text))
        sentences = len([p for p in re.split(r"(?<=[.!?])\s+", jd.text) if p.strip()])
        substantive = len(jd.text) >= 800 and words >= 120 and sentences >= 5
        if substantive and not requirements:
            pol.error("ERROR substantive JD produced zero requirements")
            final = log.stage_log("FINAL")
            final.info("resume generation skipped: requirement extraction failed")
            final.info("status=needs_review")
            result.status = "needs_review"
            result.issues.append(
                "substantive JD produced zero requirements; extraction must be fixed "
                "before this posting is processed (no provider calls were made)")
            (run_dir / "strategy.json").write_text(json.dumps({
                "status": "needs_review", "company_name": jd.company_name,
                "job_title": jd.job_title, "job_id": jd.job_id,
                "blocked_before_providers": True,
                "reason": "requirement extraction produced zero requirements",
                "jd_fingerprint": jd.fingerprint,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
            }, indent=2), encoding="utf-8")
            return result

        pol.info("top themes:")
        for index, req in enumerate(themes, start=1):
            pol.info("  theme %d [%s] %s%s", index, req.importance, req.headline[:96],
                     f" (names: {', '.join(req.terms)})" if req.terms else "")
        grad_requirement = engine.graduation_requirement(jd.text)
        if grad_requirement:
            pol.info("posting states a graduation requirement: %s", grad_requirement["text"])

        order_log = log.stage_log("SECTION ORDER")
        order_log.info("graduate-recruiting score %d (threshold %d): %s",
                       signals.graduate_score, master.section_order.threshold,
                       signals.graduate_reason)
        order = master.section_order.order_for(signals.section_mode)
        order_log.info("decision: %s -> %s", signals.section_mode, " -> ".join(order))
        order_log.info("Python owns this decision; no model input is consulted")

        # ---- experience (deterministic, protected) ----------------------
        exp_log = log.stage_log("EXPERIENCE")
        decision = engine.select_experience(template, policy, signals)
        exp_log.info("rule applied: %s", decision.rule)
        for bullet_id, action, latex in decision.shipped:
            if action == "SWAP":
                swap = next(s for s in decision.swaps if s[0] == bullet_id)
                exp_log.info("%s action=SWAP target_id=%s rule=%r text=%r",
                             bullet_id, swap[1], swap[2], engine.latex_to_plain(latex))
            else:
                exp_log.info("%s action=KEEP_EXACT text=%r",
                             bullet_id, engine.latex_to_plain(latex))
        exp_log.info("shipped ids: %s", ", ".join(decision.shipped_ids))
        exp_log.info("wording comes from %s, never from the template; no model has authority "
                     "over this section (%d approved variants exist)",
                     engine.POLICY_PATH.name, len(decision.allowed_plain))

        client = llm_client.build_client(master, policy, log.stage_log("PROVIDER"),
                                        mock=mock, credentials=credentials)

        # ---- project selection (LLM) ------------------------------------
        sel_log = log.stage_log("PROJECT SELECTION")
        selection = client.select_projects(jd.text, signals, policy.project_count)
        chosen = [master.project(pid) for pid in selection.selected]
        allocation = engine.allocate_bullets(selection.selected, policy)
        for pid, selected, reason in selection.considered:
            sel_log.info("project_id=%s selected=%s llm_rank=%s reason=%s", pid,
                         "yes" if selected else "no", selection.ranks.get(pid, "-"),
                         reason[:120] or "-")
        for rank, pid in enumerate(selection.selected, start=1):
            sel_log.info("relevance_rank=%d project=%s bullets=%d reason=%s",
                         rank, pid, allocation[pid], selection.reasons[pid][:140])
        if selection.career_stage:
            sel_log.info("model-reported career stage: %s (advisory only; section order is "
                         "already decided)", selection.career_stage)

        # Selection and allocation are relevance-driven. Rendering order is
        # chronological, and the allocation travels with the project id.
        display = engine.display_order(selection.selected, master)
        order_log = log.stage_log("PROJECT ORDER")
        for position, pid in enumerate(display, start=1):
            order_log.info("display_position=%d project=%s date=%s bullets=%d relevance_rank=%d",
                           position, pid, master.project(pid).date, allocation[pid],
                           selection.ranks[pid])
        order_log.info("chronology decides display only; it never influenced which three "
                       "projects were selected or how many bullets each earned")

        # ---- project bullets (LLM, grounded) ----------------------------
        write_log = log.stage_log("PROJECT WRITING")
        renders: list[engine.ProjectRender] = []
        grounding_problems: list[grounding.Problem] = []
        verb_counts = engine.verb_usage([l for _, _, l in decision.shipped], [])
        write_log.info("action verbs already used by Experience: %s",
                       ", ".join(f"{v}x{n}" for v, n in sorted(verb_counts.items())))
        for project in chosen:
            count = allocation[project.project_id]
            bullets, problems = client.write_bullets(project, count, jd.text, verb_counts,
                                                     BULLET_CHAR_TARGET)
            for bullet in bullets:
                verb = engine.opening_verb(bullet)
                verb_counts[verb] = verb_counts.get(verb, 0) + 1
                write_log.info("%s bullet: %s", project.project_id, bullet)
            blocking = grounding.errors(problems)
            # A write-time verb complaint is provisional: the action-verb limit is
            # a property of the finished document, so targeted repair still gets a
            # chance and the final PDF verification decides it. Factual problems
            # are terminal here.
            provisional = [p for p in blocking if p.kind == "verb_budget"]
            factual = [p for p in blocking if p.kind != "verb_budget"]
            grounding_problems.extend(factual)
            for problem in factual:
                write_log.error("%s UNRESOLVED: %s", project.project_id, problem)
            for problem in provisional:
                write_log.info("%s provisional: %s; deferring to targeted repair and the "
                               "final PDF verification", project.project_id, problem)
            if not blocking:
                write_log.info("%s grounding clean (%d bullet(s) verified against this "
                               "project's evidence only)", project.project_id, len(bullets))
            renders.append(engine.ProjectRender(project, bullets))

        by_id = {render.project.project_id: render for render in renders}
        renders = [by_id[pid] for pid in display]
        write_log.info("bullets were written in relevance order (%s) and will render in "
                       "chronological order (%s)", ", ".join(selection.selected),
                       ", ".join(display))

        # ---- skills + layout, measured on the real PDF ------------------
        skills_log = log.stage_log("SKILLS")
        state = engine.build_skills(master, policy, signals, jd.text, chosen, model_ranked=[])
        skills_log.info("baseline from the spreadsheet mandatory block:")
        for line in state.lines():
            skills_log.info("  %s", line)
        for note in state.withheld:
            skills_log.warning("POLICY: %s", note)
        project_candidates = [c.name for c in state.candidates if c.source.startswith("project")]
        jd_candidates = [c.name for c in state.candidates if c.source.startswith("jd")]
        skills_log.info("selected_project_candidates: %s", ", ".join(project_candidates) or "none")
        skills_log.info("experience_jd_candidates: %s", ", ".join(jd_candidates) or "none")

        layout = fit_to_page(run_dir, template, order, decision, state, renders, policy,
                             master, client, skills_log, log.stage_log("LAYOUT"))
        if not layout["ok"]:
            result.status = "failed"
            result.issues.append(layout["reason"])
            log.stage_log("FINAL").error("could not produce a compiled PDF: %s", layout["reason"])
            return result

        pdf_path = layout["pdf_path"]
        pdf_log = log.stage_log("PDF")
        pdf_log.info("compiled %s after %d iteration(s)", pdf_path.name, layout["iterations"])
        text = pdf.extract_text(pdf_path)
        (run_dir / "resume.txt").write_text(text, encoding="utf-8")
        pdf_log.info("extracted text written to resume.txt (%d characters)", len(text))

        # ---- independent verification -----------------------------------
        verification = verify(pdf_path, decision, policy, master, state.lines(), order,
                              {"display_order": display, "allocation": allocation},
                              log.stage_log("VERIFY"))

        # ---- cover letter ------------------------------------------------
        letter_log = log.stage_log("COVER LETTER")
        letter_problems: list[grounding.Problem] = []
        letter = ""
        try:
            letter, letter_problems = client.cover_letter(
                jd, signals, [engine.latex_to_plain(l) for _, _, l in decision.shipped],
                chosen, themes=themes)
            (run_dir / "cover_letter.txt").write_text(letter, encoding="utf-8")
            provider = next((c.provider for c in reversed(client.calls)
                             if c.purpose == "cover_letter"), "unknown")
            touched = grounding.letter_themes_covered(letter, [r.headline for r in themes])
            named_terms = sorted({term for req in themes for term in req.terms})
            named_covered = grounding.letter_named_terms(letter, named_terms)
            sources = [name for name, probe in (
                ("experience", any(engine.fold(w) in engine.fold(letter) for w in
                                   ("Thesis Mumbai", "HeinOnline", "Data Maven",
                                    "Go Digital"))),
                ("project", any(engine.fold(p.name.split("(")[0]) in engine.fold(letter)
                                or engine.fold(p.project_id) in engine.fold(letter)
                                for p in chosen)),
                ("education", "University at Buffalo" in letter or "Master" in letter),
            ) if probe]
            letter_log.info("provider=%s", provider)
            letter_log.info("words=%d characters=%d", grounding.word_count(letter), len(letter))
            letter_log.info("JD themes touched=%d/%d%s", len(touched), len(themes),
                            (": " + "; ".join(t[:60] for t in touched)) if touched else "")
            letter_log.info("named_terms_covered=%d/%d%s", len(named_covered),
                            len(named_terms),
                            (": " + ", ".join(named_covered)) if named_covered else "")
            missing_terms = [t for t in named_terms if t not in named_covered]
            if missing_terms:
                letter_log.info("named terms not mentioned (acceptable, depth over "
                                "keywords): %s", ", ".join(missing_terms))
            letter_log.info("evidence sources=%s", ", ".join(sources) or "none detected")
            for entry in grounding.classify_letter_metrics(
                    letter, master, jd_text=jd.text, company=jd.company_name,
                    masked_terms=(jd.job_id,) if jd.job_id else ()):
                letter_log.info('metric source=%s value="%s"', entry.source, entry.display)
            for problem in letter_problems:
                (letter_log.error if problem.severity == "error" else letter_log.warning)(
                    "%s", problem)
            if not grounding.errors(letter_problems):
                letter_log.info("grounding clean: every metric, threshold and ownership claim "
                                "traces to an authorized source (candidate Master evidence "
                                "or attributed job-description evidence)")
                letter_log.info("quality checks passed")
                letter_log.info("validation=pass")
            else:
                letter_log.error("validation=fail")
        except llm_client.ProviderError as error:
            letter_log.error("cover letter unavailable: category=%s %s", error.category, error)
            letter_problems = [grounding.Problem("provider", "error", str(error))]

        # ---- assessment --------------------------------------------------
        assess_log = log.stage_log("ASSESSMENT")
        assessment: dict = {}
        assessment_available = False
        assessment_error = ""
        # Bound before the try: the except path and the strategy build both
        # reference it, and assess() can raise before any assignment inside.
        known_ids = {req.requirement_id for req in requirements}
        # Bound before the try: the issues list references it even when assess()
        # raises, exactly like known_ids above.
        final_problems: list[grounding.Problem] = []
        try:
            tailoring_signals = {
                "skills_rendered_lines": verification.skills_rendered_lines,
                "experience_exact": verification.experience_exact,
                "pages": verification.pages,
                "projects": ", ".join(selection.selected),
                "issues": verification.issues,
            }
            assessment = client.assess(
                jd, signals, chosen, state.all_skills(),
                experience_plain=[engine.latex_to_plain(l) for _, _, l in decision.shipped],
                project_bullets={r.project.project_id: list(r.bullets) for r in renders},
                requirements=requirements, tailoring=tailoring_signals,
                extra_experience=(
                    [engine.latex_to_plain(entry.latex)
                     for entry in policy.experience_library if entry.is_role_header]
                    + [engine.latex_to_plain(" ".join(
                        template.blocks["Education"].split()))]))
            # TWO validation boundaries. The provider contract was enforced
            # inside assess(), BEFORE finalization. This is the second gate: a
            # deterministic integrity check on Python's own final output. The
            # provider's prose rules are never re-applied here, because they
            # would reject Python's deterministic summary.
            table = engine.requirement_table(requirements)
            final_problems = grounding.errors(grounding.validate_final_assessment(
                assessment, verdicts=getattr(client, "last_verdicts", {}) or {},
                table=table, requirements=requirements))
            for problem in final_problems:
                assess_log.error("final integrity: %s", problem.message)
            if final_problems:
                assess_log.error("final assessment validation FAILED with %d problem(s); "
                                 "this run needs review and is not recorded",
                                 len(final_problems))
            else:
                assess_log.info("final assessment validation clean "
                                "(requirement authority, verdict consistency, entry "
                                "fields, eligibility, cap, summary, schema)")
            # Verdicts, eligibility, the cap, breadth and the deterministic
            # summary were all applied inside assess(), in that order. Log the
            # outcome; do not re-apply it.
            capped = assessment.get("recommendation_capped")
            if capped:
                assess_log.info("recommendation capped %s -> %s reason=%s",
                                capped.get("from"), capped.get("to"), capped.get("reason"))
            assess_log.info("verdict_source=%s (Python deterministic; provider prose kept "
                            "only as model_summary)", assessment.get("verdict_source"))
            provider = next((c.provider for c in reversed(client.calls)
                             if c.purpose == "assessment"), "unknown")
            eligibility = assessment.get("eligibility") or {}
            tailoring_quality = assessment.get("tailoring_quality") or {}
            assess_log.info("provider=%s", provider)
            assess_log.info("fit_score=%s", assessment.get("fit_score"))
            assess_log.info("recommendation=%s", assessment.get("recommendation"))
            assess_log.info("scored_requirement_count=%s role_signal_count=%s "
                            "manual_review_count=%s fit_confidence=%s",
                            assessment.get("scored_requirement_count"),
                            assessment.get("role_signal_count"),
                            assessment.get("manual_review_count"),
                            assessment.get("fit_confidence"))
            assess_log.info("strong_matches=%d partial_matches=%d gaps=%d",
                            len(assessment.get("strong_matches") or []),
                            len(assessment.get("partial_matches") or []),
                            len(assessment.get("gaps") or []))
            assess_log.info("eligibility=%s%s", eligibility.get("status"),
                            (": " + "; ".join(eligibility.get("details") or []))
                            if eligibility.get("details") else "")
            assess_log.info("tailoring_quality=%s%s", tailoring_quality.get("score"),
                            (": " + "; ".join(tailoring_quality.get("notes") or []))
                            if tailoring_quality.get("notes") else "")
            assess_log.info("summary: %s", assessment.get("summary"))
            for match in (assessment.get("strong_matches") or [])[:8]:
                assess_log.info("strong: %s <- %s (%s)", match.get("requirement"),
                                match.get("evidence"), match.get("source"))
            for match in assessment.get("partial_matches") or []:
                assess_log.info("partial: %s (%s)", match.get("requirement"),
                                match.get("limitation"))
            for gap in assessment.get("gaps") or []:
                assess_log.info("gap: %s [%s/%s]", gap.get("requirement"),
                                gap.get("importance"), gap.get("status"))
            for entry in assessment.get("manual_review") or []:
                assess_log.info("manual review: %s [%s] %s", entry.get("requirement_id"),
                                entry.get("kind"), entry.get("reason"))
            for strength in assessment.get("complementary_strengths") or []:
                assess_log.info("complementary strength (not a requirement): %s", strength)
            for flag in assessment.get("risk_flags") or []:
                assess_log.warning("risk flag: %s", flag)
            if not (assessment.get("gaps") or []):
                assess_log.info("no unsupported requirement found; gaps array is "
                                "intentionally empty")
            assessment_available = True
        except llm_client.ProviderError as error:
            assess_log.warning("assessment unavailable: category=%s %s", error.category, error)
            assessment = {"error": f"{error.category}: {error}"}
            assessment_available = False
            assessment_error = f"{error.category}: {error}"
        (run_dir / "assessment.json").write_text(json.dumps(assessment, indent=2),
                                                 encoding="utf-8")

        # ---- status ------------------------------------------------------
        issues = list(verification.issues)
        issues += [f"assessment integrity: {p.message}" for p in final_problems]
        # A substantive production posting whose title never resolved is not a
        # clean success: the run folder and the tracking row would both be wrong,
        # so it goes to review rather than being recorded. Same substantive test
        # as the pre-provider guard, recomputed from the text here.
        if not mock and not smoke and not jd.job_title:
            jd_words = len(re.findall(r"[A-Za-z][A-Za-z0-9'\-/+.]*", jd.text))
            jd_sentences = len([s for s in re.split(r"(?<=[.!?])\s+", jd.text) if s.strip()])
            if len(jd.text) >= 800 and jd_words >= 120 and jd_sentences >= 5:
                issues.append("job title could not be resolved from a substantive posting; "
                              "metadata needs review before this run is recorded")
        issues += [f"project bullet grounding: {p.message}" for p in grounding_problems]
        issues += [f"cover letter: {p.message}" for p in grounding.errors(letter_problems)]
        issues += layout["issues"]
        # The resume can be perfect while a requested downstream artifact is
        # unusable. That package is not a clean success.
        if not assessment_available:
            issues.append(f"assessment unavailable: {assessment_error}")
        status = "success" if not issues else "needs_review"

        strategy = {
            "company_name": jd.company_name,
            "job_title": jd.job_title,
            "job_id": jd.job_id,
            "role_family": signals.role_family,
            "section_order": order,
            "experience": {
                "kept": [bid for bid, action, _ in decision.shipped if action == "KEEP_EXACT"],
                "swaps": [{"source_id": bid, "target_id": rid, "rule": rule}
                          for bid, rid, rule in decision.swaps],
                "shipped_ids": decision.shipped_ids,
                "rule": decision.rule,
                "wording_source": engine.POLICY_PATH.name,
            },
            "projects": {
                "selected_by_relevance": selection.selected,
                "display_order": display,
                "allocation": {pid: allocation[pid] for pid in selection.selected},
                "selection_reasons": {pid: selection.reasons[pid] for pid in selection.selected},
                "selected": selection.selected,
                "selected_is": ("relevance order, identical to selected_by_relevance; the PDF "
                                "renders display_order instead"),
            },
            "skills": {
                "baseline": [f"{label}: {', '.join(items)}" for label, items in policy.mandatory],
                "project_candidates": project_candidates,
                "experience_jd_candidates": jd_candidates,
                "added": [{"skill": name, "source": source, "reason": reason}
                          for name, source, reason in state.added],
                "removed": [{"skill": name, "reason": reason} for name, reason in state.removed],
                "rendered": state.lines(),
                "final_lines": verification.skills_rendered_lines,
            },
            "verification": verification.as_dict(),
            "status": status,
            "mode": "mock" if mock else ("smoke_test" if smoke else "production"),
            "provider_calls": [record.as_dict() for record in client.calls],
            "provider_accounting": {
                "generations": len(client.calls),
                "transport_ok": sum(1 for c in client.calls if c.transport_ok),
                "accepted": sum(1 for c in client.calls if c.accepted),
                "rejected": sum(1 for c in client.calls if c.transport_ok and not c.accepted),
            },
            "cover_letter": {
                "words": grounding.word_count(letter) if letter else 0,
                "provider": next((c.provider for c in reversed(client.calls)
                                  if c.purpose == "cover_letter"), None),
                "jd_themes": [r.headline for r in themes],
                "themes_touched": grounding.letter_themes_covered(
                    letter, [r.headline for r in themes]) if letter else [],
                "named_terms": sorted({t for r in themes for t in r.terms}),
                "named_terms_covered": grounding.letter_named_terms(
                    letter, sorted({t for r in themes for t in r.terms})) if letter else [],
                "validation": "pass" if letter and not grounding.errors(letter_problems)
                              else "fail",
            },
            "requirements": {
                "extracted": [req.as_dict() for req in requirements],
                "scored_ids": [req.requirement_id for req in scored_reqs],
                "manual_review_ids": [req.requirement_id for req in manual_reqs],
                "role_signal_ids": [req.requirement_id for req in signal_reqs],
                "kinds": kinds,
            },
            "assessment": {
                "assessment_available": assessment_available,
                "manual_review": len(assessment.get("manual_review") or []),
                "complementary_strengths": assessment.get("complementary_strengths") or [],
                "fit_score": assessment.get("fit_score"),
                "recommendation": assessment.get("recommendation"),
                "recommendation_capped": assessment.get("recommendation_capped"),
                "scored_requirement_count": assessment.get("scored_requirement_count"),
                "role_signal_count": assessment.get("role_signal_count"),
                "manual_review_count": assessment.get("manual_review_count"),
                "fit_confidence": assessment.get("fit_confidence"),
                "eligibility": (assessment.get("eligibility") or {}).get("status"),
                "tailoring_quality": (assessment.get("tailoring_quality") or {}).get("score"),
                "strong_matches": len(assessment.get("strong_matches") or []),
                "partial_matches": len(assessment.get("partial_matches") or []),
                "gaps": len(assessment.get("gaps") or []),
            },
            "jd_fingerprint": jd.fingerprint,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
        (run_dir / "strategy.json").write_text(json.dumps(strategy, indent=2), encoding="utf-8")
        result.strategy = strategy
        result.issues = issues
        result.status = status

        final = log.stage_log("FINAL")
        if not verification.issues:
            final.info("resume verification succeeded")
        if not assessment_available:
            final.info("assessment unavailable")
        final.info("assessment_available=%s", assessment_available)
        final.info("status=%s", status)
        for issue in issues:
            final.error("issue: %s", issue)
        if status == "success":
            final.info("every contract met: experience exact, %d skills lines, %d projects "
                       "(%s bullets), %d page",
                       verification.skills_rendered_lines, policy.project_count,
                       "/".join(str(allocation[p]) for p in selection.selected),
                       verification.pages)
        tracker = Tracker(engine.PROCESSED_CSV, engine.PROCESSED_INDEX, enabled=not isolated)
        tracker.record(jd, run_dir, status, log.stage_log("TRACKING"))
        provider_summary = ", ".join(
            f"{c.purpose}:{'accepted' if c.accepted else 'rejected'}"
            for c in client.calls) or "none"
        final.info("provider calls: %s", provider_summary)
        final.info("artifacts: %s", ", ".join(sorted(p.name for p in run_dir.iterdir())))
        return result

    except Exception as error:                       # noqa: BLE001 - reported, not hidden
        log.stage_log("FINAL").error("run failed: %s\n%s", error, traceback.format_exc())
        result.status = "failed"
        result.issues.append(f"{type(error).__name__}: {error}")
        return result
    finally:
        if file_handler:
            file_handler.close()
            logger.removeHandler(file_handler)


# ================================================= skills + layout fitting


def assemble(template: engine.Template, order: list[str], decision: engine.ExperienceDecision,
             state: engine.SkillsState, renders: list[engine.ProjectRender]) -> str:
    return template.render(order, experience=decision.latex_block, skills=state.latex(),
                           projects=engine.build_projects_latex(renders))


def fit_to_page(run_dir: Path, template: engine.Template, order: list[str],
                decision: engine.ExperienceDecision, state: engine.SkillsState,
                renders: list[engine.ProjectRender], policy: engine.Policy,
                master: engine.MasterFacts, client: llm_client.LLMClient,
                skills_log: StageLog, layout_log: StageLog) -> dict:
    """Drive Technical Skills to exactly the target rendered lines, then fix layout.

    Every decision here is measured on a freshly compiled PDF: character counts
    are never authoritative. Nothing is invented to reach a target - if the
    contract cannot be met truthfully, the caller reports needs_review.
    """
    tex_path = run_dir / f"{RESUME_STEM}.tex"
    issues: list[str] = []
    repairs: dict[tuple[str, int], int] = {}
    verb_repairs: dict[tuple[str, int], int] = {}
    iterations = 0
    target = policy.skills_target_lines
    skills_done_logged = False
    fill_exhausted_logged = False

    while iterations < engine.SKILLS_MAX_ITERATIONS:
        iterations += 1
        tex_path.write_text(assemble(template, order, decision, state, renders), encoding="utf-8")
        build = pdf.compile_pdf(tex_path, run_dir)
        if not build.ok:
            return {"ok": False, "reason": build.summary(), "iterations": iterations,
                    "issues": issues, "pdf_path": None}

        lines = pdf.extract_lines(build.pdf_path)
        right_edge = pdf.body_right_edge(lines)
        categories = pdf.measure_skill_categories(lines, state.lines(), right_edge)
        rendered = pdf.total_skill_lines(categories)
        pages = pdf.page_count(build.pdf_path)
        bullets = pdf.extract_bullets(lines, PROJECTS_HEADING, right_edge)

        skills_log.info("iteration=%d rendered_lines=%d target=%d pages=%d | %s",
                        iterations, rendered, target, pages,
                        "; ".join(f"{m.label}={m.lines}L@{m.last_line_fill:.0f}%"
                                  for m in categories))

        # 1. Technical Skills must render exactly `target` lines.
        if rendered > target:
            action = _shrink_skills(state, policy, categories, target, rendered, skills_log)
            if action:
                continue
            issues.append(f"Technical Skills renders {rendered} lines and the spreadsheet's "
                          f"removal priorities are exhausted")
            skills_log.error("no further truthful removal available at %d lines", rendered)
        elif rendered < target:
            action = _grow_skills(state, rendered, target, skills_log)
            if action:
                continue
            issues.append(f"Technical Skills renders {rendered} lines and no supported skill "
                          f"remains to add; a skill will NOT be invented to reach {target}")
            skills_log.error("cannot reach %d lines without inventing a skill", target)
        else:
            if not skills_done_logged:
                skills_log.info("TARGET_REACHED: Technical Skills renders exactly %d lines",
                                target)
                skills_done_logged = True
            # The spreadsheet also says a second line under the fill limit should
            # carry another supported skill. Filling an existing line costs no
            # extra line, and anything that does costs is reverted below.
            if _fill_skills(state, policy, categories, skills_log):
                continue
            short = [f"{m.label} ({m.last_line_fill:.0f}%)" for m in categories
                     if m.lines >= 2 and m.last_line_fill < policy.second_line_max_fill]
            if short and not fill_exhausted_logged:
                fill_exhausted_logged = True
                skills_log.info("no JD-relevant supported skill remains for %s; leaving the "
                                "line short rather than padding with unrelated evidence",
                                ", ".join(short))

        # 2. Layout: one page, two rendered lines per project bullet, no orphan tails.
        problems = _layout_problems(bullets, renders, policy, pages)
        for note in problems["notes"]:
            layout_log.info("%s", note)
        actions = list(problems["actions"])
        if not actions:
            # Layout is settled, so the remaining document-level invariant is
            # the action-verb limit. Experience is immutable, so only a
            # generated project bullet may be reworded.
            verb_action = _verb_action(lines, right_edge, renders, policy, layout_log)
            if verb_action:
                actions = [verb_action]
        if not actions:
            layout_log.info("layout contract met: %d page(s), %d project bullet lines",
                            pages, sum(b.lines for b in bullets))
            return {"ok": True, "pdf_path": build.pdf_path, "iterations": iterations,
                    "issues": issues}

        target_action = actions[0]
        budget = verb_repairs if target_action["goal"] == "reword_verb" else repairs
        key = (target_action["project_id"], target_action["bullet_index"])
        if budget.get(key, 0) >= MAX_REPAIRS_PER_BULLET:
            layout_log.warning("bullet %s#%d already repaired %d time(s); stopping here rather "
                               "than degrading it further", key[0], key[1] + 1, repairs[key])
            issues.append(f"{target_action['reason']} (bounded repair exhausted)")
            return {"ok": True, "pdf_path": build.pdf_path, "iterations": iterations,
                    "issues": issues}
        budget[key] = budget.get(key, 0) + 1

        render = next(r for r in renders if r.project.project_id == target_action["project_id"])
        current = render.bullets[target_action["bullet_index"]]
        siblings = [b for i, b in enumerate(render.bullets)
                    if i != target_action["bullet_index"]]
        layout_log.info("repair attempt %d on %s#%d (%s): %s", budget[key], key[0], key[1] + 1,
                        target_action["goal"], target_action["reason"])
        try:
            revised, problems_found = client.repair_bullet(
                render.project, current, target_action["goal"], target_action["reason"],
                siblings, available=target_action.get("available"))
        except llm_client.ProviderError as error:
            layout_log.error("repair call failed: category=%s %s", error.category, error)
            issues.append(f"layout repair unavailable: {error}")
            return {"ok": True, "pdf_path": build.pdf_path, "iterations": iterations,
                    "issues": issues}
        blocking = grounding.errors(problems_found)
        if blocking or revised == current:
            for problem in blocking:
                layout_log.error("rejected repair: %s", problem)
            if revised == current:
                layout_log.warning("the repair returned the bullet unchanged; no natural "
                                   "rewording is available without weakening the facts")
            issues.append(f"{target_action['reason']} (no truthful repair available)")
            return {"ok": True, "pdf_path": build.pdf_path, "iterations": iterations,
                    "issues": issues}
        layout_log.info("accepted repair: %s", revised)
        render.bullets[target_action["bullet_index"]] = revised

    issues.append(f"did not converge within {engine.SKILLS_MAX_ITERATIONS} iterations")
    return {"ok": True, "pdf_path": run_dir / f"{RESUME_STEM}.pdf", "iterations": iterations,
            "issues": issues}


def _shrink_skills(state: engine.SkillsState, policy: engine.Policy, categories, target: int,
                   rendered: int, log: StageLog) -> bool:
    """Remove one skill, own additions first, then the spreadsheet priorities."""
    addition = state.removable_addition()
    if addition:
        state.remove(addition, f"Technical Skills rendered {rendered} lines (target {target})")
        state.reject(addition, f"adding it rendered {rendered} lines instead of {target}")
        log.info("iteration action=REMOVE skill=%r reason=%s", addition,
                 f"adding it pushed the section to {rendered} rendered lines, past the "
                 f"{target}-line target")
        return True
    over = sorted(categories, key=lambda m: -m.lines)
    for metric in over:
        fragment = policy.removal_for(metric.label)
        if not fragment:
            continue
        for name in state.all_skills():
            if fragment.lower() in name.lower():
                state.remove(name, f"spreadsheet overflow priority for {metric.label}")
                log.info("iteration action=REMOVE skill=%r reason=%s", name,
                         f"spreadsheet removal priority for {metric.label} when the section "
                         f"exceeds the line target")
                return True
    return False


def _grow_skills(state: engine.SkillsState, rendered: int, target: int, log: StageLog) -> bool:
    """Add the next highest-priority supported skill. Never invent one."""
    while state.next_candidate < len(state.candidates):
        candidate = state.candidates[state.next_candidate]
        state.next_candidate += 1
        if state.add(candidate):
            log.info("iteration action=ADD skill=%r category=%r tier=%d reason=%s",
                     candidate.name, candidate.category, candidate.tier,
                     f"{candidate.reason}; section rendered {rendered} of {target} lines")
            return True
    return False


def _fill_skills(state: engine.SkillsState, policy: engine.Policy, categories,
                 log: StageLog) -> bool:
    """Add one supported skill to the emptiest under-filled second line.

    This is the spreadsheet's fill procedure, not line-count padding: the
    candidate must already be supported by the master, and relevance order
    (exact JD keyword, then JD-supported, then the rank-1 project's stack)
    decides which one goes in. Nothing is invented, and an addition that costs
    a ninth line is reverted on the next measurement.
    """
    under = sorted((m for m in categories
                    if m.lines >= 2 and m.last_line_fill < policy.second_line_max_fill),
                   key=lambda m: m.last_line_fill)
    for metric in under:
        for candidate in state.candidates:
            if candidate.category != metric.label:
                continue
            if candidate.tier > FILL_MAX_TIER:
                continue
            if state.is_rejected(candidate.name):
                continue
            if engine.fold_term(candidate.name) in state.keys():
                continue
            if state.add(candidate):
                log.info("iteration action=ADD skill=%r category=%r tier=%d reason=%s",
                         candidate.name, candidate.category, candidate.tier,
                         f"{candidate.reason}; {metric.label} second line was only "
                         f"{metric.last_line_fill:.0f}% full against the "
                         f"{policy.second_line_max_fill:.0f}% fill limit")
                return True
    return False


def _verb_action(lines, right_edge: float, renders: list[engine.ProjectRender],
                 policy: engine.Policy, log: StageLog) -> dict | None:
    """Target one generated project bullet when a verb exceeds the limit.

    Counted from the rendered PDF across Experience and Projects. Experience is
    approved content, so a repetition that lives only in Experience is reported
    rather than repaired.
    """
    experience = [b.text for b in pdf.extract_bullets(lines, EXPERIENCE_HEADING, right_edge)]
    project = [b.text for b in pdf.extract_bullets(lines, PROJECTS_HEADING, right_edge)]
    counts = engine.count_opening_verbs(experience + project)
    violations = engine.verb_violations(counts, policy.verb_max_uses)
    if not violations:
        return None

    verb, uses = violations[0]
    flat = [(render.project.project_id, index)
            for render in renders for index in range(len(render.bullets))]
    offenders = [position for position, text in enumerate(project)
                 if engine.opening_verb(engine.normalize_plain(text)) == verb]
    if not offenders or offenders[-1] >= len(flat):
        log.warning("action verb %r starts %d bullets but only approved Experience uses it; "
                    "Experience is never rewritten to fix repetition", verb, uses)
        return None

    project_id, index = flat[offenders[-1]]
    available = engine.available_verbs(counts, policy.verb_max_uses, grounding.PREFERRED_VERBS)
    log.info("action verb %r starts %d bullets (limit %d); rewording %s#%d, the last project "
             "bullet using it", verb, uses, policy.verb_max_uses, project_id, index + 1)
    return {"project_id": project_id, "bullet_index": index, "goal": "reword_verb",
            "available": available,
            "reason": (f"the verb {verb!r} starts {uses} bullets on this resume and the "
                       f"spreadsheet limit is {policy.verb_max_uses}")}


def _layout_problems(bullets, renders, policy: engine.Policy, pages: int) -> dict:
    """Compare rendered project bullets against the layout contract."""
    notes, actions = [], []
    flat: list[tuple[str, int]] = []
    for render in renders:
        for index in range(len(render.bullets)):
            flat.append((render.project.project_id, index))

    for position, bullet in enumerate(bullets):
        if position >= len(flat):
            break
        project_id, index = flat[position]
        verdict = bullet.verdict(policy.tail_orphan_max, policy.tail_acceptable_max,
                                 policy.tail_ideal_max)
        notes.append(f"{project_id}#{index + 1}: {bullet.lines} line(s), last line "
                     f"{bullet.fill_pct:.0f}% full -> {verdict}")
        if bullet.lines > policy.bullet_target_lines:
            actions.append({"project_id": project_id, "bullet_index": index, "goal": "shorten",
                            "reason": f"renders {bullet.lines} lines, the contract is "
                                      f"{policy.bullet_target_lines}"})
        elif bullet.lines < policy.bullet_target_lines:
            actions.append({"project_id": project_id, "bullet_index": index, "goal": "lengthen",
                            "reason": f"renders only {bullet.lines} line(s), the contract is "
                                      f"{policy.bullet_target_lines}"})
        elif verdict == "hard_orphan":
            actions.append({"project_id": project_id, "bullet_index": index, "goal": "shorten",
                            "reason": f"last line is only {bullet.fill_pct:.0f}% full, below the "
                                      f"{policy.tail_orphan_max:.0f}% orphan floor"})

    if pages > policy.page_count and not actions:
        longest = max(range(len(bullets)), key=lambda i: bullets[i].lines, default=None)
        if longest is not None and longest < len(flat):
            project_id, index = flat[longest]
            actions.append({"project_id": project_id, "bullet_index": index, "goal": "shorten",
                            "reason": f"resume runs to {pages} pages and must fit "
                                      f"{policy.page_count}"})
    return {"notes": notes, "actions": actions}


# ================================================================ revalidate


def revalidate(run_dir: Path, console: bool = True) -> RunResult:
    """Re-verify an existing run from its artifacts. Makes zero API calls."""
    logger, buffers = build_logger([], console=console)
    log = StageLog(logger, "REVALIDATE")
    result = RunResult(jd_path=run_dir, run_dir=run_dir)
    handler = buffers[0].attach_file(run_dir / "revalidation.log", logger)
    try:
        master = engine.load_master()
        policy = engine.load_policy()
        template = engine.load_template()
        jd_file = run_dir / "job_description.txt"
        if not jd_file.exists():
            raise engine.PolicyError(f"{run_dir} has no job_description.txt to revalidate against")
        jd = engine.read_jd(jd_file)
        signals = engine.classify_jd(jd.text, master.section_order)
        decision = engine.select_experience(template, policy, signals)
        order = master.section_order.order_for(signals.section_mode)
        log.info("re-derived the deterministic decisions from the stored JD: role family=%s, "
                 "rule=%r, swaps=%s", signals.role_family, decision.rule, decision.swap_summary)

        pdf_path = run_dir / f"{RESUME_STEM}.pdf"
        if not pdf_path.exists():
            tex = run_dir / f"{RESUME_STEM}.tex"
            if not tex.exists():
                raise engine.PolicyError(f"{run_dir} has neither resume.pdf nor resume.tex")
            build = pdf.compile_pdf(tex, run_dir)
            if not build.ok:
                raise engine.PolicyError(build.summary())
            pdf_path = build.pdf_path

        strategy_path = run_dir / "strategy.json"
        strategy = json.loads(strategy_path.read_text(encoding="utf-8")) \
            if strategy_path.exists() else {}
        labels = strategy.get("skills", {}).get("rendered") or _skills_labels_from_tex(run_dir)
        log.info("measuring %d recorded skill line(s) against the compiled PDF", len(labels))
        recorded = strategy.get("projects") or {}
        selected = (recorded.get("selected_by_relevance") or recorded.get("selected") or [])
        projects_meta = {
            "display_order": (recorded.get("display_order")
                              or (engine.display_order(selected, master) if selected else [])),
            "allocation": recorded.get("allocation") or {},
        }
        verification = verify(pdf_path, decision, policy, master, labels, order,
                              projects_meta, log)
        letter_file = run_dir / "cover_letter.txt"
        letter_problems = []
        if letter_file.exists():
            letter_problems = grounding.validate_cover_letter(
                letter_file.read_text(encoding="utf-8"), master,
                masked_terms=(jd.job_id,) if jd.job_id else (),
                banned=policy.banned_phrases)
            for problem in letter_problems:
                (log.error if problem.severity == "error" else log.warning)("letter %s", problem)

        issues = verification.issues + [f"cover letter: {p.message}"
                                        for p in grounding.errors(letter_problems)]
        result.status = "success" if not issues else "needs_review"
        for issue in issues:
            log.error("issue: %s", issue)
        log.info("revalidation status=%s (zero API calls)", result.status)
        (run_dir / "revalidation.json").write_text(
            json.dumps({"status": result.status, "verification": verification.as_dict(),
                        "issues": issues,
                        "revalidated_at": datetime.now().isoformat(timespec="seconds")},
                       indent=2), encoding="utf-8")
        result.issues = issues
        return result
    except Exception as error:                       # noqa: BLE001
        log.error("revalidation failed: %s", error)
        result.status = "failed"
        result.issues = [str(error)]
        return result
    finally:
        handler.close()
        logger.removeHandler(handler)


def _skills_labels_from_tex(run_dir: Path) -> list[str]:
    """Recover the emitted skill lines from the generated LaTeX."""
    tex = (run_dir / f"{RESUME_STEM}.tex")
    if not tex.exists():
        return []
    body = tex.read_text().split(r"\section*{TECHNICAL SKILLS}", 1)
    if len(body) != 2:
        return []
    section = body[1].split(r"\sectiondivider", 1)[0]
    labels = []
    for match in re.finditer(r"\\item \\textbf\{([^}]*):\}\s*([^\n]*)", section):
        label = match.group(1).replace("\\&", "&").replace("\\", "")
        body = engine.latex_to_plain(match.group(2))
        labels.append(f"{label}: {body}")
    return labels


# ===================================================================== batch


def discover_jobs(folder: Path, tracker: Tracker) -> tuple[list[Path], list[Path]]:
    """Split `folder`/*.txt into (pending, already processed).

    Non-recursive on purpose: jobs/trial holds regression fixtures that must
    never become production applications. Identity is the JD fingerprint from
    the existing tracking index, so renaming a file changes nothing and editing
    its text makes it pending again.
    """
    pending: list[Path] = []
    skipped: list[Path] = []
    for path in sorted(folder.glob("*.txt")):
        try:
            fingerprint = engine.read_jd(path).fingerprint
        except Exception:
            pending.append(path)        # unreadable: let the normal run report it
            continue
        if tracker.already_processed(fingerprint):
            skipped.append(path)
        else:
            pending.append(path)
    return pending, skipped


def run_batch(folder: Path, *, mock: bool = False, smoke: bool = False,
              csv_path: Path | None = None, index_path: Path | None = None) -> int:
    """Process every unprocessed JD in one folder. The only batch implementation.

    Writing production tracking stays governed by mock/smoke, so a mock or smoke
    batch never records anything. Skip-reads follow the index actually in use:
    with no explicit `index_path` a mock/smoke batch consults nothing and
    processes everything exactly as before, while an explicit index is always
    consulted so skipping is exercisable without touching production files.
    """
    # Two independent concerns. Writing production tracking stays governed by
    # mock/smoke exactly as before. Reading the index to skip work follows the
    # index actually in use: an explicitly supplied one is always consulted, so
    # skipping is real (and testable) without a mock batch ever touching or
    # reading the production index.
    writes = Tracker(csv_path or engine.PROCESSED_CSV,
                     index_path or engine.PROCESSED_INDEX,
                     enabled=not (mock or smoke))
    tracker = (Tracker(writes.csv_path, index_path, enabled=True)
               if index_path is not None
               else Tracker(writes.csv_path, writes.index_path,
                            enabled=not (mock or smoke)))
    found = sorted(folder.glob("*.txt"))
    if not found:
        print(f"no .txt job descriptions in {folder}", file=sys.stderr)
        return 2

    pending, skipped = discover_jobs(folder, tracker)
    print(f"[BATCH] found {len(found)} JD file(s)")
    print(f"[BATCH] already processed: {len(skipped)}")
    print(f"[BATCH] pending: {len(pending)}")
    for path in skipped:
        print(f"[BATCH] skip already processed: {path.name}")
    if not pending:
        print("[BATCH] no unprocessed JDs found")
        return 0

    results = []
    for position, path in enumerate(pending, start=1):
        print(f"[BATCH] processing {position}/{len(pending)}: {path.name}")
        results.append(run_one(path, mock=mock, smoke=smoke, console=False))

    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    print("[BATCH] complete")
    print(f"  successes: {counts.get('success', 0)}")
    print(f"  needs_review: {counts.get('needs_review', 0)}")
    print(f"  failed: {sum(n for s, n in counts.items() if s not in ('success', 'needs_review'))}")
    print(f"  skipped: {len(skipped)}")
    for result in results:
        print(f"  {result.status:12s} {result.jd_path.name:34s} "
              f"{result.run_dir.name if result.run_dir else '-'}")
        for issue in result.issues[:3]:
            print(f"      - {issue}")
    return 0 if all(r.status == "success" for r in results) else 1


# ======================================================================= cli


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Tailor a resume to a job description, verified against the compiled PDF.")
    parser.add_argument("job_description", nargs="?", type=Path,
                        help="path to a job description text file")
    parser.add_argument("--mock", action="store_true",
                        help="run the full pipeline with mocked provider responses (no API calls)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="use real providers but write to output/_smoke_tests and never "
                             "touch production tracking")
    parser.add_argument("--batch", nargs="?", const=str(engine.JOBS_DIR), default=None,
                        help="process every .txt job description in a folder (default: jobs/)")
    parser.add_argument("--revalidate", type=Path, default=None,
                        help="re-verify an existing run folder with zero API calls")
    args = parser.parse_args(argv)

    if args.revalidate:
        result = revalidate(args.revalidate)
        print(f"\n{result.status.upper()}: {args.revalidate}")
        for issue in result.issues:
            print(f"  - {issue}")
        return 0 if result.status == "success" else 1

    if args.batch:
        return run_batch(Path(args.batch), mock=args.mock, smoke=args.smoke_test)

    if not args.job_description:
        # No positional JD: process every unprocessed real posting in jobs/.
        return run_batch(engine.JOBS_DIR, mock=args.mock, smoke=args.smoke_test)
    if not args.job_description.exists():
        parser.error(f"no such job description: {args.job_description}")

    result = run_one(args.job_description, mock=args.mock, smoke=args.smoke_test)
    print(f"\n{result.status.upper()}: {result.run_dir}")
    for issue in result.issues:
        print(f"  - {issue}")
    return 0 if result.status == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
