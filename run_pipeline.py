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
import concurrent.futures
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
REQUIRED_ARTIFACTS = ("run.log", "strategy.json", "resume.tex",
                      "resume.txt", "cover_letter.txt", "assessment.txt")
# Removed after successful verification; kept for diagnosis on a failure.
LATEX_BUILD_ARTIFACTS = ("resume.aux", "resume.log", "resume.out")
# No longer produced; removed if an earlier run of a reused folder left one.
STALE_ARTIFACTS = ("assessment.json",)
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

# Where a stale index reference is parked for a human to reconcile. Append
# only: the tracking record itself is never rewritten by this path.
RECONCILE_PATH = engine.PROJECT_ROOT / ".tracking_reconcile.json"

# Outcome states. Only "recorded" means the tracking contract was satisfied;
# "stale_reference" and "aborted" are failures that must reach the status line
# rather than being swallowed behind a clean SUCCESS.
TRACKING_OK = "recorded"
TRACKING_SKIPPED = "skipped"
TRACKING_IDEMPOTENT = "idempotent"
TRACKING_STALE = "stale_reference"
TRACKING_ABORTED = "aborted"
TRACKING_FAILURES = (TRACKING_STALE, TRACKING_ABORTED)


@dataclass
class TrackingOutcome:
    """What tracking actually did, so the caller can tell the difference.

    Truthy exactly when the application was recorded, so existing callers that
    treat this as a boolean keep working.
    """

    state: str
    detail: str = ""

    def __bool__(self) -> bool:
        return self.state == TRACKING_OK

    @property
    def failed(self) -> bool:
        return self.state in TRACKING_FAILURES


def park_stale_reference(fingerprint: str, stale_folder: str, jd, run_dir: Path,
                         log: StageLog, path: Path | None = None) -> Path:
    """Record enough to identify and repair a stale index entry.

    Written beside the tracking files, APPENDING to whatever is already there.
    Nothing in processed_jobs.csv or .processed_index.json is touched: the
    previous record and its history survive exactly as they were, and a human
    (or a later reconciliation task) decides what the truth is.
    """
    target = path or RECONCILE_PATH
    entries = []
    if target.exists():
        try:
            loaded = json.loads(target.read_text(encoding="utf-8"))
            entries = loaded if isinstance(loaded, list) else [loaded]
        except (OSError, json.JSONDecodeError):
            # A damaged reconciliation file must not cost us this diagnosis.
            log.warning("could not read %s; starting a fresh reconciliation list",
                        target.name)
            entries = []
    entry = {
        "fingerprint": fingerprint,
        "indexed_run_folder": stale_folder,
        "indexed_folder_exists": False,
        "current_run_folder": run_dir.name,
        "company_name": jd.company_name,
        "job_title": jd.job_title,
        "job_id": jd.job_id,
        "source_file": jd.source_file,
        "detected_at": datetime.now().isoformat(timespec="seconds"),
        "state": TRACKING_STALE,
        "resolution": "unresolved",
        "note": ("the index points at a run folder that no longer exists; tracking was "
                 "NOT written and no application record was invented"),
    }
    if not any(e.get("fingerprint") == fingerprint
               and e.get("current_run_folder") == run_dir.name for e in entries):
        entries.append(entry)
    write_json_atomic(target, entries)
    return target



class Tracker:
    """Appends a production application to processed_jobs.csv, once."""

    def __init__(self, csv_path: Path, index_path: Path, enabled: bool = True,
                 reconcile_path: Path | None = None):
        # Beside the index by default, so a test never writes the real one.
        self.reconcile_path = reconcile_path or (
            index_path.with_name(".tracking_reconcile.json")
            if index_path != engine.PROCESSED_INDEX else RECONCILE_PATH)
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

    def record(self, jd: engine.JobPosting, run_dir: Path, status: str,
               log: StageLog) -> TrackingOutcome:
        if not self.enabled:
            log.info("tracking skipped: this is a mock/smoke run, production tracking "
                     "files are never opened")
            return TrackingOutcome(TRACKING_SKIPPED, "mock or smoke run")
        if status != "success":
            log.info("tracking skipped: status=%s (only a verified success is recorded)", status)
            return TrackingOutcome(TRACKING_SKIPPED, f"status={status}")
        missing = self.missing_artifacts(run_dir)
        if missing:
            log.warning("tracking skipped: missing required artifact(s) %s", ", ".join(missing))
            return TrackingOutcome(TRACKING_SKIPPED,
                                   "missing artifacts: " + ", ".join(missing))

        try:
            index = load_index(self.index_path)
        except TrackingStateError as error:
            log.error("tracking aborted: %s", error)
            return TrackingOutcome(TRACKING_ABORTED, str(error))

        # Idempotence: an explicit rerun of an already-recorded fingerprint must
        # not append a second CSV row or repoint the index.
        existing = index.get(jd.fingerprint)
        if isinstance(existing, dict):
            folder = (existing.get("run_folder") or "").strip()
            if folder and (engine.OUTPUT_DIR / folder).exists():
                log.info("tracking idempotent: fingerprint already recorded against run "
                         "folder %s; no CSV row appended and the index is unchanged",
                         folder)
                return TrackingOutcome(TRACKING_IDEMPOTENT,
                                       f"already recorded against {folder}")
            # The existing record is preserved untouched - overwriting it would
            # destroy the only evidence of the earlier application. The stale
            # reference is parked with enough metadata to repair it.
            parked = park_stale_reference(jd.fingerprint, folder, jd, run_dir, log,
                                          path=self.reconcile_path)
            log.error("tracking STOPPED: fingerprint %s is recorded against run folder "
                      "%r, which no longer exists. The existing record is UNCHANGED and "
                      "no application row was invented; the stale reference is parked in "
                      "%s for reconciliation.", jd.fingerprint[:12], folder, parked.name)
            return TrackingOutcome(
                TRACKING_STALE,
                f"fingerprint {jd.fingerprint[:12]} points at missing run folder "
                f"{folder!r}; parked in {parked.name}")

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
            return TrackingOutcome(TRACKING_OK, f"index committed; CSV append failed: {error}")

        log.info("recorded in %s and %s (index committed first)",
                 self.index_path.name, self.csv_path.name)
        return TrackingOutcome(TRACKING_OK)


# =================================================================== result


@dataclass
class RunResult:
    jd_path: Path
    status: str = "failed"
    run_dir: Path | None = None
    issues: list[str] = field(default_factory=list)
    strategy: dict = field(default_factory=dict)
    assessment: dict = field(default_factory=dict)
    pdf_path: Path | None = None
    # What tracking did: recorded / skipped / idempotent / stale_reference /
    # aborted. Only "recorded" means the application is logged as submitted.
    tracking: str = ""


def sanitize(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", (value or "").strip()).strip("_")
    return cleaned[:60] or "Unknown"


def final_pdf_name(jd: engine.JobPosting) -> str:
    """<Company>_<Job_Title>.pdf. No date: the run directory already carries one."""
    parts = [sanitize(jd.company_name or "Unknown_Company"),
             sanitize(jd.job_title or "Unknown_Role")]
    return "_".join(parts) + ".pdf"


def finalize_artifacts(run_dir: Path, jd: engine.JobPosting, log: StageLog) -> Path | None:
    """Rename the verified PDF and remove LaTeX build junk.

    Only ever called after verification succeeds, so a failed run keeps
    resume.pdf plus its .aux/.log/.out for diagnosis.
    """
    source = run_dir / f"{RESUME_STEM}.pdf"
    target = run_dir / final_pdf_name(jd)
    renamed: Path | None = None
    if source.exists():
        if target.exists() and target != source:
            target.unlink()
        source.replace(target)
        renamed = target
        log.info("final resume PDF: %s", target.name)
    elif target.exists():
        renamed = target
    for name in LATEX_BUILD_ARTIFACTS + STALE_ARTIFACTS:
        junk = run_dir / name
        if junk.exists():
            junk.unlink()
    log.info("removed LaTeX build artifacts: %s", ", ".join(LATEX_BUILD_ARTIFACTS))
    return renamed


def write_json_atomic(path: Path, payload: dict) -> None:
    """Write JSON so the file is either the old content or the new, never half.

    Same-directory temp file, flushed and fsynced, then os.replace(), which is
    atomic within a filesystem. The temp name carries the pid so two processes
    cannot collide on it. A failure part-way through leaves the previous valid
    file in place and removes the temp.
    """
    temp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        with temp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise


# ================================================== post-run Groq audit layer
#
# Three Groq calls run AFTER the resume and cover letter are final. They are
# advisory: their results reach assessment.txt, strategy.json and the terminal
# and nothing else. No audit value is ever read back into generation, and each
# call is isolated so one failure cannot blank the other two.


def project_catalogue(master: engine.MasterFacts, selection, chosen: list,
                      bullets: dict[str, list[str]]) -> list[dict]:
    """EVERY candidate project, not only the three that shipped.

    Project selection cannot be audited from the selected three alone, so the
    whole catalogue goes to the audit with its ranking and rationale.
    """
    selected_ids = list(getattr(selection, "selected", []) or [])
    ranks = getattr(selection, "ranks", {}) or {}
    reasons = getattr(selection, "reasons", {}) or {}
    catalogue: list[dict] = []
    for project in master.projects:
        pid = project.project_id
        catalogue.append({
            "project_id": pid,
            "name": project.name,
            "tech": list(project.tech),
            "tags": list(getattr(project, "tags", ()) or ()),
            "evidence": " | ".join(project.evidence),
            "selected": pid in selected_ids,
            "llm_rank": ranks.get(pid),
            "reason": reasons.get(pid, ""),
            "final_bullets": list(bullets.get(pid, [])),
        })
    return catalogue


def run_audits(client, jd: engine.JobPosting, signals, decision, *, policy, master,
               selection, chosen, project_bullets: dict[str, list[str]],
               display_order: list[str], allocation: dict[str, int],
               skills: list[str], experience_plain: list[str],
               requirements: list, tailoring: dict, extra_experience: list[str],
               log: StageLog, experience_context: dict | None = None) -> dict:
    """The two Groq audits, run CONCURRENTLY. Never raises.

    SINGLE WRITER by construction. Each worker is a pure function of its
    inputs: it performs the network call, normalizes the reply and RETURNS a
    result object. No worker touches strategy.json, assessment.txt, tracking
    or any other artifact - this coordinator collects both futures, merges
    them in a fixed order and hands one dict back for the caller to persist
    exactly once.

    The two calls are independent and hit different Groq models with separate
    pacing budgets, so one failing or stalling cannot cancel or delay the
    other.
    """
    catalogue = project_catalogue(master, selection, chosen, project_bullets)
    selection_detail = {
        "selected": list(getattr(selection, "selected", []) or []),
        "display_order": list(display_order or []),
        "allocation": dict(allocation or {}),
    }
    # Read-only inputs for the workers. Copies, so a worker cannot mutate
    # anything the coordinator or the caller still relies on.
    context_copy = dict(experience_context or {})
    catalogue_copy = [dict(entry) for entry in catalogue]
    detail_copy = dict(selection_detail)
    bullets_copy = {pid: list(values) for pid, values in (project_bullets or {}).items()}

    def application_worker() -> dict:
        """Worker 1: Qwen. Returns a result object; writes nothing."""
        return client.assess(
            jd, signals, chosen, list(skills), experience_plain=list(experience_plain),
            project_bullets=bullets_copy, requirements=requirements,
            tailoring=dict(tailoring or {}), extra_experience=list(extra_experience or []),
            project_catalogue=catalogue_copy, selection_detail=detail_copy,
            experience_context=context_copy)

    def research_worker() -> dict:
        """Worker 2: GPT-OSS with browser search. Returns a result object."""
        return client.research_company(
            jd, jd_posted=grounding.job_posted_date(jd.text),
            today=date.today().isoformat())

    out: dict = {"application_audit": {}, "assessment": {}, "research": {},
                 "errors": {}, "catalogue": catalogue, "selection_detail": selection_detail,
                 "experience_context": context_copy}

    audit_log = log.stage_log("APPLICATION AUDIT")
    research_log = log.stage_log("COMPANY RESEARCH")
    log.info("launching the assessment and company-research audits in parallel "
             "(max_workers=2); neither worker writes any artifact")
    # Both futures are SUBMITTED before either result is awaited, which is what
    # makes this parallel rather than two sequential calls.
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            "application_audit": pool.submit(application_worker),
            "company_research": pool.submit(research_worker),
        }
        results: dict[str, object] = {}
        errors: dict[str, str] = {}
        # Fixed collection order, so output is deterministic no matter which
        # future finishes first.
        for name in ("application_audit", "company_research"):
            try:
                results[name] = futures[name].result()
            except llm_client.ProviderError as error:
                errors[name] = f"{error.category}: {error}"
            except Exception as error:               # noqa: BLE001 - advisory
                errors[name] = f"{type(error).__name__}: {error}"

    # ---- merge, in the coordinator thread only ---------------------------
    if "application_audit" in results:
        assessment = results["application_audit"] or {}
        out["assessment"] = assessment
        out["application_audit"] = assessment.get("application_audit") or {}
        block = out["application_audit"]
        audit_log.info("experience_selection=%s resume_tailoring=%s project_selection=%s "
                       "project_bullets=%s callback_likelihood=%s",
                       block.get("experience_selection_score"),
                       block.get("resume_tailoring_score"),
                       block.get("project_selection_score"),
                       block.get("project_bullet_score"),
                       block.get("callback_likelihood"))
        for name, value in (block.get("project_selection_components") or {}).items():
            audit_log.info("project selection component %s=%s", name, value)
        for name, value in (block.get("project_bullet_components") or {}).items():
            audit_log.info("project bullet component %s=%s", name, value)
        for note in (block.get("experience_selection_notes") or []):
            audit_log.info("experience selection note: %s", note)
        for note in (block.get("project_selection_notes") or []):
            audit_log.info("project selection note: %s", note)
        for note in (block.get("project_bullet_notes") or []):
            audit_log.info("project bullet note: %s", note)
        audit_log.info("ADVISORY ONLY: the shipped Experience ids are unchanged (%s)",
                       ", ".join(decision.shipped_ids))
    else:
        out["errors"]["application_audit"] = errors["application_audit"]
        audit_log.warning("application audit unavailable: %s",
                          errors["application_audit"])

    if "company_research" in results:
        research = results["company_research"] or {}
        out["research"] = research
        research_log.info("visa=%s (%s) stem_opt=%s (%s) job_posted=%s (%s)",
                          research.get("company_visa_sponsorship"),
                          research.get("company_visa_confidence"),
                          research.get("company_stem_opt_support"),
                          research.get("company_stem_opt_confidence"),
                          research.get("job_posted"),
                          research.get("job_posted_confidence"))
        research_log.info("checked_at=%s sources=%d",
                          research.get("checked_at") or "unknown",
                          len(research.get("sources") or []))
        for source in research.get("sources") or []:
            research_log.info("source: [%s] %s %s", source.get("scope", "unscoped"),
                              source.get("title"), source.get("url"))
    else:
        out["errors"]["company_research"] = errors["company_research"]
        research_log.warning("company research unavailable: %s",
                             errors["company_research"])
    return out


def assessment_report(audits: dict, letter_score) -> dict:
    """The ten user-facing fields, from whichever audits succeeded."""
    return grounding.audit_report(
        application_audit=audits.get("application_audit"),
        research=audits.get("research"),
        letter_score=letter_score)


def print_assessment(report: dict, *, company: str, job_title: str) -> None:
    """The compact block printed after every completed job."""
    for line in grounding.render_assessment_terminal(
            report, company=company, job_title=job_title):
        print(line)


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


        # ---- semantic JD signals (Gemini recall, Python decides) --------
        sem_log = log.stage_log("JD SIGNALS")
        deterministic_map = grounding.deterministic_signal_map(signals)
        validated_signals, signal_problems = grounding.validate_semantic_signals(
            selection.semantic_signals_raw, jd.text)
        for problem in signal_problems:
            sem_log.warning("%s", problem.message)
        merged_map, signal_overrides = grounding.merge_semantic_signals(
            signals, validated_signals)
        for name, value in sorted(deterministic_map.items()):
            sem_log.info("deterministic %s=%s | gemini=%s | merged=%s", name,
                         value, validated_signals.get(name, "not reported"),
                         merged_map.get(name))
        if signal_overrides["added"]:
            sem_log.info("Gemini recovered %d signal(s) the deterministic classifier "
                         "missed: %s", len(signal_overrides["added"]),
                         ", ".join(signal_overrides["added"]))
        else:
            sem_log.info("no semantic signal was added; the deterministic view stands")
        merged_signals = grounding.apply_semantic_overrides(signals, signal_overrides)
        sem_log.info("Gemini reads the posting; it never selects an Experience bullet, "
                     "proposes one, or writes any Experience wording")

        # ---- experience (deterministic, protected) ----------------------
        exp_log = log.stage_log("EXPERIENCE")
        # MERGED signals feed the existing deterministic policy. Nothing else
        # in the run uses them, so project selection, bullets, skills, order
        # and verification are all byte-identical to the deterministic path.
        decision = engine.select_experience(template, policy, merged_signals)
        exp_log.info("rule applied: %s", decision.rule)
        if merged_signals is not signals:
            exp_log.info("the rule was resolved from MERGED signals (deterministic plus "
                         "%s recovered by validated Gemini evidence)",
                         ", ".join(signal_overrides["added"]))
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
        letter_priorities: list[dict] = []
        letter_sources: list[str] = []
        letter = ""
        try:
            # Deterministic authority for what the letter may claim: evaluate
            # every JD-named concept against the resume that is actually being
            # sent, and forbid candidate-owned phrasing for the unsupported ones.
            letter_evidence = engine.ResumeEvidence(
                experience_text=" ".join(engine.latex_to_plain(l)
                                         for _, _, l in decision.shipped),
                project_bullets=tuple((r.project.project_id, " ".join(r.bullets))
                                      for r in renders),
                skills=tuple(state.all_skills()))
            unsupported_concepts = engine.unsupported_jd_concepts(
                requirements, master, letter_evidence)
            if unsupported_concepts:
                letter_log.info("JD concepts the resume does not support (never claimable "
                                "as experience): %s", ", ".join(unsupported_concepts))
            # Evidence grouped by source, so a paragraph cannot silently move
            # facts between one role or project and another.
            letter_capsules = engine.source_capsules(master, chosen)
            # WHICH supported evidence the letter should lead with. Read-only
            # over the requirements and capsules built above: it changes no
            # resume decision and authorizes no new claim.
            letter_priorities = grounding.letter_priorities(
                jd_text=jd.text, job_title=jd.job_title or "", requirements=requirements,
                capsules=letter_capsules, master=master,
                on_resume=[p.project_id for p in chosen])
            for index, priority in enumerate(letter_priorities, start=1):
                if priority["supported"]:
                    letter_log.info("distinctive priority %d: %s (weight=%d) -> %s (%s)",
                                    index, priority["label"], priority["weight"],
                                    priority["source_name"],
                                    ", ".join(priority["evidence_terms"]))
                else:
                    letter_log.info("distinctive priority %d: %s (weight=%d) -> no supported "
                                    "evidence, leaving it uncovered", index,
                                    priority["label"], priority["weight"])
            if not letter_priorities:
                letter_log.info("no distinctive priorities: this posting reads as a generic "
                                "software-engineering role")
            letter, letter_problems = client.cover_letter(
                jd, signals, [engine.latex_to_plain(l) for _, _, l in decision.shipped],
                chosen, themes=themes, unsupported=unsupported_concepts,
                capsules=letter_capsules, priorities=letter_priorities)
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
            letter_sources = sources
            letter_log.info("evidence sources=%s", ", ".join(sources) or "none detected")
            for priority in grounding.supported_priorities(letter_priorities):
                letter_log.info("priority %r addressed=%s own_evidence=%s",
                                priority["label"],
                                grounding.priority_addressed(letter, priority),
                                grounding.priority_uses_own_source(letter, priority))
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
        tailoring_signals = {
            "skills_rendered_lines": verification.skills_rendered_lines,
            "experience_exact": verification.experience_exact,
            "pages": verification.pages,
            "projects": ", ".join(selection.selected),
            "issues": verification.issues,
        }
        # THE POST-RUN AUDIT LAYER. The resume and cover letter are final and
        # verified before this line; every value below is advisory. run_audits
        # never raises and isolates each call, so a failed audit leaves the
        # other two intact.
        audits = run_audits(
                client, jd, signals, decision, policy=policy, master=master,
                selection=selection, chosen=chosen,
                project_bullets={r.project.project_id: list(r.bullets) for r in renders},
                display_order=display, allocation=allocation,
                skills=state.all_skills(),
                experience_plain=[engine.latex_to_plain(l) for _, _, l in decision.shipped],
                requirements=requirements, tailoring=tailoring_signals,
                extra_experience=(
                    [engine.latex_to_plain(entry.latex)
                     for entry in policy.experience_library if entry.is_role_header]
                    + [engine.latex_to_plain(" ".join(
                        template.blocks["Education"].split()))]),
                log=log,
                experience_context={
                    "merged_signals": merged_map,
                    "rule": decision.rule,
                    "swaps": [(source, target) for source, target, _ in decision.swaps],
                    "shipped_ids": list(decision.shipped_ids),
                })
        assessment = audits["assessment"]
        assessment_error = audits["errors"].get("application_audit", "")
        try:
            if assessment_error:
                raise llm_client.ProviderError("audit_unavailable", assessment_error)
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
            # The per-requirement breakdown stays in strategy.json; the terminal
            # keeps only the compact advisory summary printed at the end.
            assess_log.info("classified %d strong, %d partial, %d unsupported, %d manual "
                            "review (detail in strategy.json)",
                            len(assessment.get("strong_matches") or []),
                            len(assessment.get("partial_matches") or []),
                            len(assessment.get("gaps") or []),
                            len(assessment.get("manual_review") or []))
            for flag in assessment.get("risk_flags") or []:
                assess_log.warning("risk flag: %s", flag)
            assessment_available = True
        except llm_client.ProviderError as error:
            assess_log.warning("application audit unavailable: category=%s %s",
                               error.category, error)
            assessment = {"error": f"{error.category}: {error}"}
            assessment_available = False
            assessment_error = f"{error.category}: {error}"
        # The rich structure stays in memory for validation and strategy.json;
        # the ARTIFACT is the small advisory report.
        letter_theme_hits = grounding.letter_themes_covered(
            letter, [r.headline for r in themes]) if letter else []
        letter_named_hits = grounding.letter_named_terms(
            letter, sorted({t for r in themes for t in r.terms})) if letter else []
        # How many distinct evidence sources the letter actually drew on. The
        # company-name probe above misses a role paragraph that never names the
        # employer, and collapses two different roles into one "experience",
        # so capsule labels are counted alongside it.
        letter_breadth = set(letter_sources) | {
            priority["source"] for priority in
            grounding.supported_priorities(letter_priorities)
            if letter and grounding.priority_uses_own_source(letter, priority)}
        # The Cover Letter Score stays deterministic Python on the existing
        # role-distinctive rubric; the other nine fields come from the audits.
        letter_score = grounding.cover_letter_score(
            letter=letter, problems=letter_problems,
            words=grounding.word_count(letter) if letter else 0,
            priorities=letter_priorities, themes_covered=len(letter_theme_hits),
            evidence_sources=len(letter_breadth),
            named_terms_covered=len(letter_named_hits),
            company=jd.company_name or "", job_title=jd.job_title or "")
        compact = assessment_report(audits, letter_score)
        (run_dir / "assessment.txt").write_text(
            grounding.render_assessment_txt(compact), encoding="utf-8")
        result.assessment = compact
        for name, message in audits["errors"].items():
            assess_log.warning("%s unavailable (%s); its fields report UNKNOWN",
                               name.replace("_", " "), message)

        # ---- status ------------------------------------------------------
        issues = list(verification.issues)
        # Assessment is ADVISORY: a verified resume and a valid cover letter are
        # a success even when scoring is unavailable or malformed.
        for problem in final_problems:
            assess_log.warning("assessment integrity (advisory): %s", problem.message)
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
        if not assessment_available:
            final_warning = (f"assessment unavailable ({assessment_error}); the advisory "
                             f"fields fall back to UNKNOWN and the run is judged on the "
                             f"resume and cover letter alone")
            assess_log.warning("%s", final_warning)
        status = "success" if not issues else "needs_review"

        strategy = {
            "company_name": jd.company_name,
            "job_title": jd.job_title,
            "job_id": jd.job_id,
            "role_family": signals.role_family,
            "section_order": order,
            # Auditability: the two signal sources, the validated subset and
            # what the Experience policy actually acted on, kept separate.
            "jd_signals": {
                "deterministic_signals": deterministic_map,
                "gemini_semantic_signals_raw": selection.semantic_signals_raw,
                "gemini_semantic_signals_validated": validated_signals,
                "merged_signals": merged_map,
                "recovered_by_gemini": signal_overrides["added"],
                "role_family_promoted_to": signal_overrides["role_family"],
                "rejected": [p.message for p in signal_problems],
            },
            "experience": {
                "kept": [bid for bid, action, _ in decision.shipped if action == "KEEP_EXACT"],
                "swaps": [{"source_id": bid, "target_id": rid, "rule": rule}
                          for bid, rid, rule in decision.swaps],
                "shipped_ids": decision.shipped_ids,
                "rule": decision.rule,
                "wording_source": engine.POLICY_PATH.name,
                "rule_fired": decision.rule,
                "resolved_from": ("merged deterministic + validated Gemini signals"
                                  if signal_overrides["added"]
                                  else "deterministic signals only"),
                # Recorded so --assess can rebuild the shipped Experience
                # exactly, without re-running selection.
                "shipped": [{"bullet_id": bid, "action": action,
                             "text": engine.latex_to_plain(latex)}
                            for bid, action, latex in decision.shipped],
            },
            "projects": {
                "selected_by_relevance": selection.selected,
                "display_order": display,
                "allocation": {pid: allocation[pid] for pid in selection.selected},
                "selection_reasons": {pid: selection.reasons[pid] for pid in selection.selected},
                "selected": selection.selected,
                "selected_is": ("relevance order, identical to selected_by_relevance; the PDF "
                                "renders display_order instead"),
                # The final bullets and the full candidate catalogue, so
                # --assess audits the same option space production did.
                "bullets": {r.project.project_id: list(r.bullets) for r in renders},
                "ranks": {pid: selection.ranks.get(pid) for pid in selection.selected},
                "catalogue": audits["catalogue"],
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
                "distinctive_priorities": letter_priorities,
                "priority_coverage": [
                    {"label": p["label"], "source": p["source_name"],
                     "addressed": grounding.priority_addressed(letter, p),
                     "used_own_evidence": grounding.priority_uses_own_source(letter, p)}
                    for p in grounding.supported_priorities(letter_priorities)] if letter
                    else [],
                "relevance": [p.message for p in letter_problems
                              if p.kind == "relevance"],
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
                # The per-requirement breakdown lives here now: assessment.txt is
                # deliberately small, and strategy.json is the debug artifact.
                "detail": {bucket: assessment.get(bucket) or []
                           for bucket in ("strong_matches", "partial_matches", "gaps",
                                          "manual_review")},
                "summary": assessment.get("summary"),
                "model_summary": assessment.get("model_summary"),
                "verdict_source": assessment.get("verdict_source"),
                "eligibility_detail": (assessment.get("eligibility") or {}).get("details"),
                "tailoring_notes": (assessment.get("tailoring_quality") or {}).get("notes")
                                   or [],
                "risk_flags": assessment.get("risk_flags") or [],
            },
            # The audit layer's own record. Component scores, explanations,
            # missed signals, confidences and source URLs live here; the ten
            # user-facing fields live in assessment.txt.
            "audit": {
                "report": compact,
                "experience_selection": (audits["application_audit"] or {}).get(
                    "experience_selection_score"),
                "application": audits["application_audit"],
                "company_research": audits["research"],
                "errors": audits["errors"],
                "cover_letter_score": letter_score,
                "provider_calls": [
                    c.as_dict() for c in client.calls
                    if c.purpose in ("assessment", "company_research")],
                "audited_at": datetime.now().isoformat(timespec="seconds"),
            },
            "jd_fingerprint": jd.fingerprint,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
        # ONE persistence operation, after both audit futures resolved, written
        # atomically so a crash mid-write cannot leave truncated JSON behind.
        write_json_atomic(run_dir / "strategy.json", strategy)
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
        if status == "success":
            final_pdf = finalize_artifacts(run_dir, jd, final)
            if final_pdf:
                result.pdf_path = final_pdf
        tracker = Tracker(engine.PROCESSED_CSV, engine.PROCESSED_INDEX, enabled=not isolated)
        tracking = tracker.record(jd, run_dir, status, log.stage_log("TRACKING"))
        result.tracking = tracking.state
        if tracking.failed and status == "success":
            # The artifacts are verified and kept, but the application was NOT
            # recorded, so this must not read as an ordinary clean success: the
            # posting would look unprocessed to the next batch run.
            status = "success_with_tracking_warning"
            result.status = status
            issues.append(f"tracking not persisted ({tracking.state}): {tracking.detail}")
            final.error("artifacts are complete and verified, but TRACKING FAILED: %s. "
                        "The application is NOT recorded as submitted.", tracking.detail)
        provider_summary = ", ".join(
            f"{c.purpose}:{'accepted' if c.accepted else 'rejected'}"
            for c in client.calls) or "none"
        final.info("provider calls: %s", provider_summary)
        final.info("artifacts: %s", ", ".join(sorted(p.name for p in run_dir.iterdir())))
        summary = log.stage_log("ASSESSMENT")
        for field in grounding.ASSESSMENT_FIELDS:
            summary.info("%s: %s", field, compact.get(field, "UNKNOWN"))
        # The compact terminal block, printed after every completed job. In
        # batch mode this lands before the next [BATCH] processing line.
        print_assessment(compact, company=jd.company_name or "",
                         job_title=jd.job_title or "")
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
    # Best measured rendering per bullet, so a failed repair can be undone.
    best_bullets: dict[tuple[str, int], tuple[tuple[int, float], str]] = {}
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
        _track_best(best_bullets, bullets, renders, policy)
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
            _restore_best(best_bullets, renders, layout_log)
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
            _restore_best(best_bullets, renders, layout_log)
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
    under = [(m, FILL_MAX_TIER) for m in sorted(
        (m for m in categories
         if m.lines >= 2 and m.last_line_fill < policy.second_line_max_fill),
        key=lambda m: m.last_line_fill)]
    # A DIRECT JD match (tier 1-2: the posting names it and the master supports
    # it) may also join a single-line category that still has room on that
    # line. This is what lets "React" surface when a posting asks for it while
    # the category holds only one line. An addition that costs a ninth line is
    # reverted on the next measurement, so the 8-line contract is unchanged.
    under += [(m, 2) for m in sorted(
        (m for m in categories if m.lines == 1 and m.last_line_fill < 90.0),
        key=lambda m: m.last_line_fill)]
    for metric, max_tier in under:
        for candidate in state.candidates:
            if candidate.category != metric.label:
                continue
            if candidate.tier > max_tier:
                continue
            if state.is_rejected(candidate.name):
                continue
            if engine.fold_term(candidate.name) in state.keys():
                continue
            if state.add(candidate):
                log.info("iteration action=ADD skill=%r category=%r tier=%d reason=%s",
                         candidate.name, candidate.category, candidate.tier,
                         f"{candidate.reason}; {metric.label} last line was only "
                         f"{metric.last_line_fill:.0f}% full")
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


def _bullet_rank(lines: int, fill: float, target: int, orphan: float) -> tuple[int, float]:
    """Order candidate renderings of one bullet, best last.

    A two-line bullet with a thin tail is genuinely better than a one-line
    bullet, even though neither satisfies the contract, so a repair can never
    justify collapsing two lines into one.
    """
    if lines == target and fill >= orphan:
        return (3, fill)
    if lines == target:
        return (2, fill)
    if lines > target:
        return (1, -float(lines))
    return (0, fill)


def _track_best(best: dict, bullets, renders, policy: engine.Policy) -> None:
    """Remember the best measured text seen for every project bullet."""
    flat: list[tuple[str, int]] = []
    for render in renders:
        for index in range(len(render.bullets)):
            flat.append((render.project.project_id, index))
    for position, measured in enumerate(bullets):
        if position >= len(flat):
            break
        key = flat[position]
        render = next(r for r in renders if r.project.project_id == key[0])
        text = render.bullets[key[1]]
        rank = _bullet_rank(measured.lines, measured.fill_pct,
                            policy.bullet_target_lines, policy.tail_orphan_max)
        if key not in best or rank > best[key][0]:
            best[key] = (rank, text)


def _restore_best(best: dict, renders, log: StageLog) -> None:
    """Put every bullet back to the best rendering measured for it."""
    for render in renders:
        for index, text in enumerate(render.bullets):
            entry = best.get((render.project.project_id, index))
            if entry and entry[1] != text:
                render.bullets[index] = entry[1]
                log.info("restored the best measured text for %s#%d rather than keeping a "
                         "degraded repair", render.project.project_id, index + 1)


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
            # Two lines with a thin tail needs MORE text on that second line,
            # not less: shortening collapses it to a single line, which is the
            # worse contract violation.
            actions.append({"project_id": project_id, "bullet_index": index,
                            "goal": "lengthen",
                            "reason": f"last line is only {bullet.fill_pct:.0f}% full, below the "
                                      f"{policy.tail_orphan_max:.0f}% orphan floor; it needs "
                                      f"more of this project's evidence, not less"})

    if pages > policy.page_count and not actions:
        longest = max(range(len(bullets)), key=lambda i: bullets[i].lines, default=None)
        if longest is not None and longest < len(flat):
            project_id, index = flat[longest]
            actions.append({"project_id": project_id, "bullet_index": index, "goal": "shorten",
                            "reason": f"resume runs to {pages} pages and must fit "
                                      f"{policy.page_count}"})
    return {"notes": notes, "actions": actions}


# ================================================================ revalidate


# Files --assess is forbidden to touch. Hashed before and after, so a bug
# cannot quietly rewrite a shipped artifact.
ASSESS_READONLY = ("resume.tex", "resume.txt", "cover_letter.txt", "job_description.txt")


def _digests(run_dir: Path) -> dict[str, str]:
    """SHA256 of every artifact --assess must leave alone, PDFs included."""
    import hashlib

    out: dict[str, str] = {}
    for name in ASSESS_READONLY:
        path = run_dir / name
        if path.exists():
            out[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for pdf_file in sorted(run_dir.glob("*.pdf")):
        out[pdf_file.name] = hashlib.sha256(pdf_file.read_bytes()).hexdigest()
    return out


def _decision_from_strategy(strategy: dict, policy: engine.Policy
                            ) -> engine.ExperienceDecision:
    """Rebuild the shipped Experience decision from the recorded ids.

    Reconstruction, never re-selection: the ids and the rule come from
    strategy.json, and the wording comes from the approved library by id. No
    swap rule is re-evaluated and no signal is re-read.
    """
    block = strategy.get("experience") or {}
    shipped_ids = list(block.get("shipped_ids") or [])
    by_id = {entry.bullet_id: entry for entry in policy.experience_library}
    by_id.update({entry.bullet_id: entry for entry in policy.alternate_library})
    recorded = {row.get("bullet_id"): row for row in (block.get("shipped") or [])}
    shipped: list[tuple[str, str, str]] = []
    for bullet_id in shipped_ids:
        entry = by_id.get(bullet_id)
        action = (recorded.get(bullet_id) or {}).get("action", "KEEP_EXACT")
        shipped.append((bullet_id, action, entry.latex if entry else
                        (recorded.get(bullet_id) or {}).get("text", "")))
    swaps = [(row.get("source_id", ""), row.get("target_id", ""), row.get("rule", ""))
             for row in (block.get("swaps") or [])]
    return engine.ExperienceDecision(
        latex_block="", shipped=shipped, swaps=swaps,
        rule=block.get("rule") or "unrecorded", allowed_plain=set(),
        expected_plain=[], shipped_ids=shipped_ids)


def assess_run(run_dir: Path, *, mock: bool = False, console: bool = True) -> RunResult:
    """Standalone audit of an existing run folder.

    Runs exactly the same three audit functions production uses, reading the
    artifacts that are already on disk. It regenerates nothing, compiles
    nothing and never opens production tracking: the only things it may write
    are assessment.txt and the audit block inside strategy.json.
    """
    credentials = engine.load_credentials()
    logger, buffers = build_logger(credentials.secrets, console=console)
    log = StageLog(logger, "ASSESS")
    result = RunResult(jd_path=run_dir, run_dir=run_dir)
    handler = buffers[0].attach_file(run_dir / "assessment.log", logger)
    before = _digests(run_dir)
    try:
        strategy_path = run_dir / "strategy.json"
        jd_file = run_dir / "job_description.txt"
        for required in (strategy_path, jd_file):
            if not required.exists():
                raise engine.PolicyError(f"{run_dir} has no {required.name}; "
                                         f"--assess needs the original run artifacts")
        strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
        master = engine.load_master()
        policy = engine.load_policy()
        jd = engine.read_jd(jd_file)
        signals = engine.classify_jd(jd.text, master.section_order)
        requirements = engine.extract_jd_requirements(jd.text, master)
        decision = _decision_from_strategy(strategy, policy)
        log.info("auditing %s", run_dir.name)
        log.info("company=%r title=%r", jd.company_name, jd.job_title)
        log.info("reconstructed %d shipped Experience bullet(s) from strategy.json: %s",
                 len(decision.shipped_ids), ", ".join(decision.shipped_ids))

        projects_block = strategy.get("projects") or {}
        selected_ids = list(projects_block.get("selected_by_relevance") or [])
        bullets = {pid: list(value) for pid, value in
                   (projects_block.get("bullets") or {}).items()}
        chosen = [master.project(pid) for pid in selected_ids if master.project(pid)]
        if not bullets:
            # An older run folder predates the recorded bullets. The bullet
            # audit degrades rather than inventing wording.
            log.warning("strategy.json records no project bullets; the project bullet "
                        "audit will score from evidence alone")
        selection = llm_client.Selection(
            selected=selected_ids,
            ranks={pid: rank for pid, rank in (projects_block.get("ranks") or {}).items()
                   if isinstance(rank, int)},
            reasons=dict(projects_block.get("selection_reasons") or {}),
            considered=[])
        skills = list((strategy.get("skills") or {}).get("rendered") or [])
        verification = strategy.get("verification") or {}
        tailoring_signals = {
            "skills_rendered_lines": verification.get("skills_rendered_lines"),
            "experience_exact": verification.get("experience_exact"),
            "pages": verification.get("pages"),
            "projects": ", ".join(selected_ids),
            "issues": verification.get("issues") or [],
        }
        # 0 Gemini calls: the audit transport is the only one built here, and
        # audits never fall back to it anyway.
        client = llm_client.build_client(master, policy, log.stage_log("PROVIDER"),
                                         mock=mock, credentials=credentials)
        audits = run_audits(
            client, jd, signals, decision, policy=policy, master=master,
            selection=selection, chosen=chosen, project_bullets=bullets,
            display_order=list(projects_block.get("display_order") or []),
            allocation=dict(projects_block.get("allocation") or {}),
            skills=skills,
            experience_plain=[engine.latex_to_plain(latex)
                              for _, _, latex in decision.shipped],
            requirements=requirements, tailoring=tailoring_signals,
            extra_experience=[entry.plain for entry in policy.experience_library
                              if entry.is_role_header],
            log=log,
            experience_context={
                # Recorded by the original run, so the audit judges the same
                # decision that shipped rather than re-deriving one.
                "merged_signals": (strategy.get("jd_signals") or {}).get(
                    "merged_signals") or grounding.deterministic_signal_map(signals),
                "rule": (strategy.get("experience") or {}).get("rule") or decision.rule,
                "swaps": [(row.get("source_id", ""), row.get("target_id", ""))
                          for row in ((strategy.get("experience") or {}).get("swaps") or [])],
                "shipped_ids": list(decision.shipped_ids),
            })

        # The Cover Letter Score is recomputed deterministically from the
        # letter already on disk. No provider call, no regeneration.
        letter_path = run_dir / "cover_letter.txt"
        letter = letter_path.read_text(encoding="utf-8") if letter_path.exists() else ""
        letter_score = (strategy.get("audit") or {}).get("cover_letter_score")
        if letter:
            capsules = engine.source_capsules(master, chosen)
            priorities = grounding.letter_priorities(
                jd_text=jd.text, job_title=jd.job_title or "", requirements=requirements,
                capsules=capsules, master=master, on_resume=selected_ids)
            themes = engine.jd_themes(requirements, 5)
            named = sorted({term for req in themes for term in req.terms})
            sources = {p["source"] for p in grounding.supported_priorities(priorities)
                       if grounding.priority_uses_own_source(letter, p)}
            letter_score = grounding.cover_letter_score(
                letter=letter, problems=[], words=grounding.word_count(letter),
                priorities=priorities,
                themes_covered=len(grounding.letter_themes_covered(
                    letter, [r.headline for r in themes])),
                evidence_sources=len(sources),
                named_terms_covered=len(grounding.letter_named_terms(letter, named)),
                company=jd.company_name or "", job_title=jd.job_title or "")
            log.info("cover letter score recomputed deterministically from the artifact "
                     "on disk: %s", letter_score)

        report = assessment_report(audits, letter_score)
        (run_dir / "assessment.txt").write_text(
            grounding.render_assessment_txt(report), encoding="utf-8")
        # ONLY the audit/assessment metadata is updated. Every other key in
        # strategy.json is written back exactly as it was read.
        strategy["audit"] = {
            "report": report,
            "experience_selection": (audits["application_audit"] or {}).get(
                "experience_selection_score"),
            "application": audits["application_audit"],
            "company_research": audits["research"],
            "errors": audits["errors"],
            "cover_letter_score": letter_score,
            "provider_calls": [c.as_dict() for c in client.calls],
            "audited_at": datetime.now().isoformat(timespec="seconds"),
            "mode": "standalone --assess",
        }
        # strategy.json was loaded ONCE above and is persisted ONCE here, in
        # the coordinator thread, atomically. The audit workers wrote nothing.
        write_json_atomic(strategy_path, strategy)
        result.assessment = report
        result.strategy = strategy

        after = _digests(run_dir)
        changed = [name for name, digest in before.items() if after.get(name) != digest]
        if changed:
            # Belt and braces: report it loudly rather than let it pass.
            raise engine.PolicyError(
                "--assess modified artifacts it must never touch: " + ", ".join(changed))
        log.info("generation artifacts unchanged (sha256 verified): %s",
                 ", ".join(sorted(before)))
        gemini = [c for c in client.calls if c.provider == "gemini"]
        log.info("provider calls: %s", ", ".join(
            f"{c.purpose}:{c.provider}" for c in client.calls) or "none")
        log.info("gemini calls: %d (must be 0)", len(gemini))
        result.issues = [f"{name}: {message}" for name, message in audits["errors"].items()]
        result.status = "success" if not audits["errors"] else "needs_review"
        print_assessment(report, company=jd.company_name or "",
                         job_title=jd.job_title or "")
        return result
    except Exception as error:                       # noqa: BLE001 - reported, not hidden
        log.error("standalone assessment failed: %s\n%s", error, traceback.format_exc())
        result.status = "failed"
        result.issues.append(f"{type(error).__name__}: {error}")
        return result
    finally:
        handler.close()
        logger.removeHandler(handler)


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
            # A successful run renames the PDF to <Company>_<Job_Title>.pdf.
            renamed = sorted(p for p in run_dir.glob("*.pdf")
                             if p.name != f"{RESUME_STEM}.pdf")
            if len(renamed) == 1:
                pdf_path = renamed[0]
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


@dataclass
class ReconcileReport:
    """The outcome of one reconciliation, dry-run or applied."""

    state: str                       # refused | would_repoint | repointed | already
    checks: list[tuple[str, bool, str]] = field(default_factory=list)
    detail: str = ""
    fingerprint: str = ""
    stale_folder: str = ""
    run_folder: str = ""
    applied: bool = False

    @property
    def ok(self) -> bool:
        return self.state in ("would_repoint", "repointed", "already")


def reconcile_tracking(run_dir: Path, *, apply: bool = False,
                       csv_path: Path | None = None, index_path: Path | None = None,
                       reconcile_path: Path | None = None,
                       output_dir: Path | None = None) -> ReconcileReport:
    """Repoint ONE stale tracking fingerprint at a verified current run.

    Dry-run by default: without `apply` nothing on disk is touched at all.
    Every condition below must hold, and any ambiguity refuses rather than
    guessing - the whole point of the stale-reference state is that the
    pipeline would not guess either.
    """
    csv_path = csv_path or engine.PROCESSED_CSV
    index_path = index_path or engine.PROCESSED_INDEX
    reconcile_path = reconcile_path or (
        index_path.with_name(".tracking_reconcile.json")
        if index_path != engine.PROCESSED_INDEX else RECONCILE_PATH)
    output_dir = output_dir or engine.OUTPUT_DIR
    report = ReconcileReport(state="refused", run_folder=run_dir.name)
    checks = report.checks

    def check(name: str, condition: bool, detail: str = "") -> bool:
        checks.append((name, bool(condition), detail))
        return bool(condition)

    def refuse(detail: str) -> ReconcileReport:
        report.state = "refused"
        report.detail = detail
        return report

    # 1. the requested run folder exists
    if not check("run folder exists", run_dir.is_dir(), str(run_dir)):
        return refuse(f"{run_dir} is not a directory")

    # 2. the finalized artifacts are all present
    missing = [name for name in REQUIRED_ARTIFACTS if not (run_dir / name).exists()]
    pdfs = sorted(run_dir.glob("*.pdf"))
    if not check("required artifacts present", not missing,
                 "missing: " + ", ".join(missing) if missing else "all present"):
        return refuse(f"{run_dir.name} is missing required artifact(s): "
                      f"{', '.join(missing)}")
    # A shipped run has exactly one PDF and it is the RENAMED one:
    # finalize_artifacts only runs on success, so a lone resume.pdf means the
    # run never finalized and was never an application.
    finalized_pdfs = [p for p in pdfs if p.name != f"{RESUME_STEM}.pdf"]
    if not check("exactly one finalized PDF", len(pdfs) == 1 and len(finalized_pdfs) == 1,
                 ", ".join(p.name for p in pdfs) or "none"):
        return refuse(f"{run_dir.name} does not hold exactly one finalized PDF "
                      f"({', '.join(p.name for p in pdfs) or 'none'}); a shipped run's PDF "
                      f"is renamed to <Company>_<Job_Title>.pdf")

    # 3. the run is a verified success, or a success whose ONLY problem was
    #    the stale tracking reference itself
    try:
        strategy = json.loads((run_dir / "strategy.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        check("strategy.json readable", False, str(error))
        return refuse(f"{run_dir.name}/strategy.json could not be read: {error}")
    status = str(strategy.get("status") or "")
    issues = [str(i) for i in (strategy.get("issues") or [])]
    non_tracking = [i for i in issues if "tracking not persisted" not in i]
    verified = status in ("success", "success_with_tracking_warning") and not non_tracking
    if not check("run verified successful", verified,
                 f"status={status or 'unrecorded'}, other issues={non_tracking or 'none'}"):
        return refuse(f"{run_dir.name} is not a clean success (status={status!r}; "
                      f"unrelated issues: {non_tracking})")

    # 4. the fingerprint reconstructs from the run's own stored JD
    jd_file = run_dir / "job_description.txt"
    try:
        jd = engine.read_jd(jd_file)
    except Exception as error:                   # noqa: BLE001 - reported below
        check("fingerprint reconstructs", False, str(error))
        return refuse(f"could not read {jd_file}: {error}")
    fingerprint = jd.fingerprint
    report.fingerprint = fingerprint
    recorded = str(strategy.get("jd_fingerprint") or "")
    if not check("fingerprint matches strategy.json", not recorded or recorded == fingerprint,
                 f"stored={recorded[:12]} recomputed={fingerprint[:12]}"):
        return refuse("the stored JD no longer hashes to the fingerprint this run "
                      "recorded; refusing to reconcile an altered posting")

    # 9. an unresolved reconciliation entry must exist for this pair
    entries = _load_reconcile_entries(reconcile_path)
    matching = [e for e in entries
                if e.get("fingerprint") == fingerprint
                and e.get("current_run_folder") == run_dir.name]
    unresolved = [e for e in matching if e.get("resolution") == "unresolved"]
    resolved = [e for e in matching if e.get("resolution") == "repointed"]
    if not check("reconciliation entry exists", bool(matching),
                 f"{len(entries)} entr(y/ies) parked"):
        return refuse(f"no reconciliation entry links fingerprint {fingerprint[:12]} to "
                      f"{run_dir.name}; only a parked stale reference can be reconciled")

    # 5/6. the index must still name this fingerprint, and that folder must
    #      still be missing - otherwise there is nothing stale to repair
    try:
        index = load_index(index_path)
    except TrackingStateError as error:
        check("tracking index readable", False, str(error))
        return refuse(f"tracking index is unusable: {error}")
    entry = index.get(fingerprint)
    if not check("fingerprint present in index", isinstance(entry, dict),
                 "absent" if entry is None else type(entry).__name__):
        return refuse(f"fingerprint {fingerprint[:12]} is not in the tracking index; "
                      f"this is not a stale-reference repair")
    indexed = (entry.get("run_folder") or "").strip()
    report.stale_folder = indexed
    if indexed == run_dir.name:
        # Already authoritative. Idempotent no-op, whatever the parked entry says.
        check("already reconciled", True, indexed)
        report.state = "already"
        report.detail = (f"the index already points at {run_dir.name}; nothing to do")
        if apply and unresolved:
            _resolve_reconcile_entries(reconcile_path, entries, unresolved, run_dir.name)
            report.applied = True
            report.detail += " (the parked entry was marked resolved)"
        return report
    if not check("indexed folder is missing", not (output_dir / indexed).exists(),
                 indexed or "empty"):
        return refuse(f"the indexed run folder {indexed!r} still exists, so tracking is "
                      f"not stale; refusing to repoint a live record")

    # 7. no OTHER live folder is already AUTHORITATIVE for this fingerprint.
    #    Authoritative means a finalized success: artifacts complete, the PDF
    #    renamed by finalize_artifacts, and a clean status. A needs_review run
    #    for the same posting is not a rival - it is a failed attempt, and the
    #    real output/ directory holds exactly such a folder.
    rivals = []
    for candidate in sorted(output_dir.glob("*")):
        if not candidate.is_dir() or candidate.name in (run_dir.name, indexed):
            continue
        stored = candidate / "job_description.txt"
        if not stored.exists():
            continue
        try:
            if engine.read_jd(stored).fingerprint != fingerprint:
                continue
        except Exception:                        # noqa: BLE001 - not a rival then
            continue
        if [n for n in REQUIRED_ARTIFACTS if not (candidate / n).exists()]:
            continue
        finalized = [p for p in candidate.glob("*.pdf")
                     if p.name != f"{RESUME_STEM}.pdf"]
        if len(finalized) != 1:
            continue                             # never finalized, so never applied
        try:
            rival_status = str(json.loads(
                (candidate / "strategy.json").read_text(encoding="utf-8")
            ).get("status") or "")
        except (OSError, json.JSONDecodeError):
            continue
        if rival_status in ("success", "success_with_tracking_warning"):
            rivals.append(candidate.name)
    if not check("no rival authoritative run", not rivals, ", ".join(rivals) or "none"):
        return refuse(f"another complete run already covers this posting ({', '.join(rivals)}); "
                      f"choose one explicitly rather than reconciling ambiguously")

    # 8. metadata compatibility, where the parked entry recorded any
    parked = (unresolved or resolved or matching)[0]
    # company/title are posting identity. source_file is NOT: the parked entry
    # records the original jobs/*.txt name while the run folder's copy is
    # always job_description.txt, so comparing them can never match. The
    # fingerprint already proves the posting text is byte-identical, which is
    # a far stronger statement than a filename.
    mismatches = [
        f"{field}: parked={parked.get(field)!r} run={value!r}"
        for field, value in (("company_name", jd.company_name),
                             ("job_title", jd.job_title))
        if parked.get(field) not in (None, "", value)]
    if not check("metadata compatible", not mismatches, "; ".join(mismatches) or "compatible"):
        return refuse("the parked entry describes a different posting than this run "
                      f"({'; '.join(mismatches)})")
    if not check("reconciliation entry unresolved", bool(unresolved),
                 "unresolved" if unresolved else "already resolved"):
        return refuse(f"the reconciliation entry for {run_dir.name} is already marked "
                      f"{parked.get('resolution')!r}")

    if not apply:
        report.state = "would_repoint"
        report.detail = (f"would repoint {fingerprint[:12]} from the missing {indexed!r} "
                         f"to {run_dir.name}; no CSV row would be appended")
        return report

    # ---- the single mutation -------------------------------------------
    # Only this fingerprint moves, and the record it replaces is kept inline
    # so the history of the earlier application is not lost.
    history = list(entry.get("superseded") or [])
    history.append({k: entry.get(k) for k in ("run_folder", "source_file", "recorded_at")
                    if entry.get(k)} | {"missing_at": datetime.now().isoformat(
                        timespec="seconds")})
    index[fingerprint] = {
        "run_folder": run_dir.name,
        # The original jobs/*.txt name, as the parked entry and the superseded
        # record both spell it - not the run folder's job_description.txt copy.
        "source_file": parked.get("source_file") or entry.get("source_file")
                       or jd.source_file,
        "recorded_at": entry.get("recorded_at") or datetime.now().isoformat(
            timespec="seconds"),
        "reconciled_at": datetime.now().isoformat(timespec="seconds"),
        "reconciled_from": indexed,
        "superseded": history,
    }
    _write_index_atomic(index_path, index)
    _resolve_reconcile_entries(reconcile_path, entries, unresolved, run_dir.name)
    report.state = "repointed"
    report.applied = True
    # The CSV is a human log of APPLICATIONS. This fingerprint already has one
    # from the original run, so appending another would invent a second
    # submission; the index stays the authority for skip decisions.
    report.detail = (f"repointed {fingerprint[:12]} to {run_dir.name}; the previous record "
                     f"for the missing {indexed!r} is preserved under 'superseded' and no "
                     f"{csv_path.name} row was appended")
    return report


def _load_reconcile_entries(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [e for e in (loaded if isinstance(loaded, list) else [loaded])
            if isinstance(e, dict)]


def _resolve_reconcile_entries(path: Path, entries: list[dict], targets: list[dict],
                               run_folder: str) -> None:
    """Mark entries resolved in place. The event itself is never deleted."""
    stamp = datetime.now().isoformat(timespec="seconds")
    for entry in targets:
        entry["resolution"] = "repointed"
        entry["resolved_at"] = stamp
        entry["resolved_run_folder"] = run_folder
    write_json_atomic(path, entries)


def _write_index_atomic(index_path: Path, index: dict) -> None:
    """Same temp + fsync + os.replace contract the Tracker itself uses."""
    write_json_atomic(index_path, index)


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
        # run_one prints the compact [ASSESSMENT] block itself, so it lands
        # here - after this job, before the next [BATCH] processing line.
        results.append(run_one(path, mock=mock, smoke=smoke, console=False))

    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    print("[BATCH] complete")
    warned = counts.get("success_with_tracking_warning", 0)
    print(f"  successes: {counts.get('success', 0)}")
    if warned:
        # Artifacts are complete, tracking is not: these postings will look
        # unprocessed to the next batch run until they are reconciled.
        print(f"  success_with_tracking_warning: {warned}")
    print(f"  needs_review: {counts.get('needs_review', 0)}")
    print(f"  failed: {sum(n for s, n in counts.items() if s not in ('success', 'needs_review', 'success_with_tracking_warning'))}")
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
    parser.add_argument("--assess", type=Path, default=None, metavar="RUN_FOLDER",
                        help="re-run the Groq audits against an existing run folder; "
                             "regenerates nothing and never touches tracking")
    parser.add_argument("--reconcile-tracking", type=Path, default=None,
                        metavar="RUN_FOLDER",
                        help="repoint one stale tracking fingerprint at this verified run. "
                             "DRY RUN unless --apply is also given")
    parser.add_argument("--apply", action="store_true",
                        help="perform the mutation that --reconcile-tracking describes")
    args = parser.parse_args(argv)

    if args.reconcile_tracking:
        report = reconcile_tracking(args.reconcile_tracking, apply=args.apply)
        print(f"\nRECONCILE {report.state.upper()}: {args.reconcile_tracking}")
        for name, passed, detail in report.checks:
            print(f"  [{'ok ' if passed else 'NO '}] {name}"
                  + (f" ({detail})" if detail else ""))
        if report.detail:
            print(f"\n  {report.detail}")
        if report.state == "would_repoint":
            print("\n  DRY RUN: nothing was written. Re-run with --apply to perform it.")
        return 0 if report.ok else 1

    if args.apply and not args.reconcile_tracking:
        parser.error("--apply only means something with --reconcile-tracking")

    if args.assess:
        if not args.assess.is_dir():
            parser.error(f"no such run folder: {args.assess}")
        result = assess_run(args.assess, mock=args.mock)
        print(f"\n{result.status.upper()}: {args.assess}")
        for issue in result.issues:
            print(f"  - {issue}")
        return 0 if result.status == "success" else 1

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
