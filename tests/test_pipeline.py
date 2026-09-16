"""The final suite: a few meaningful checks against the real artifact.

Every test here runs the production CLI once (mocked providers, zero API calls)
and then inspects the COMPILED PDF, because the PDF is what the contracts are
written about. There is no test-only renderer and no simplified pipeline.
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

import grounding
import pdf_utils as pdf
import resume_engine as engine
import run_pipeline

ROOT = Path(__file__).resolve().parent.parent
# The role fixtures moved to jobs/trial/ when jobs/ became the home of real
# postings. Same Northstar JD, so every Backend-specific assertion still holds.
JOB = ROOT / "jobs" / "trial" / "backend_swe.txt"

# The rewrite the pipeline exists to prevent: an LLM "improving" SWE-1.
DISALLOWED_REWRITE = ("Led end-to-end delivery of platform modules, translating client "
                      "requirements into prototypes and decomposing finalized scope across "
                      "a small team.")


@pytest.fixture(scope="session")
def run(tmp_path_factory) -> dict:
    """Run the actual CLI, then hand back the artifacts it produced."""
    completed = subprocess.run(
        [sys.executable, "run_pipeline.py", str(JOB), "--mock", "--smoke-test"],
        cwd=ROOT, capture_output=True, text=True, timeout=1800)
    # Take the folder the CLI reports. Picking the newest directory instead is
    # wrong: rewriting existing files does not change a directory's mtime, so
    # another run's folder can look newer than this one's.
    reported = re.search(r"^(?:SUCCESS|NEEDS_REVIEW|FAILED): (.+)$",
                         completed.stdout, re.MULTILINE)
    assert reported, f"the CLI reported no run folder\n{completed.stdout[-3000:]}"
    run_dir = Path(reported.group(1).strip())
    # A successful run renames the verified PDF to <Company>_<Job_Title>.pdf,
    # so there is exactly one PDF and it is not called resume.pdf.
    produced = sorted(run_dir.glob("*.pdf"))
    assert len(produced) == 1, f"expected one final PDF, got {produced}"
    pdf_path = produced[0]
    assert pdf_path.name != "resume.pdf", pdf_path.name

    master = engine.load_master()
    policy = engine.load_policy()
    template = engine.load_template()
    jd = engine.read_jd(run_dir / "job_description.txt")
    signals = engine.classify_jd(jd.text, master.section_order)
    lines = pdf.extract_lines(pdf_path)
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "dir": run_dir,
        "pdf": pdf_path,
        "text": pdf.extract_text(pdf_path),
        "lines": lines,
        "right_edge": pdf.body_right_edge(lines),
        "strategy": json.loads((run_dir / "strategy.json").read_text()),
        "tex": (run_dir / "resume.tex").read_text(),
        "log": (run_dir / "run.log").read_text(),
        "master": master, "policy": policy, "template": template, "signals": signals,
        "decision": engine.select_experience(template, policy, signals),
    }


# ---- A. Professional Experience is immutable -------------------------------

def test_experience_matches_approved_wording_in_the_pdf(run):
    bullets = pdf.extract_bullets(run["lines"], run_pipeline.EXPERIENCE_HEADING,
                                  run["right_edge"])
    exact, problems = engine.verify_experience_in_pdf([b.text for b in bullets], run["decision"])
    assert exact, "\n".join(problems)
    assert len(bullets) == len(run["policy"].experience_bullet_ids)
    assert len(bullets) == len(run["decision"].shipped_ids)


def test_experience_swap_is_the_one_the_spreadsheet_authorizes(run):
    assert "Containerized 3+ production projects using Docker" in run["text"]
    assert "color-blindness" not in run["text"].lower()


def test_no_llm_paraphrase_of_experience(run):
    """Every shipped id's spreadsheet text must appear verbatim in the PDF."""
    normalized = engine.normalize_plain(run["text"])
    assert engine.normalize_plain(DISALLOWED_REWRITE) not in normalized
    policy = run["policy"]
    for bullet_id in run["decision"].shipped_ids:
        approved = policy.approved_text(bullet_id)
        assert approved, f"no spreadsheet wording for id={bullet_id}"
        plain = engine.normalize_plain(engine.latex_to_plain(approved))
        assert plain in normalized, f"id={bullet_id} wording is not in the PDF: {plain[:80]!r}"


def test_experience_swaps_are_keyed_by_id(run):
    swaps = run["strategy"]["experience"]["swaps"]
    assert [(s["source_id"], s["target_id"]) for s in swaps] == [("SWE-4", "SWE-ALT-DOCKER")]
    assert run["strategy"]["experience"]["wording_source"] == "Resume_analysis.xlsx"
    assert "SWE-ALT-DOCKER" in run["strategy"]["experience"]["shipped_ids"]
    assert "SWE-4" not in run["strategy"]["experience"]["shipped_ids"]


def test_template_owns_no_experience_prose():
    """The template is presentation only; its Experience block is a placeholder."""
    raw = engine.TEMPLATE_PATH.read_text()
    block = engine.load_template().blocks["Experience"]
    assert engine.PLACEHOLDER_EXPERIENCE in block
    assert r"\item" not in block
    for prose in ("platform modules", "prototyped", "color-blindness", "fuzzy-matching",
                  "telemedicine", "Architected relational", "HeinOnline", "Thesis Mumbai"):
        assert prose not in raw, f"template still contains Experience prose: {prose!r}"


def test_a_wording_edit_does_not_break_a_swap_mapping(run):
    """Swaps resolve by id, so re-wording a bullet must not unmap anything."""
    import dataclasses

    policy = run["policy"]
    edited = tuple(
        dataclasses.replace(entry, latex=r"\item Completely different approved wording here.")
        if entry.bullet_id == "SWE-4" else entry
        for entry in policy.experience_library)
    rewritten = dataclasses.replace(policy, experience_library=edited)

    assert engine.verify_experience_ids(rewritten) == []
    decision = engine.select_experience(run["template"], rewritten, run["signals"])
    assert [(a, b) for a, b, _ in decision.swaps] == [("SWE-4", "SWE-ALT-DOCKER")]
    assert "SWE-ALT-DOCKER" in decision.shipped_ids


def test_integrity_catches_a_missing_alternate_id(run):
    import dataclasses

    policy = run["policy"]
    broken = dataclasses.replace(
        policy, alternate_library=tuple(e for e in policy.alternate_library
                                        if e.bullet_id != "SWE-ALT-INTERVIEWS"))
    problems = engine.verify_experience_ids(broken)
    assert any("SWE-ALT-INTERVIEWS" in p for p in problems), problems

    duplicated = dataclasses.replace(
        policy, experience_library=policy.experience_library + (policy.base_entry("SWE-1"),))
    assert any("duplicate" in p and "SWE-1" in p
               for p in engine.verify_experience_ids(duplicated))


def test_run_log_reports_the_integrity_gate(run):
    assert "EXPERIENCE ID INTEGRITY VERIFIED" in run["log"]
    # The gate must run before any provider work begins.
    assert run["log"].index("EXPERIENCE ID INTEGRITY VERIFIED") < run["log"].index("[PROVIDER]")


# ---- B. Technical Skills renders exactly the target line count -------------

def test_technical_skills_render_exactly_eight_lines(run):
    labels = [f"{label}: {', '.join(items)}"
              for label, items in _rendered_skill_categories(run)]
    categories = pdf.measure_skill_categories(run["lines"], labels, run["right_edge"])
    rendered = pdf.total_skill_lines(categories)
    assert rendered == run["policy"].skills_target_lines, \
        f"{rendered} rendered lines: " + "; ".join(f"{m.label}={m.lines}" for m in categories)
    assert run["strategy"]["verification"]["skills_rendered_lines"] == rendered


def test_skills_are_supported_and_deduplicated(run):
    skills = [s for _label, items in _rendered_skill_categories(run) for s in items]
    folded = [engine.fold_term(s) for s in skills]
    assert len(folded) == len(set(folded)), "a skill appears in more than one category"
    master = run["master"]
    mandatory = {engine.fold_term(i) for _l, items in run["policy"].mandatory for i in items}
    for skill in skills:
        if engine.fold_term(skill) in mandatory:
            continue                              # the spreadsheet's own baseline
        assert master.canonical_skill(skill), f"{skill!r} is not in the claimable pool"


def _section_tex(run, heading: str) -> str:
    """The LaTeX of exactly one rendered section."""
    body = run["tex"].split(r"\section*{" + heading + "}", 1)
    assert len(body) == 2, f"{heading} is not in the generated LaTeX"
    return body[1].split(r"\sectiondivider", 1)[0]


def _rendered_skill_categories(run) -> list[tuple[str, list[str]]]:
    import re
    out = []
    for match in re.finditer(r"\\item \\textbf\{([^}]*):\}\s*([^\n]*)",
                             _section_tex(run, "TECHNICAL SKILLS")):
        label = match.group(1).replace("\\&", "&").replace("\\", "")
        body = engine.latex_to_plain(match.group(2))
        out.append((label, [s.strip() for s in engine._split_skill_items(body)]))
    assert out, "no Technical Skills categories were rendered"
    return out


# ---- C. Project headers carry no technology stack --------------------------

def test_project_headers_have_no_tech_stack(run):
    headers = pdf.extract_non_bullet_lines(run["lines"], run_pipeline.PROJECTS_HEADING)
    assert len(headers) == run["policy"].project_count
    for header in headers:
        leaks = run_pipeline.header_tech_leaks(header, run["master"])
        assert not leaks, f"{header!r} -> {leaks}"
    assert "| Python," not in run["text"]
    assert run["strategy"]["verification"]["project_headers_clean"] is True


def test_projects_follow_the_allocation_contract(run):
    allocation = run["strategy"]["projects"]["allocation"]
    selected = run["strategy"]["projects"]["selected_by_relevance"]
    assert len(selected) == run["policy"].project_count
    assert [allocation[p] for p in selected] == list(run["policy"].bullet_allocation)
    bullets = pdf.extract_bullets(run["lines"], run_pipeline.PROJECTS_HEADING,
                                  run["right_edge"])
    assert len(bullets) == sum(run["policy"].bullet_allocation)


# ---- D. Grounding rejects unsupported facts --------------------------------

def test_grounding_accepts_the_shipped_bullets(run):
    master = run["master"]
    for project_id in run["strategy"]["projects"]["selected"]:
        project = master.project(project_id)
        for bullet in _bullets_for(run, project_id):
            problems = grounding.errors(
                grounding.check_bullet_grounding(bullet, project, master,
                                                 run["policy"].banned_phrases))
            assert not problems, f"{project_id}: {[str(p) for p in problems]}"


@pytest.mark.parametrize("bullet, expected_kind", [
    ("Implemented the real-time subsystem serving 90+ concurrent users with sub-200 ms latency.",
     "metric"),
    ("Deployed the service on Kubernetes with a Kafka event bus for grading events.",
     "explicitly_unsupported"),
    ("Built the scheduler and cut p95 REST latency by 26%.", "metric"),
    ("Trained a GraphSAGE model on IEEE-CIS transactions inside the LMS.", "borrowed_technology"),
])
def test_grounding_rejects_invented_facts(run, bullet, expected_kind):
    project = run["master"].project("lms")
    problems = grounding.check_bullet_grounding(bullet, project, run["master"])
    kinds = {p.kind for p in grounding.errors(problems)}
    assert expected_kind in kinds, f"expected {expected_kind}, got {kinds}"


def test_metric_semantics_have_no_tolerance():
    evidence = grounding.extract_metrics(
        "Supported 45+ users with sub-500 ms latency, ~90% less manual work, 20+ interviews.")
    for claim, supported in [("45+ users", True), ("45 users", False),
                             ("sub-500 ms", True), ("500 ms", False),
                             ("under 500 ms", True), ("~90%", True), ("90%", False),
                             ("20+ interviews", True), ("more than 20 interviews", False)]:
        parsed = grounding.extract_metrics(claim)
        got = bool(parsed) and grounding.metric_supported(parsed[0], evidence) is not None
        assert got is supported, f"{claim!r} should be {'supported' if supported else 'rejected'}"


def _bullets_for(run, project_id: str) -> list[str]:
    """The bullets shipped for one project, read back out of the LaTeX."""
    import re
    project = run["master"].project(project_id)
    section = _section_tex(run, "ACADEMIC PROJECTS")
    chunk = section.split(engine.typeset(project.name), 1)
    assert len(chunk) == 2, f"{project_id} is not in the projects section"
    body = chunk[1].split(r"\noindent", 1)[0]
    bullets = [engine.latex_to_plain(m.group(0))
               for m in re.finditer(r"\\item [^\n]+", body)]
    assert bullets, f"no bullets found for {project_id}"
    return bullets


# ---- E. The end-to-end production path -------------------------------------

def test_cli_produces_every_required_artifact(run):
    for name in run_pipeline.REQUIRED_ARTIFACTS:
        path = run["dir"] / name
        assert path.exists() and path.stat().st_size > 0, f"missing or empty: {name}"
    assert run["returncode"] == 0, run["stdout"][-3000:]
    assert run["strategy"]["status"] == "success", run["strategy"]["verification"]["issues"]


def test_run_log_documents_every_stage(run):
    for stage in ("[JOB]", "[INPUT]", "[POLICY]", "[SECTION ORDER]", "[EXPERIENCE]",
                  "[PROJECT SELECTION]", "[PROJECT WRITING]", "[SKILLS]", "[LAYOUT]",
                  "[PDF]", "[VERIFY]", "[COVER LETTER]", "[ASSESSMENT]", "[FINAL]"):
        assert stage in run["log"], f"run.log never reaches {stage}"
    assert "KEEP_EXACT" in run["log"] and "action=SWAP" in run["log"]
    assert "TARGET_REACHED" in run["log"]
    assert "EXPERIENCE VERIFIED" in run["log"]


def test_strategy_json_has_the_agreed_shape(run):
    strategy = run["strategy"]
    for key in ("company_name", "job_title", "job_id", "role_family", "section_order",
                "experience", "projects", "skills", "verification"):
        assert key in strategy, f"strategy.json is missing {key}"
    assert set(strategy["experience"]) >= {"kept", "swaps"}
    assert set(strategy["projects"]) >= {"selected", "allocation", "selection_reasons"}
    assert set(strategy["skills"]) >= {"baseline", "project_candidates",
                                       "experience_jd_candidates", "added", "removed",
                                       "final_lines"}
    assert set(strategy["verification"]) >= {"pages", "skills_rendered_lines",
                                             "experience_exact", "project_headers_clean",
                                             "issues"}


def test_resume_is_one_page_without_layout_tricks(run):
    assert pdf.page_count(run["pdf"]) == run["policy"].page_count
    assert run["strategy"]["verification"]["pages"] == run["policy"].page_count
    for hack in engine.FORBIDDEN_LAYOUT_HACKS:
        assert hack not in run["tex"], f"generated LaTeX contains {hack!r}"
    bullets = pdf.extract_bullets(run["lines"], run_pipeline.PROJECTS_HEADING,
                                  run["right_edge"])
    policy = run["policy"]
    orphans = [b.text[:60] for b in bullets
               if b.verdict(policy.tail_orphan_max, policy.tail_acceptable_max,
                            policy.tail_ideal_max) == "hard_orphan"]
    assert not orphans, f"hard orphan tails: {orphans}"


def test_mock_run_never_touches_production_tracking(run):
    assert run["dir"].parent.name == "_smoke_tests"
    assert "tracking skipped" in run["log"]
    csv_rows = engine.PROCESSED_CSV.read_text().splitlines() \
        if engine.PROCESSED_CSV.exists() else []
    assert not any(run["dir"].name in row for row in csv_rows)
    assert run["strategy"]["mode"] == "mock"


def test_zero_live_provider_calls(run):
    assert all(call["provider"] == "mock" for call in run["strategy"]["provider_calls"]), \
        run["strategy"]["provider_calls"]
    assert "no network call" in run["log"]


# ---- F. Spreadsheet-derived tail bands ------------------------------------

@pytest.mark.parametrize("fill, expected", [
    (19, "hard_orphan"), (20, "acceptable"), (36, "acceptable"), (40, "acceptable"),
    (41, "ideal"), (60, "ideal"), (61, "healthy"),
])
def test_tail_bands_match_the_spreadsheet(fill, expected):
    policy = engine.load_policy()
    assert (policy.tail_orphan_max, policy.tail_acceptable_max,
            policy.tail_ideal_max, policy.tail_healthy_min) == (20.0, 40.0, 60.0, 61.0)
    rendered = pdf.BulletRender(text="x", lines=2, fill_pct=fill, index_in_section=0)
    assert rendered.verdict(policy.tail_orphan_max, policy.tail_acceptable_max,
                            policy.tail_ideal_max) == expected


# ---- G. The action-verb limit is a final-PDF invariant --------------------

def test_action_verb_limit_holds_in_the_final_pdf(run):
    experience = pdf.extract_bullets(run["lines"], run_pipeline.EXPERIENCE_HEADING,
                                     run["right_edge"])
    projects = pdf.extract_bullets(run["lines"], run_pipeline.PROJECTS_HEADING,
                                   run["right_edge"])
    counts = engine.count_opening_verbs([b.text for b in experience]
                                        + [b.text for b in projects])
    limit = run["policy"].verb_max_uses
    assert engine.verb_violations(counts, limit) == []
    assert run["strategy"]["verification"]["action_verbs_ok"] is True
    assert "action verb repetition verified" in run["log"]


def test_project_writer_rejects_a_verb_experience_already_used_twice(run):
    """Experience owns 'Engineered' twice; a project may not take a third."""
    import logging

    import llm_client

    master, policy = run["master"], run["policy"]
    project = master.project("lms")
    forced = json.dumps({"project_id": "lms", "bullets": [
        "Engineered a decoupled learning management system with a Django REST Framework "
        "backend and a React SPA frontend, using PostgreSQL for durable persistence.",
        "Implemented the real-time subsystem with Django Channels, Daphne, Redis, and "
        "WebSockets, serving live quizzes to 45+ concurrent users.",
        "Optimized REST paths under Locust load tests of up to 100 concurrent users, "
        "cutting p95 REST latency by approximately 26%.",
    ]})
    transport = _StubTransport(forced)
    client = llm_client.LLMClient(transport, transport, master, policy,
                                  logging.getLogger("test"))
    # Experience has already spent the whole budget for this verb.
    bullets, problems = client.write_bullets(project, 3, run["strategy"]["role_family"],
                                            {"engineered": 2}, (150, 215))
    kinds = {p.kind for p in grounding.errors(problems)}
    assert "verb_budget" in kinds, [str(p) for p in problems]
    assert len(transport.requests) == 3, "the writer should re-prompt within its bounded budget"

    counts = engine.count_opening_verbs(["Engineered x"] * 2 + [bullets[0]])
    assert engine.verb_violations(counts, policy.verb_max_uses) == [("engineered", 3)]


def test_available_verbs_excludes_exhausted_ones():
    counts = {"built": 2, "engineered": 3, "designed": 1}
    options = engine.available_verbs(counts, 2, grounding.PREFERRED_VERBS)
    assert "Built" not in options and "Engineered" not in options
    assert "Designed" in options


# ---- H. Provider mode and provenance -------------------------------------

class _StubTransport:
    """Records the requests it receives and replays a fixed payload."""

    name = "stub"

    def __init__(self, payload: str, provider: str = "stub"):
        self.payload = payload
        self.provider = provider
        self.requests = []

    def generate(self, request):
        import llm_client
        self.requests.append(request)
        return llm_client.Reply(self.payload, self.provider)


def test_cover_letter_is_requested_as_plain_text(run):
    import logging

    import llm_client

    letter = llm_client.MockTransport(run["master"], logging.getLogger("test"))._letter(
        {"job_title": "Backend Software Engineer", "company": "Northstar Software",
         "project_ids": ["lms"], "jd_text": "", "needs_eligibility": False})
    transport = _StubTransport(letter)
    client = llm_client.LLMClient(transport, transport, run["master"], run["policy"],
                                  logging.getLogger("test"))
    jd = engine.read_jd(run["dir"] / "job_description.txt")
    themes = engine.jd_themes(engine.extract_jd_requirements(jd.text, run["master"]), 5)
    text, _problems = client.cover_letter(jd, run["signals"], [],
                                          [run["master"].project("lms")], themes=themes)
    assert transport.requests[0].json is False, "the letter must not go through JSON mode"
    assert transport.requests[0].purpose == "cover_letter"
    assert text.startswith("Dear Hiring Manager,")
    assert "{" not in text.splitlines()[0]


def test_assessment_still_uses_json_mode(run):
    import logging

    import llm_client

    jd = engine.read_jd(run["dir"] / "job_description.txt")
    master = run["master"]
    requirements = engine.jd_themes(engine.extract_jd_requirements(jd.text, master), 3)
    # The stub must cite a real id: the schema gate rejects invented identities.
    payload = {"fit_score": 8.0, "recommendation": "apply", "summary": "Good match.",
               "eligibility": {"status": "meets", "details": ["no stated requirement"]},
               "strong_matches": [{"requirement_id": requirements[0].requirement_id,
                                   "evidence": "Python is used in Professional Experience",
                                   "source": "experience"}],
               "partial_matches": [], "gaps": [],
               "tailoring_quality": {"score": 9.0, "notes": ["8 skills lines"]},
               "risk_flags": []}
    transport = _StubTransport(json.dumps(payload))
    client = llm_client.LLMClient(transport, transport, run["master"], run["policy"],
                                  logging.getLogger("test"))
    data = client.assess(jd, run["signals"], [master.project("lms")], ["Python"],
                         experience_plain=["Built things"],
                         project_bullets={"lms": ["Engineered a thing."]},
                         requirements=requirements)
    assert transport.requests[0].json is True
    assert data["fit_score"] == 8.0


def test_groq_failure_reports_gemini_as_the_provider():
    import logging

    import llm_client

    credentials = engine.Credentials(gemini_accounts=(1,), has_groq=True,
                                     _gemini_keys={1: "unused-in-this-test"},
                                     _groq_key="unused-in-this-test")
    fallback = _StubTransport("Dear Hiring Manager, ...", provider="gemini")

    class _FailingGroq(llm_client.GroqTransport):
        def _generate(self, request):
            raise llm_client.ProviderError("bad_request",
                                           "400 json_validate_failed")

    transport = _FailingGroq(credentials, logging.getLogger("test"), fallback=fallback)
    reply = transport.generate(llm_client.Request("cover_letter", "prompt", json=False))
    assert (reply.provider, reply.detail) == ("gemini", "fallback_from_groq")

    client = llm_client.LLMClient(fallback, transport, engine.load_master(),
                                  engine.load_policy(), logging.getLogger("test"))
    client._invoke(transport, llm_client.Request("cover_letter", "prompt", json=False))
    record = client.calls[-1]
    assert (record.purpose, record.provider) == ("cover_letter", "gemini")
    assert record.transport_ok and record.accepted
    assert record.detail == "fallback_from_groq"


# ---- I. Relevance decides allocation; chronology decides display ----------

def _dated_master(dates: dict[str, str]):
    """A minimal stand-in carrying only canonical project dates."""
    import types
    projects = {pid: engine.Project(project_id=pid, name=pid, context="", date=date,
                                    tech=(), evidence=(), tags=())
                for pid, date in dates.items()}
    return types.SimpleNamespace(project=projects.get, projects=tuple(projects.values()))


def _allocate(relevance: list[str]) -> dict[str, int]:
    return engine.allocate_bullets(relevance, engine.load_policy())


def test_display_order_when_most_relevant_is_newest():
    master = _dated_master({"A": "August 2026", "B": "July 2026", "C": "November 2025"})
    relevance = ["A", "B", "C"]
    assert engine.display_order(relevance, master) == ["A", "B", "C"]
    assert _allocate(relevance) == {"A": 3, "B": 2, "C": 2}


def test_display_order_when_most_relevant_is_second_newest():
    master = _dated_master({"A": "August 2026", "B": "September 2026", "C": "November 2025"})
    relevance = ["A", "B", "C"]
    display = engine.display_order(relevance, master)
    allocation = _allocate(relevance)
    assert display == ["B", "A", "C"]
    # The first displayed project must not inherit the 3-bullet allocation.
    assert [allocation[pid] for pid in display] == [2, 3, 2]
    assert allocation["A"] == 3


def test_display_order_when_most_relevant_is_oldest():
    master = _dated_master({"A": "January 2025", "B": "June 2026", "C": "December 2026"})
    relevance = ["A", "B", "C"]
    display = engine.display_order(relevance, master)
    allocation = _allocate(relevance)
    assert display == ["C", "B", "A"]
    assert display[-1] == "A" and allocation["A"] == 3
    assert [allocation[pid] for pid in display] == [2, 2, 3]


def test_same_date_falls_back_to_relevance_rank():
    master = _dated_master({"A": "August 2026", "B": "August 2026", "C": "March 2024"})
    assert engine.display_order(["A", "B", "C"], master) == ["A", "B", "C"]
    assert engine.display_order(["B", "A", "C"], master) == ["B", "A", "C"]


def test_real_project_dates_drive_the_order():
    master = engine.load_master()
    assert engine.display_order(["lms", "temp", "tailor_pipeline"], master) == [
        "tailor_pipeline", "lms", "temp"]


def test_pdf_renders_projects_chronologically_with_relevance_allocation(run):
    """The compiled PDF is the authority for both order and allocation."""
    projects = run["strategy"]["projects"]
    display = projects["display_order"]
    allocation = projects["allocation"]
    relevance = projects["selected_by_relevance"]

    blocks = pdf.project_blocks(run["lines"], run_pipeline.PROJECTS_HEADING, run["right_edge"])
    assert len(blocks) == run["policy"].project_count
    rendered = [run_pipeline._project_id_for_header(header, run["master"])
                for header, _bullets in blocks]
    assert rendered == display, f"rendered {rendered}, expected {display}"

    for project_id, (_header, bullets) in zip(rendered, blocks):
        assert len(bullets) == allocation[project_id], \
            f"{project_id} rendered {len(bullets)} bullets, earned {allocation[project_id]}"

    # The most relevant project keeps 3 bullets wherever it is displayed.
    top = relevance[0]
    assert allocation[top] == max(allocation.values()) == 3
    assert dict(zip(rendered, (len(b) for _h, b in blocks)))[top] == 3

    # Dates really are descending in the rendered order.
    keys = [engine.project_date_key(run["master"].project(pid)) for pid in rendered]
    assert keys == sorted(keys, reverse=True), keys
    assert "[PROJECT ORDER]" in run["log"]
    assert "PROJECT ORDER VERIFIED" in run["log"]


# ---- J. Cover-letter quality ----------------------------------------------

def test_cover_letter_passes_quality_and_grounding(run):
    letter = (run["dir"] / "cover_letter.txt").read_text()
    jd = engine.read_jd(run["dir"] / "job_description.txt")
    quality = grounding.errors(grounding.validate_letter_quality(
        letter, company=jd.company_name, job_title=jd.job_title))
    assert not quality, [str(p) for p in quality]
    ground = grounding.errors(grounding.validate_cover_letter(
        letter, run["master"], masked_terms=(jd.job_id,) if jd.job_id else (),
        banned=run["policy"].banned_phrases))
    assert not ground, [str(p) for p in ground]


def test_cover_letter_names_company_role_and_stays_in_range(run):
    letter = (run["dir"] / "cover_letter.txt").read_text()
    jd = engine.read_jd(run["dir"] / "job_description.txt")
    assert jd.company_name in letter
    assert any(f.lower() in letter.lower()
               for f in grounding._normalized_title_forms(jd.job_title))
    words = grounding.word_count(letter)
    assert 140 <= words <= 300, words
    assert run["strategy"]["cover_letter"]["validation"] == "pass"


def test_cover_letter_preserves_metric_and_threshold_semantics(run):
    letter = (run["dir"] / "cover_letter.txt").read_text()
    assert "10k+" in letter and "10,000" not in letter
    for forbidden in ("three production projects", "500 ms broadcast latency without"):
        assert forbidden not in letter
    evidence = grounding.extract_metrics(run["master"].raw)
    for claim in grounding.extract_metrics(letter):
        if claim.unit in ("%", "s", "ms") or claim.structure != "exact":
            assert grounding.metric_supported(claim, evidence), claim.describe()


def test_refreshed_mock_letter_uses_current_approved_wording():
    import logging

    import llm_client

    transport = llm_client.MockTransport(engine.load_master(), logging.getLogger("test"))
    letter = transport._letter({"job_title": "Backend Software Engineer",
                                "company": "Northstar Software",
                                "project_ids": ["lms"], "jd_text": "real-time websockets",
                                "needs_eligibility": False})
    # A2 allows paraphrase in the letter; the canonical form lives in the sheet.
    assert "prototyped to validate scope" in (
        engine.load_policy().approved_text("SWE-1") or "")
    assert "prototyping to validate scope" in letter
    assert "built prototypes to validate scope" not in letter
    assert "built prototypes" not in letter
    assert "the kind of production ownership this role describes" not in letter
    # Templated per posting, not a copy of one specific letter.
    for field in ("{job_title}", "{company}", "{eligibility}"):
        assert field in llm_client.MOCK_OPENING
    for field in ("{paragraph_one}", "{paragraph_two}", "{project_paragraph}"):
        assert field in llm_client.MOCK_LETTER
    assert "Backend Software Engineer" in letter and "Northstar Software" in letter


def test_letter_validator_rejects_fabrication_and_ownership_transfer(run):
    master = run["master"]
    fabricated = ("Dear Hiring Manager, I built healthcare modules supporting 50k+ records "
                  "and cut latency by 40% on the WebSocket pipeline I led.")
    problems = grounding.validate_cover_letter(fabricated, master)
    kinds = {p.kind for p in grounding.errors(problems)}
    assert "metric" in kinds
    assert grounding.errors(grounding.check_ownership_fusion(
        "I led the WebSocket and Redis pipeline for IoT sensor data.", master))


def test_letter_quality_rejects_generic_and_markdown():
    base = " ".join(["Backend Software Engineer at Northstar Software"] * 30)
    generic = "I am writing to express my strong interest in this role. " + base
    kinds = {p.kind for p in grounding.errors(grounding.validate_letter_quality(
        generic, company="Northstar Software", job_title="Backend Software Engineer"))}
    assert "generic" in kinds
    markdown = "**Backend Software Engineer** at Northstar Software. " + base
    kinds = {p.kind for p in grounding.errors(grounding.validate_letter_quality(
        markdown, company="Northstar Software", job_title="Backend Software Engineer"))}
    assert "format" in kinds


# ---- K. Assessment quality -----------------------------------------------

def _assessment(run) -> dict:
    # assessment.json is gone by design; the rich structure lives in
    # strategy.json while assessment.txt stays a small advisory report.
    block = json.loads((run["dir"] / "strategy.json").read_text())["assessment"]
    detail = block.get("detail") or {}
    return {**block, **detail,
            "eligibility": {"status": block.get("eligibility"),
                            "details": block.get("eligibility_detail") or []},
            "tailoring_quality": {"score": block.get("tailoring_quality"),
                                  "notes": block.get("tailoring_notes") or []}}


def test_assessment_matches_the_schema(run):
    jd = engine.read_jd(run["dir"] / "job_description.txt")
    problems = grounding.errors(grounding.validate_assessment(_assessment(run), jd.text))
    assert not problems, [str(p) for p in problems]


def test_assessment_rejects_a_vague_gap():
    vague = {"fit_score": 8, "recommendation": "apply", "summary": "Fine.",
             "eligibility": {"status": "meets", "details": []},
             "strong_matches": [], "partial_matches": [],
             "gaps": [{"requirement_id": "REQ-001",
                       "requirement": "No experience with the parts of the stack the master "
                                      "does not evidence",
                       "importance": "required", "status": "unsupported", "evidence": None}],
             "tailoring_quality": {"score": 9, "notes": []}, "risk_flags": []}
    problems = grounding.errors(grounding.validate_assessment(
        vague, "python backend redis", {"REQ-001"}))
    assert any(p.kind == "vague_gap" for p in problems), [str(p) for p in problems]


def test_every_gap_names_something_from_the_posting(run):
    jd = engine.read_jd(run["dir"] / "job_description.txt")
    for gap in _assessment(run).get("gaps") or []:
        assert grounding._is_vague(gap["requirement"], jd.text) is None, gap


def test_strong_matches_carry_specific_evidence(run):
    strong = _assessment(run)["strong_matches"]
    assert strong, "the resume should evidence at least one requirement"
    for match in strong:
        assert len(match["evidence"].split()) >= 3, match
        assert match["source"] in grounding.EVIDENCE_SOURCES


def test_fit_score_and_tailoring_quality_are_separate(run):
    data = _assessment(run)
    assert isinstance(data["fit_score"], (int, float))
    assert isinstance(data["tailoring_quality"]["score"], (int, float))
    assert "score" in data["tailoring_quality"] and "notes" in data["tailoring_quality"]
    assert data["tailoring_quality"]["notes"], "tailoring notes should explain the score"


def test_docker_never_satisfies_kubernetes(run):
    master = run["master"]
    evidence = engine.ResumeEvidence(
        experience_text="Containerized 3+ production projects using Docker and Docker Compose.",
        project_bullets=(("lms", "Containerized the stack with Docker Compose."),),
        skills=("Docker", "Docker Compose", "Linux"))
    requirement = engine.Requirement(
        "Experience with Kubernetes in production is required", "required",
        ("Kubernetes",), "requirements")
    match = engine.classify_requirement(requirement, master, evidence)
    assert match.verdict == "unsupported", match
    assert "Kubernetes" in match.term


def test_unsupported_required_technology_becomes_a_specific_gap(run):
    master, policy = run["master"], run["policy"]
    requirements = [engine.Requirement("CI/CD pipeline experience is required", "required",
                                       ("CI/CD",), "requirements")]
    transport = llm_client_module().MockTransport(master, __import__("logging").getLogger("t"))
    requirements = [dataclasses.replace(requirements[0], requirement_id="REQ-001")]
    data = transport._assess({"requirements": requirements, "skills": ["Python"],
                              "evidence": engine.ResumeEvidence(skills=("Python",)),
                              "tailoring": {}, "jd_text": "CI/CD pipeline experience"})
    assert data["gaps"], data
    assert data["gaps"][0]["requirement_id"] == "REQ-001"
    assert "CI/CD" in data["gaps"][0]["detail"]
    assert data["gaps"][0]["importance"] == "required"


def test_eligibility_recognizes_the_graduation_window():
    import logging

    master = engine.load_master()
    jd = engine.read_jd(ROOT / "jobs_synthetic" / "newgrad_swe.txt")
    grad = engine.graduation_requirement(jd.text)
    assert grad and grad["low"] == (2026, 12) and grad["high"] == (2027, 6)
    transport = llm_client_module().MockTransport(master, logging.getLogger("t"))
    data = transport._assess({"requirements": engine.extract_jd_requirements(jd.text, master),
                              "skills": [], "tailoring": {},
                              "evidence": engine.ResumeEvidence(), "jd_text": jd.text})
    assert data["eligibility"]["status"] == "meets", data["eligibility"]
    assert any("December 2026" in d for d in data["eligibility"]["details"])


def llm_client_module():
    import llm_client
    return llm_client


# ---- L. Compound requirements and final-resume evidence -------------------

def _final_evidence(run) -> engine.ResumeEvidence:
    """The evidence corpus exactly as the pipeline builds it for this run."""
    decision = run["decision"]
    bullets = {}
    for project_id in run["strategy"]["projects"]["display_order"]:
        bullets[project_id] = " ".join(_bullets_for(run, project_id))
    skills = tuple(s for _label, items in _rendered_skill_categories(run) for s in items)
    return engine.ResumeEvidence(
        experience_text=" ".join(engine.latex_to_plain(l) for _b, _a, l in decision.shipped),
        project_bullets=tuple(bullets.items()), skills=skills)


def _requirement(run, text: str, importance: str = "required") -> engine.Requirement:
    parsed = engine.extract_jd_requirements(f"Requirements:\n- {text}\n", run["master"])
    assert parsed, text
    return engine.Requirement(text, importance, parsed[0].terms, "requirements")


def test_compound_requirement_needs_every_group(run):
    """One matched term must not carry a whole compound line."""
    master = run["master"]
    requirement = _requirement(
        run, "PostgreSQL or MySQL, Kubernetes, and Terraform")
    groups = engine.requirement_groups(requirement)
    assert len(groups) >= 2, groups
    # PostgreSQL alone is matched, but Kubernetes/Terraform are unsupported.
    match = engine.classify_requirement(requirement, master, _final_evidence(run))
    assert match.verdict != "strong_match", match
    assert "Kubernetes" in match.limitation or "Terraform" in match.limitation


def test_or_alternative_group_is_satisfied_by_either_term(run):
    requirement = _requirement(run, "Experience with PostgreSQL or MySQL")
    groups = engine.requirement_groups(requirement)
    assert len(groups) == 1 and len(groups[0]) == 2, groups
    match = engine.classify_requirement(requirement, run["master"], _final_evidence(run))
    assert match.verdict == "strong_match", match


def test_python_plus_kubernetes_is_never_strong(run):
    requirement = _requirement(run, "Experience with Python and Kubernetes in production")
    match = engine.classify_requirement(requirement, run["master"], _final_evidence(run))
    assert match.verdict in ("partial_match", "unsupported"), match
    assert "Kubernetes" in match.limitation


def test_c_never_satisfies_cplusplus(run):
    requirement = _requirement(run, "Strong C++ systems programming")
    evidence = engine.ResumeEvidence(
        experience_text="", project_bullets=(("pintos", "an x86 kernel written in C"),),
        skills=("C", "Operating Systems"))
    match = engine.classify_requirement(requirement, run["master"], evidence)
    assert match.verdict != "strong_match", match
    assert "C++" in (match.term or "") or "C++" in match.limitation
    assert "C++" not in (match.evidence or ""), "C must never be presented as C++"


def test_final_docker_experience_satisfies_containerization(run):
    requirement = _requirement(
        run, "Containerize applications and keep development and deployment "
             "environments consistent")
    match = engine.classify_requirement(requirement, run["master"], _final_evidence(run))
    assert match.verdict == "strong_match", match
    assert match.source in ("experience", "skills", "project")


def test_final_pintos_bullets_are_visible_to_the_testing_classifier(run):
    """Testing/debugging evidence must come from what actually shipped."""
    evidence = _final_evidence(run)
    assert any(pid == "pintos" for pid, _text in evidence.project_bullets) or \
        "pintos" in run["strategy"]["projects"]["display_order"]
    requirement = _requirement(run, "Work with teammates on testing and debugging")
    match = engine.classify_requirement(requirement, run["master"], evidence)
    assert match.verdict in ("strong_match", "partial_match"), match
    assert match.evidence, "the explanation must cite the shipped evidence"
    # The shipped Pintos bullet says "tests"; morphology must find it there
    # rather than falling back to the master.
    testing = engine.Requirement("Experience with testing", "required",
                                 ("testing",), "requirements")
    testing_match = engine.classify_requirement(testing, run["master"], evidence)
    assert testing_match.verdict == "strong_match", testing_match
    assert "not surfaced" not in (testing_match.evidence or ""), testing_match


def test_unsupported_section_is_never_positive_evidence():
    master = engine.load_master()
    for forbidden in ("Kubernetes", "CI/CD", "FastAPI", "Kafka", "Azure", "Terraform"):
        assert not master.supported_anywhere(forbidden), forbidden
    for real in ("Python", "Redis", "Docker", "PostgreSQL"):
        assert master.supported_anywhere(real), real


def test_theme_touch_and_named_term_metrics_are_consistent(run):
    letter = (run["dir"] / "cover_letter.txt").read_text()
    block = run["strategy"]["cover_letter"]
    assert set(block["themes_touched"]) <= set(block["jd_themes"])
    assert set(block["named_terms_covered"]) <= set(block["named_terms"])
    assert block["named_terms_covered"] == grounding.letter_named_terms(
        letter, block["named_terms"])
    for term in block["named_terms_covered"]:
        assert term.split(" (")[0].lower().rstrip("s") in letter.lower()


def test_assessment_counts_match_the_written_arrays(run):
    data = _assessment(run)
    assert f"{len(data['strong_matches'])} of" in data["summary"], data["summary"]
    notes = " ".join(data["tailoring_quality"]["notes"])
    assert f"{len(data['strong_matches'])} of" in notes, notes


# ---- M. Corpus completeness and honest gaps ------------------------------

def test_latex_to_plain_keeps_the_first_word_of_a_bold_header():
    """`\\noindent\\textbf{Cloud Engineer Intern}` must not lose "Cloud"."""
    header = (r"\noindent\textbf{Cloud Engineer Intern} $|$ Data Maven Pvt Ltd, Mumbai, "
              r"India \hfill \textit{November 2023 -- May 2024}")
    plain = engine.latex_to_plain(header)
    assert plain.startswith("Cloud Engineer Intern"), plain
    assert "Data Maven" in plain
    policy = engine.load_policy()
    for entry in policy.experience_library:
        if entry.is_role_header:
            flat = engine.latex_to_plain(entry.latex)
            assert not flat.startswith(("Student", "Engineer ", "'s")), flat
    education = engine.latex_to_plain(
        " ".join(engine.load_template().blocks["Education"].split()))
    assert "Master" in education and "\\" not in education, education


def test_shipped_internship_roles_are_not_reported_as_a_gap(run):
    """The resume ships two Intern roles; that cannot be an unsupported gap."""
    evidence = _final_evidence(run)
    policy = run["policy"]
    enriched = engine.ResumeEvidence(
        experience_text=evidence.experience_text + " " + " ".join(
            engine.latex_to_plain(e.latex) for e in policy.experience_library
            if e.is_role_header),
        project_bullets=evidence.project_bullets, skills=evidence.skills)
    assert "intern" in enriched.experience_text.lower()
    requirement = _requirement(
        run, "Coursework or internship experience in software engineering")
    match = engine.classify_requirement(requirement, run["master"], enriched)
    assert match.verdict != "unsupported", match
    assert match.evidence


def test_a_gap_requires_zero_supporting_overlap(run):
    """Weak evidence is a partial match; a gap means nothing matched at all."""
    evidence = _final_evidence(run)
    for gap in _assessment(run).get("gaps") or []:
        headline = gap["requirement"].split(" (")[0]
        requirement = _requirement(run, headline)
        match = engine.classify_requirement(requirement, run["master"], evidence)
        assert match.verdict == "unsupported", (gap, match)
    # A requirement that shares real vocabulary with the resume is not a gap.
    performance = _requirement(run, "Improve response latency as concurrent usage grows")
    assert engine.classify_requirement(
        performance, run["master"], evidence).verdict != "unsupported"


# ---- N. Domain-aware generic concepts and REST API canonicalization ------

_ISHIHARA = ("Integrated a color-blindness diagnostic module (Ishihara plate testing, "
             "~97-99% clinical sensitivity) with automated reporting and real-time result "
             "visualization for practitioners.")
_TELEMEDICINE = ("Delivered a telemedicine consultation feature (WebRTC) connecting on-site "
                 "nurses with remote doctors to review test results and procedures with "
                 "patients, alongside secure PDF report generation.")
_PINTOS_TESTS = ("Hardened the kernel using synchronization primitives and byte-wise memory "
                 "validation to prevent deadlocks, kernel panics, and invalid-memory "
                 "failures, passing 100% of 80 concurrency, memory-fault, and edge-case "
                 "tests.")


def test_clinical_testing_is_not_software_testing_evidence():
    """Ishihara plate testing and clinical test results must not qualify."""
    assert not engine._term_present(_ISHIHARA, "testing")
    assert not engine._term_present(_TELEMEDICINE, "testing")
    assert engine._term_present(_PINTOS_TESTS, "testing")
    assert engine._term_present(
        "Wrote unit tests and integration tests for the payment service.", "testing")


def test_software_testing_requirement_uses_shipped_software_evidence():
    master = engine.load_master()
    requirement = engine.extract_jd_requirements(
        "Requirements:\n- Work with teammates on testing and debugging\n", master)[0]

    clinical_only = engine.ResumeEvidence(
        experience_text=" ".join([_ISHIHARA, _TELEMEDICINE]), skills=("Python",))
    clinical = engine.classify_requirement(requirement, master, clinical_only)
    assert "Professional Experience" not in (clinical.evidence or ""), clinical

    with_pintos = engine.ResumeEvidence(
        experience_text=" ".join([_ISHIHARA, _TELEMEDICINE]),
        project_bullets=(("pintos", _PINTOS_TESTS),), skills=("Python",))
    match = engine.classify_requirement(requirement, master, with_pintos)
    assert match.verdict in ("strong_match", "partial_match"), match
    assert "pintos" in (match.evidence or ""), match.evidence


def test_rest_api_is_present_in_the_compound_groups(run):
    requirement = _requirement(
        run, "PostgreSQL or MySQL, REST API development, Docker, Linux, and Git")
    groups = engine.requirement_groups(requirement)
    flat = [list(group) for group in groups]
    assert ["PostgreSQL", "MySQL"] in flat or ["MySQL", "PostgreSQL"] in flat, flat
    assert any("REST API" in term for group in groups for term in group), flat
    for expected in ("Docker", "Linux", "Git"):
        assert [expected] in flat, (expected, flat)
    match = engine.classify_requirement(requirement, run["master"], _final_evidence(run))
    assert match.verdict == "strong_match", match
    assert "REST API" in match.evidence, match.evidence


def test_rest_api_plus_kubernetes_is_not_strong(run):
    requirement = _requirement(run, "Python, REST API development, and Kubernetes")
    groups = engine.requirement_groups(requirement)
    assert any("Kubernetes" in term for group in groups for term in group), groups
    match = engine.classify_requirement(requirement, run["master"], _final_evidence(run))
    assert match.verdict != "strong_match", match
    assert "Kubernetes" in match.limitation, match.limitation


def test_or_alternatives_do_not_imply_the_unsupported_branch(run):
    """Django satisfies "Django or FastAPI" without claiming FastAPI."""
    requirement = _requirement(run, "Experience with Django or FastAPI")
    match = engine.classify_requirement(requirement, run["master"], _final_evidence(run))
    assert match.verdict == "strong_match", match
    assert "FastAPI" not in (match.evidence or ""), match.evidence
    assert not run["master"].supported_anywhere("FastAPI")


# ---- O. Assessment interface robustness ----------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("academic projects", "project"),
    ("Academic Projects", "project"),
    ("  projects  ", "project"),
    ("Professional Experience", "experience"),
    ("work experience", "experience"),
    ("Technical Skills", "skills"),
    ("technical  skills  section", "skills"),
    ("Education Section", "education"),
])
def test_assessment_source_synonyms_are_canonicalized(raw, expected):
    data = {"strong_matches": [{"requirement": "r", "evidence": "e", "source": raw}]}
    out, events = grounding.canonicalize_assessment(data)
    assert out["strong_matches"][0]["source"] == expected
    if raw != expected:
        assert any(repr(raw) in event and repr(expected) in event for event in events), events


def test_other_assessment_enums_are_canonicalized():
    data = {"recommendation": "Strong Apply",
            "eligibility": {"status": "Does Not Meet"},
            "gaps": [{"requirement": "CI/CD required", "importance": "Must Have",
                      "status": "Not Supported"}]}
    out, events = grounding.canonicalize_assessment(data)
    assert out["recommendation"] == "strong_apply"
    assert out["eligibility"]["status"] == "does_not_meet"
    assert out["gaps"][0]["importance"] == "required"
    assert out["gaps"][0]["status"] == "unsupported"
    assert len(events) == 4


def test_unknown_source_still_fails_validation():
    data = {"fit_score": 8, "recommendation": "apply", "summary": "Fine.",
            "eligibility": {"status": "meets", "details": []},
            "strong_matches": [{"requirement_id": "REQ-001", "evidence": "Python used",
                                "source": "portfolio"}],
            "partial_matches": [], "gaps": [],
            "tailoring_quality": {"score": 9, "notes": ["ok"]}, "risk_flags": []}
    out, events = grounding.canonicalize_assessment(data)
    assert out["strong_matches"][0]["source"] == "portfolio", "no fuzzy matching"
    assert not events
    problems = grounding.errors(grounding.validate_assessment(out, "python backend"))
    assert any("portfolio" in p.message for p in problems), problems


def _stub_client(run, payload: str):
    import logging

    import llm_client
    transport = _StubTransport(payload)
    return transport, llm_client.LLMClient(transport, transport, run["master"],
                                           run["policy"], logging.getLogger("test"))


def test_schema_failure_records_transport_ok_but_not_accepted(run):
    import llm_client

    bad = json.dumps({"fit_score": 8, "recommendation": "apply", "summary": "Fine.",
                      "eligibility": {"status": "meets", "details": []},
                      "strong_matches": [{"requirement_id": "REQ-001", "evidence": "e",
                                          "source": "portfolio"}],
                      "partial_matches": [], "gaps": [],
                      "tailoring_quality": {"score": 9, "notes": ["n"]}, "risk_flags": []})
    transport, client = _stub_client(run, bad)
    jd = engine.read_jd(run["dir"] / "job_description.txt")
    with pytest.raises(llm_client.ProviderError):
        client.assess(jd, run["signals"], [run["master"].project("lms")], ["Python"],
                      experience_plain=["Built things"],
                      project_bullets={"lms": ["Engineered a thing."]},
                      requirements=engine.jd_themes(
                          engine.extract_jd_requirements(jd.text, run["master"]), 3))
    record = client.calls[-1]
    assert record.transport_ok is True
    assert record.accepted is False
    assert "schema validation failed" in record.detail
    assert record.as_dict()["ok"] is False


def test_synonym_assessment_is_accepted_after_normalization(run):
    payload = json.dumps({"fit_score": 8.4, "recommendation": "Apply",
                          "summary": "Good match.",
                          "eligibility": {"status": "Meets", "details": ["no requirement"]},
                          "strong_matches": [{"requirement_id": "REQ-001",
                                              "evidence": "Python in Professional Experience",
                                              "source": "Academic Projects"}],
                          "partial_matches": [], "gaps": [],
                          "tailoring_quality": {"score": 9.5, "notes": ["8 skills lines"]},
                          "risk_flags": []})
    transport, client = _stub_client(run, payload)
    jd = engine.read_jd(run["dir"] / "job_description.txt")
    data = client.assess(jd, run["signals"], [run["master"].project("lms")], ["Python"],
                         experience_plain=["Built things"],
                         project_bullets={"lms": ["Engineered a thing."]},
                         requirements=engine.jd_themes(
                             engine.extract_jd_requirements(jd.text, run["master"]), 3))
    # The provider response is normalized and accepted, but its requirement
    # buckets/evidence are NOT authoritative: Python's verdicts replace them.
    # Enum canonicalization itself is covered directly against
    # canonicalize_assessment() above.
    assert client.calls[-1].accepted is True
    assert data["verdict_source"] == "python_deterministic"
    themes = engine.jd_themes(engine.extract_jd_requirements(jd.text, run["master"]), 3)
    expected = engine.verdict_index(engine.deterministic_assessment(
        themes, run["master"],
        engine.ResumeEvidence(experience_text="Built things",
                              project_bullets=(("lms", "Engineered a thing."),),
                              skills=("Python",))))
    actual = {e["requirement_id"]: bucket for bucket in
              ("strong_matches", "partial_matches", "gaps", "manual_review")
              for e in data[bucket]}
    assert actual == expected
    # the model claimed REQ-001 was a strong match sourced from projects; Python
    # says partial, so nothing the model authored for it survives
    assert "REQ-001" not in [s["requirement_id"] for s in data["strong_matches"]]
    assert not any("Python in Professional Experience" == e.get("evidence")
                   for bucket in ("strong_matches", "partial_matches")
                   for e in data[bucket])


def test_grounding_rejection_then_accepted_retry_is_audited(run):
    """A rejected generation and its accepted retry both stay in the history."""
    import logging

    import llm_client

    master, policy = run["master"], run["policy"]
    project = master.project("lms")

    class _TwoShot:
        name = "stub"

        def __init__(self):
            self.sent = 0

        def generate(self, request):
            self.sent += 1
            if self.sent == 1:
                bullets = ["Deployed the stack on Kubernetes with a Kafka event bus.",
                           "Implemented Redis and WebSockets for the real-time subsystem.",
                           "Optimized REST paths under Locust load tests."]
            else:
                bullets = [
                    "Designed a decoupled learning management system with a Django REST "
                    "Framework backend and a React SPA frontend, using PostgreSQL.",
                    "Implemented the real-time subsystem with Django Channels, Daphne, Redis "
                    "and WebSockets, serving 45+ concurrent users.",
                    "Optimized REST paths under Locust and asyncio load tests of up to 100 "
                    "concurrent users, cutting p95 REST latency by approximately 26%.",
                ]
            return llm_client.Reply(json.dumps({"project_id": "lms", "bullets": bullets}),
                                    "stub")

    transport = _TwoShot()
    client = llm_client.LLMClient(transport, transport, master, policy,
                                  logging.getLogger("test"))
    _bullets, problems = client.write_bullets(project, 3, "python backend", {}, (150, 215))
    assert transport.sent >= 2, "the first generation should have been rejected"
    assert client.calls[0].transport_ok is True and client.calls[0].accepted is False
    assert "Kubernetes" in client.calls[0].detail or "Kafka" in client.calls[0].detail
    assert client.calls[-1].accepted is True
    assert not grounding.errors(problems)


def test_run_reports_assessment_availability(run):
    """A green run must record the assessment as available and stay success."""
    strategy = run["strategy"]
    assert strategy["assessment"]["assessment_available"] is True
    assert strategy["status"] == "success"
    assert "assessment unavailable" not in " ".join(strategy["verification"]["issues"])
    assert "assessment_available=True" in run["log"]
    assert "resume verification succeeded" in run["log"]
    accounting = strategy["provider_accounting"]
    assert accounting["accepted"] == sum(1 for c in strategy["provider_calls"] if c["accepted"])
    assert accounting["transport_ok"] >= accounting["accepted"]
    for call in strategy["provider_calls"]:
        assert call["ok"] == call["accepted"], call


# ---- P. Real-JD ingestion and requirement authority ----------------------

EPIC_JD = ROOT / "jobs" / "epic.txt"


@pytest.fixture(scope="session")
def epic() -> dict:
    master = engine.load_master()
    jd = engine.read_jd(EPIC_JD)
    requirements = engine.extract_jd_requirements(jd.text, master)
    return {"master": master, "jd": jd, "requirements": requirements}


def test_sectionless_jd_extracts_requirements(epic):
    """Epic has no "Requirements:" heading and no bullet markers."""
    text = epic["jd"].text
    assert "Requirements:" not in text
    assert not [l for l in text.splitlines() if re.match(r"^\s*[-*•]", l)]
    assert len(epic["requirements"]) >= 5, epic["requirements"]
    assert engine.scored_requirements(epic["requirements"])


def _epic_requirement(epic, needle: str) -> engine.Requirement:
    """Find one Epic requirement by a distinctive fragment of its own wording."""
    found = [r for r in epic["requirements"] if needle.lower() in r.original_text.lower()]
    assert len(found) == 1, (needle, [r.original_text[:60] for r in epic["requirements"]])
    return found[0]


def test_epic_requirement_kinds_and_ids(epic):
    expected = {"Relocation to the Madison": "logistics",
                "BS/BA or greater": "qualification",
                "history of academic excellence": "qualification",
                "Eligible to work in the United States": "eligibility",
                "COVID-19 vaccination": "condition"}
    for needle, kind in expected.items():
        assert _epic_requirement(epic, needle).kind == kind, needle
    # Immutable, unique, document-ordered ids.
    ids = [r.requirement_id for r in epic["requirements"]]
    assert ids == sorted(ids) and len(set(ids)) == len(ids)
    assert all(re.match(r"^REQ-\d{3}$", i) for i in ids), ids
    # A degree line is a scored qualification; the visa line never is.
    assert _epic_requirement(epic, "BS/BA or greater").scored
    assert not _epic_requirement(epic, "Eligible to work in the United States").scored
    assert not _epic_requirement(epic, "COVID-19 vaccination").scored


def test_epic_marketing_and_benefits_prose_is_ignored(epic):
    joined = " ".join(r.original_text.lower() for r in epic["requirements"])
    for noise in ("mayo clinic", "kayaked", "sabbatical", "greenest city",
                  "equal opportunity", "comprehensive benefits", "restaurant-quality",
                  "fastest growing market"):
        assert noise not in joined, noise


def test_epic_role_prose_is_a_signal_not_a_requirement(epic):
    signals = engine.role_signals(epic["requirements"])
    assert signals, "the JS/TS/C# prose should be recorded as a role signal"
    terms = {t for req in signals for t in req.terms}
    assert {"JS", "TS", "C#"} & terms
    for req in signals:
        assert not req.scored, "a role signal must never reach the fit score"


def test_epic_metadata_resolves_from_prose(epic):
    assert epic["jd"].company_name == "Epic"
    assert epic["jd"].job_title == "Software Developer"
    assert epic["jd"].job_id is None, "no job id is present; none may be invented"


def test_epic_cover_letter_themes_are_nonzero(epic):
    themes = engine.jd_themes(epic["requirements"], 5)
    assert themes, "a substantive posting must yield usable themes"
    assert any(t.terms for t in themes), "at least one theme should name something concrete"


def test_epic_eligibility_is_uncertain_and_consistent(epic):
    status, details, flags = engine.eligibility_assessment(
        epic["requirements"], epic["master"], epic["jd"].text)
    assert status == "uncertain", (status, details)
    assert any("manual review" in d.lower() for d in details), details
    assert flags and any("MANUAL REVIEW" in f for f in flags), flags
    joined = " ".join(details).lower()
    assert "opt" in joined, "the known OPT fact must be surfaced"
    for overclaim in ("not eligible", "ineligible", "definitely eligible"):
        assert overclaim not in joined, overclaim


def test_epic_unsupported_languages_never_enter_skills(epic):
    master, policy = epic["master"], engine.load_policy()
    signals = engine.classify_jd(epic["jd"].text, master.section_order)
    selected = [master.project(p) for p in ("lms", "pintos", "fraud")]
    state = engine.build_skills(master, policy, signals, epic["jd"].text, selected,
                                model_ranked=[])
    offered = {engine.fold_term(c.name) for c in state.candidates}
    for forbidden in ("JS", "JavaScript", "TypeScript", "C#", "CI/CD"):
        assert engine.fold_term(forbidden) not in offered, forbidden
        assert not master.canonical_skill(forbidden), forbidden
    # "C#" must never resolve to the claimable language "C".
    assert master.canonical_skill("C") == "C"
    assert master.canonical_skill("C#") is None
    assert master.canonical_skill("C++") is None


def test_substantive_jd_with_zero_requirements_is_blocked(tmp_path):
    """Extraction failure must stop the run before any provider call."""
    blocked = tmp_path / "opaque.txt"
    blocked.write_text(("Our mission is a journey. " * 60) + "\n", encoding="utf-8")
    jd = engine.read_jd(blocked)
    assert len(jd.text) >= 800
    assert engine.extract_jd_requirements(jd.text, engine.load_master()) == []

    result = run_pipeline.run_one(blocked, mock=True, smoke=True, console=False)
    assert result.status == "needs_review", result.issues
    assert any("zero requirements" in issue for issue in result.issues), result.issues
    log = (result.run_dir / "run.log").read_text()
    assert "ERROR substantive JD produced zero requirements" in log
    assert "[PROVIDER]" not in log, "no provider call may be made"
    strategy = json.loads((result.run_dir / "strategy.json").read_text())
    assert strategy["blocked_before_providers"] is True


def test_assessment_cannot_invent_a_requirement(epic):
    ids = {r.requirement_id for r in epic["requirements"]}
    invented = {"fit_score": 8, "recommendation": "apply", "summary": "Good.",
                "eligibility": {"status": "meets", "details": []},
                "strong_matches": [{"requirement": "Version control and CI practices",
                                    "evidence": "Git is listed", "source": "skills"}],
                "partial_matches": [], "gaps": [],
                "tailoring_quality": {"score": 9, "notes": ["n"]}, "risk_flags": []}
    kinds = {p.kind for p in grounding.errors(
        grounding.validate_assessment(invented, epic["jd"].text, ids))}
    assert "requirement_id" in kinds, kinds

    unknown = {**invented, "strong_matches": [{"requirement_id": "REQ-999",
                                               "evidence": "e", "source": "skills"}]}
    kinds = {p.kind for p in grounding.errors(
        grounding.validate_assessment(unknown, epic["jd"].text, ids))}
    assert "unknown_requirement" in kinds, kinds


def test_one_requirement_cannot_be_strong_and_a_gap(epic):
    ids = {r.requirement_id for r in epic["requirements"]}
    target = sorted(ids)[0]
    data = {"fit_score": 8, "recommendation": "apply", "summary": "Good.",
            "eligibility": {"status": "meets", "details": []},
            "strong_matches": [{"requirement_id": target, "evidence": "e",
                                "source": "skills"}],
            "partial_matches": [],
            "gaps": [{"requirement_id": target, "importance": "required",
                      "status": "unsupported"}],
            "tailoring_quality": {"score": 9, "notes": ["n"]}, "risk_flags": []}
    problems = grounding.errors(grounding.validate_assessment(data, epic["jd"].text, ids))
    assert any(p.kind == "duplicate_requirement" for p in problems), problems


def test_epic_mock_assessment_scores_only_authoritative_requirements(epic):
    import logging

    import llm_client

    master = epic["master"]
    evidence = engine.ResumeEvidence(
        experience_text="Architected relational schemas and SQL queries. Master's in "
                        "Computer Science, University at Buffalo, GPA: 3.85 / 4.0.",
        project_bullets=(("lms", "Implemented Redis and WebSockets."),),
        skills=("Python", "PostgreSQL", "AWS (EC2/RDS)", "WebSockets"))
    transport = llm_client.MockTransport(master, logging.getLogger("test"))
    data = transport._assess({"requirements": epic["requirements"],
                              "skills": list(evidence.skills), "evidence": evidence,
                              "tailoring": {"skills_rendered_lines": 8}, 
                              "jd_text": epic["jd"].text})
    ids = {r.requirement_id for r in epic["requirements"]}
    assert not grounding.errors(
        grounding.validate_assessment(data, epic["jd"].text, ids))
    scored_ids = {r.requirement_id for r in engine.scored_requirements(epic["requirements"])}
    classified = {e["requirement_id"] for bucket in ("strong_matches", "partial_matches",
                                                     "gaps")
                  for e in data[bucket]}
    assert classified <= scored_ids, "only scored requirements may be classified"
    manual_ids = {e["requirement_id"] for e in data["manual_review"]}
    assert manual_ids == {r.requirement_id
                          for r in engine.manual_review_requirements(epic["requirements"])}
    # AWS/PostgreSQL/WebSockets are useful but unrequested: colour, not matches.
    assert data["complementary_strengths"]
    assert data["eligibility"]["status"] == "uncertain"


# ---- Q. Cover-letter fact sources and assessment confidence ---------------

def _master_metric(master) -> grounding.Metric:
    """A distinctive metric the master states exactly.

    Approximation lives in the context ("~90%"), not in `raw`, so an approximate
    metric would not match a plainly worded sentence. Take an exact one.
    """
    for metric in grounding.extract_metrics(master.raw):
        if (metric.unit == "%" and metric.structure == "exact"
                and not metric.approximate and not metric.spelled):
            return metric
    raise AssertionError("the master states no exact percentage to test with")


def _sources(letter: str, epic) -> list[tuple[str, str]]:
    return [(e.source, e.attribution) for e in grounding.classify_letter_metrics(
        letter, epic["master"], jd_text=epic["jd"].text, company="Epic")]


def _metric_problems(letter: str, epic) -> list[grounding.Problem]:
    return [p for p in grounding.validate_cover_letter(
        letter, epic["master"], jd_text=epic["jd"].text, company="Epic")
        if p.kind == "metric"]


def test_candidate_metric_from_the_master_is_supported(epic):
    """A number the candidate owns traces to the master, as it always has."""
    metric = _master_metric(epic["master"])
    letter = f"I benchmarked the extraction pipeline and recorded {metric.raw} accuracy."
    assert _sources(letter, epic) == [("master", "candidate")]
    assert _metric_problems(letter, epic) == []


def test_candidate_metric_found_only_in_the_jd_is_rejected(epic):
    """A JD number must never migrate into candidate ownership."""
    letter = "I built systems serving 325 million patients."
    assert "325 million" in epic["jd"].text
    assert _sources(letter, epic) == [("none", "candidate")]
    problems = _metric_problems(letter, epic)
    assert len(problems) == 1
    assert "own work" in problems[0].message


def test_employer_metric_attributed_to_the_employer_traces_to_the_jd(epic):
    """The posting is authoritative for the employer's own scale."""
    letter = "Epic serves 325 million patients worldwide."
    assert _sources(letter, epic) == [("jd", "employer")]
    assert _metric_problems(letter, epic) == []


def test_employer_metric_absent_from_the_jd_is_rejected(epic):
    letter = "Epic serves 900 million patients worldwide."
    assert "900 million" not in epic["jd"].text
    assert _sources(letter, epic) == [("none", "employer")]
    problems = _metric_problems(letter, epic)
    assert len(problems) == 1
    assert "does not state it" in problems[0].message


def test_ambiguous_ownership_of_a_jd_metric_is_rejected(epic):
    """Unattributed is not a pass: the reader cannot tell whose number it is."""
    letter = "Care for 325 million patients is the goal."
    assert _sources(letter, epic) == [("none", "ambiguous")]
    problems = _metric_problems(letter, epic)
    assert len(problems) == 1
    assert "does not attribute" in problems[0].message


def test_the_live_epic_sentence_is_now_valid(epic):
    """The exact sentence that failed the live run, which quoted the posting."""
    letter = ("I am applying for the Software Developer position at Epic because my "
              "experience building real-time, patient-focused software aligns directly "
              "with Epic's mission to improve care for 325 million patients.")
    assert _sources(letter, epic) == [("jd", "employer")]
    assert _metric_problems(letter, epic) == []


def test_the_letter_prompt_bans_the_em_dash_explicitly():
    import llm_client
    assert "NEVER use an em dash" in llm_client._LETTER_RULES


def test_the_letter_prompt_bans_innovative_and_jd_echoes():
    import llm_client
    rules = llm_client._LETTER_RULES
    assert "innovative" in rules
    assert "NOT exempt" in rules


def test_one_defect_found_by_two_checks_is_reported_once():
    """Deduplication is semantic, and never merges distinct problems."""
    problems = [
        grounding.Problem("style", "error", "contains an em dash"),
        grounding.Problem("format", "error", "cover letter contains an em dash"),
        grounding.Problem("style", "error", "uses discouraged wording 'innovative'"),
        grounding.Problem("style", "error", "uses discouraged wording 'innovative'"),
        grounding.Problem("style", "error", "uses discouraged wording 'seamless'"),
        grounding.Problem("length", "error", "cover letter is 40 words; the allowed range "
                                             "is 140-300"),
    ]
    deduped = grounding.dedupe_problems(problems)
    assert len(deduped) == 4
    assert sum(1 for p in deduped if "em dash" in p.message) == 1
    assert sum(1 for p in deduped if "innovative" in p.message) == 1
    assert sum(1 for p in deduped if "seamless" in p.message) == 1
    assert sum(1 for p in deduped if p.kind == "length") == 1


def test_subjective_requirement_reason_acknowledges_real_evidence(epic):
    """REQ-005 stays manual, but the reason may not deny evidence that exists."""
    subjective = _epic_requirement(epic, "history of academic excellence")
    assert subjective.subjective
    reason = engine.manual_review_reason(subjective, epic["master"])
    assert "positive academic and professional evidence" in reason
    assert "GPA" in reason
    assert "subjective standard" in reason
    assert "not documented" not in reason
    logistics = _epic_requirement(epic, "Relocation to the Madison")
    assert not logistics.subjective
    assert "logistics requirement" in engine.manual_review_reason(logistics, epic["master"])


def test_one_scored_requirement_yields_low_fit_confidence():
    thin = {"strong_matches": [{"requirement_id": "REQ-004"}], "partial_matches": [],
            "gaps": []}
    assert grounding.assessment_breadth(thin, scored=1, signals=2, manual=4) == {
        "scored_requirement_count": 1, "role_signal_count": 2,
        "manual_review_count": 4, "fit_confidence": "low"}
    assert grounding.assessment_breadth(
        thin, scored=3, signals=0, manual=0)["fit_confidence"] == "medium"
    broad = {"strong_matches": [{}] * 8, "partial_matches": [], "gaps": []}
    assert grounding.assessment_breadth(
        broad, scored=10, signals=0, manual=0)["fit_confidence"] == "high"
    shallow = {"strong_matches": [{}] * 2, "partial_matches": [], "gaps": []}
    assert grounding.assessment_breadth(
        shallow, scored=6, signals=0, manual=0)["fit_confidence"] == "normal"


def test_unresolved_required_eligibility_prevents_strong_apply(epic):
    capped = grounding.cap_recommendation(
        {"recommendation": "strong_apply", "eligibility": {"status": "uncertain"}},
        epic["requirements"])
    assert capped is not None
    target, reason = capped
    assert target == "apply"
    assert "REQ-006" in reason
    assert "eligibility unresolved" in reason


def test_unresolved_required_eligibility_still_allows_apply(epic):
    """The cap only forbids strong_apply; it never forces a worse verdict."""
    for recommendation in ("apply", "borderline", "skip"):
        assert grounding.cap_recommendation(
            {"recommendation": recommendation, "eligibility": {"status": "uncertain"}},
            epic["requirements"]) is None
    assert grounding.cap_recommendation(
        {"recommendation": "strong_apply", "eligibility": {"status": "meets"}},
        epic["requirements"]) is None
    without = [r for r in epic["requirements"] if r.kind != "eligibility"]
    assert grounding.cap_recommendation(
        {"recommendation": "strong_apply", "eligibility": {"status": "uncertain"}},
        without) is None


_INVALID_LETTER = (
    "Dear Hiring Manager,\n\n"
    "I am applying for the Backend Engineer position at Northstar Robotics because my "
    "experience building real-time services fits this role, and I want to keep doing that "
    "kind of work — especially on an innovative platform. I have shipped production "
    "services, reviewed teammates' code, and kept systems healthy under sustained load.\n\n"
    "Sincerely,\nJay Niketan Pathare\n")


def _invalid_letter_call(run):
    import llm_client
    transport, client = _stub_client(run, _INVALID_LETTER)
    jd = engine.read_jd(run["dir"] / "job_description.txt")
    letter, problems = client.cover_letter(
        jd, run["signals"], ["Built things"], [run["master"].project("lms")],
        themes=engine.jd_themes(
            engine.extract_jd_requirements(jd.text, run["master"]), 4))
    return client, letter, problems


def test_cover_letter_validation_failure_is_not_an_accepted_artifact(run):
    client, letter, problems = _invalid_letter_call(run)
    messages = [p.message for p in grounding.errors(problems)]
    assert sum(1 for m in messages if "em dash" in m) == 1
    assert sum(1 for m in messages if "innovative" in m) == 1
    record = client.calls[-1]
    assert record.transport_ok is True
    assert record.accepted is False
    assert record.as_dict()["ok"] is False
    assert "cover-letter validation failed" in record.detail
    # The provider's raw output is preserved, never discarded.
    assert letter.startswith("Dear Hiring Manager,")


def test_provider_accounting_counts_the_rejected_cover_letter(run):
    """Bounded repair means one invalid letter costs LETTER_ATTEMPTS generations."""
    import llm_client

    client, _, _ = _invalid_letter_call(run)
    calls = client.calls
    assert {"generations": len(calls),
            "transport_ok": sum(1 for c in calls if c.transport_ok),
            "accepted": sum(1 for c in calls if c.accepted),
            "rejected": sum(1 for c in calls if c.transport_ok and not c.accepted)} == {
        "generations": llm_client.LETTER_ATTEMPTS,
        "transport_ok": llm_client.LETTER_ATTEMPTS,
        "accepted": 0, "rejected": llm_client.LETTER_ATTEMPTS}


def test_capped_recommendation_summary_never_says_strong_apply(epic):
    """A capped verdict may not leave "strong apply" standing in the prose."""
    assessment = {"recommendation": "strong_apply",
                  "eligibility": {"status": "uncertain"},
                  "summary": "1 of 1 scored requirement(s) are directly evidenced and 0 are "
                             "unsupported, so this posting is a strong apply.",
                  "strong_matches": [{"requirement_id": "REQ-004"}],
                  "partial_matches": [], "gaps": []}
    assessment.update(grounding.assessment_breadth(assessment, scored=1, signals=2, manual=4))
    capped = grounding.cap_recommendation(assessment, epic["requirements"])
    assert capped is not None
    target, reason = capped
    assessment["recommendation"] = target
    assessment["model_summary"] = assessment["summary"]
    assessment["summary"] = grounding.deterministic_summary(assessment)

    assert assessment["recommendation"] == "apply"
    lowered = assessment["summary"].lower()
    assert "strong apply" not in lowered
    assert "strong_apply" not in lowered
    assert "Final recommendation: apply." in assessment["summary"]
    assert "evidence breadth is low" in lowered
    assert _terminators(assessment["summary"]) <= 3
    # the provider's prose is preserved for audit, not discarded
    assert assessment["model_summary"].endswith("a strong apply.")


def test_uncapped_assessment_is_not_capped_but_still_summarized(epic):
    """No cap fires, yet the user-facing summary is still deterministic."""
    assessment = {"recommendation": "apply", "eligibility": {"status": "meets"},
                  "summary": "Six of eight scored requirements are evidenced, so this is "
                             "worth applying to.",
                  "strong_matches": [], "partial_matches": [], "gaps": []}
    assert grounding.cap_recommendation(assessment, epic["requirements"]) is None
    summary = grounding.deterministic_summary(assessment)
    assert "Final recommendation: apply." in summary
    assert grounding.denies_evidence(summary) is None


# ---- R. No-argument batch discovery ---------------------------------------

def _tracker(tmp_path, index: dict | None = None):
    """A Tracker over throwaway paths, so production tracking is never read."""
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(index or {}))
    return run_pipeline.Tracker(tmp_path / "processed.csv", index_path, enabled=True)


def _jd_file(path: Path, body: str = "") -> Path:
    path.write_text("Company: Acme Corp\nJob Title: Backend Engineer\n\n"
                    "Requirements:\n- Python\n- PostgreSQL\n" + body, encoding="utf-8")
    return path


def test_no_arguments_discovers_the_real_jobs_folder(monkeypatch):
    """Bare `run_pipeline.py` batches jobs/, not a single JD and not an error."""
    seen = {}
    monkeypatch.setattr(run_pipeline, "run_batch",
                        lambda folder, **kw: seen.update(folder=folder, **kw) or 0)
    assert run_pipeline.main([]) == 0
    assert seen["folder"] == engine.JOBS_DIR
    assert seen["mock"] is False and seen["smoke"] is False


def test_discovery_is_not_recursive(tmp_path):
    """jobs/trial holds fixtures; a nested folder must never be picked up."""
    _jd_file(tmp_path / "top.txt")
    nested = tmp_path / "trial"
    nested.mkdir()
    _jd_file(nested / "fixture.txt")
    pending, skipped = run_pipeline.discover_jobs(tmp_path, _tracker(tmp_path))
    assert [p.name for p in pending] == ["top.txt"]
    assert skipped == []
    # the real tree: trial fixtures are invisible to production discovery
    real, _ = run_pipeline.discover_jobs(engine.JOBS_DIR, _tracker(tmp_path))
    assert all(p.parent == engine.JOBS_DIR for p in real)
    assert "backend_swe.txt" not in [p.name for p in real]


def test_already_tracked_fingerprint_is_skipped(tmp_path):
    job = _jd_file(tmp_path / "done.txt")
    fingerprint = engine.read_jd(job).fingerprint
    tracker = _tracker(tmp_path, {fingerprint: {"run_folder": "Acme_Corp_2026-09-13"}})
    pending, skipped = run_pipeline.discover_jobs(tmp_path, tracker)
    assert pending == []
    assert [p.name for p in skipped] == ["done.txt"]


def test_renamed_duplicate_jd_is_still_skipped(tmp_path):
    """Identity is the fingerprint, so a rename alone never re-applies."""
    original = _jd_file(tmp_path / "original.txt")
    fingerprint = engine.read_jd(original).fingerprint
    original.rename(tmp_path / "renamed_copy.txt")
    tracker = _tracker(tmp_path, {fingerprint: {"run_folder": "Acme_Corp_2026-09-13"}})
    pending, skipped = run_pipeline.discover_jobs(tmp_path, tracker)
    assert pending == []
    assert [p.name for p in skipped] == ["renamed_copy.txt"]


def test_edited_jd_gets_a_new_fingerprint_and_processes(tmp_path):
    job = _jd_file(tmp_path / "posting.txt")
    old = engine.read_jd(job).fingerprint
    _jd_file(job, body="- Kubernetes and a new responsibility line\n")
    assert engine.read_jd(job).fingerprint != old
    tracker = _tracker(tmp_path, {old: {"run_folder": "Acme_Corp_2026-09-13"}})
    pending, skipped = run_pipeline.discover_jobs(tmp_path, tracker)
    assert [p.name for p in pending] == ["posting.txt"]
    assert skipped == []


def test_needs_review_is_never_marked_processed(tmp_path):
    """An unsuccessful run stays pending, so the next invocation retries it."""
    import logging

    job = _jd_file(tmp_path / "retry.txt")
    jd = engine.read_jd(job)
    tracker = _tracker(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    log = run_pipeline.StageLog(logging.getLogger("test-batch"), "TEST")
    for status in ("needs_review", "failed"):
        assert tracker.record(jd, run_dir, status, log) is False
    assert json.loads((tmp_path / "index.json").read_text()) == {}
    pending, skipped = run_pipeline.discover_jobs(tmp_path, tracker)
    assert [p.name for p in pending] == ["retry.txt"]
    assert skipped == []


def test_batch_with_nothing_pending_exits_cleanly(tmp_path, capsys, monkeypatch):
    job = _jd_file(tmp_path / "done.txt")
    fingerprint = engine.read_jd(job).fingerprint
    (tmp_path / "index.json").write_text(json.dumps(
        {fingerprint: {"run_folder": "Acme_Corp_2026-09-13"}}))
    monkeypatch.setattr(run_pipeline, "run_one",
                        lambda *a, **k: pytest.fail("run_one must not be called"))
    code = run_pipeline.run_batch(tmp_path, csv_path=tmp_path / "processed.csv",
                                  index_path=tmp_path / "index.json")
    out = capsys.readouterr().out
    assert code == 0
    assert "[BATCH] found 1 JD file(s)" in out
    assert "[BATCH] already processed: 1" in out
    assert "[BATCH] pending: 0" in out
    assert "[BATCH] skip already processed: done.txt" in out
    assert "[BATCH] no unprocessed JDs found" in out


def test_explicit_single_job_invocation_is_unchanged(monkeypatch):
    """An explicit path still runs exactly one job, never the batch engine."""
    calls = {}
    monkeypatch.setattr(run_pipeline, "run_batch",
                        lambda *a, **k: pytest.fail("batch must not run for one JD"))
    monkeypatch.setattr(run_pipeline, "run_one",
                        lambda path, **kw: calls.update(path=path, **kw)
                        or run_pipeline.RunResult(jd_path=path, status="success"))
    assert run_pipeline.main([str(EPIC_JD), "--mock"]) == 0
    assert calls["path"] == EPIC_JD
    assert calls["mock"] is True


# ---- S. BAE real-JD ingestion and honest degree classification ------------

BAE_JD = ROOT / "jobs" / "bae.txt"


@pytest.fixture(scope="session")
def bae() -> dict:
    master = engine.load_master()
    template = engine.load_template()
    jd = engine.read_jd(BAE_JD)
    requirements = engine.extract_jd_requirements(jd.text, master)
    education = engine.latex_to_plain(" ".join(template.blocks["Education"].split()))
    evidence = engine.ResumeEvidence(
        experience_text=education + " Parallelized the pipeline via a worker pool. "
                        "Architected relational schemas and SQL queries.",
        project_bullets=(
            ("lms", "Reduced p95 REST latency by approximately 26% through load testing "
                    "with Locust and asyncio up to 100 concurrent users."),
            ("pintos", "Designed process execution using memory validation to pass 100% "
                       "of 80 concurrency, memory-fault, and edge-case tests."),
        ),
        skills=("Python", "SQL", "OOP", "Linux", "Docker", "Operating Systems",
                "Machine Learning", "Parallel Processing"))
    return {"master": master, "jd": jd, "requirements": requirements,
            "evidence": evidence, "education": education}


def _bae_requirement(bae, needle: str) -> engine.Requirement:
    found = [r for r in bae["requirements"] if needle.lower() in r.original_text.lower()]
    assert len(found) == 1, (needle, [r.original_text[:60] for r in bae["requirements"]])
    return found[0]


def _bae_verdict(bae, needle: str):
    return engine.classify_requirement(
        _bae_requirement(bae, needle), bae["master"], bae["evidence"])


def test_bae_company_resolves_from_prose(bae):
    """"The BAE Systems GXP Software Team ... is seeking ..." names the company."""
    assert bae["jd"].company_name == "BAE Systems"


def test_bae_title_is_normalized_to_singular(bae):
    assert bae["jd"].job_title == "Entry Level Software Engineer"


def test_job_board_chrome_is_never_a_requirement(bae):
    texts = [r.original_text.lower() for r in bae["requirements"]]
    assert not [t for t in texts if "primary posting" in t or t.strip() == "primary"]
    assert not [t for t in texts if t.strip() in ("additional posting", "description")]


def test_benefits_and_union_text_is_never_a_requirement(bae):
    """The CBA/SCA sentence is boilerplate, not a capability or a signal."""
    for text in (r.original_text.lower() for r in bae["requirements"]):
        assert "collective bargaining" not in text
        assert "union employees" not in text
        assert "service contract act" not in text
        assert "401(k)" not in text


def test_concatenated_requirement_line_is_split(bae):
    """Four clauses were run together on one physical line in the source."""
    for needle in ("Python, Java, C++, or JavaScript",
                   "object oriented programming",
                   "analytical, problem solving",
                   "both Linux and Windows"):
        assert _bae_requirement(bae, needle) is not None
    langs = _bae_requirement(bae, "Python, Java, C++, or JavaScript")
    assert "Understanding of" not in langs.original_text
    assert "Familiarity with" not in langs.original_text


def test_language_list_is_an_or_group_not_a_checklist(bae):
    langs = _bae_requirement(bae, "Python, Java, C++, or JavaScript")
    assert engine.any_of_list(langs.original_text)
    groups = engine.requirement_groups(langs)
    assert len(groups) == 1, groups
    assert "Python" in groups[0]


def test_python_alone_satisfies_the_language_alternative(bae):
    match = _bae_verdict(bae, "Python, Java, C++, or JavaScript")
    assert match.verdict == "strong_match"
    assert "Python" in match.evidence


def test_unsupported_languages_are_never_claimed(bae):
    match = _bae_verdict(bae, "Python, Java, C++, or JavaScript")
    for absent in ("Java (", "C++ (", "JavaScript ("):
        assert absent not in match.evidence
    master = bae["master"]
    for term in ("Java", "C++", "JavaScript"):
        assert master.canonical_skill(term) is None
        assert master.supported_anywhere(term) is False


def test_linux_is_supported_but_windows_blocks_the_and_group(bae):
    """"both Linux and Windows" is AND, so Linux alone is only partial."""
    requirement = _bae_requirement(bae, "both Linux and Windows")
    assert not engine.any_of_list(requirement.original_text)
    assert "Windows" in requirement.terms
    match = _bae_verdict(bae, "both Linux and Windows")
    assert match.verdict == "partial_match"
    assert "Linux" in match.evidence
    assert "Windows" in match.limitation
    assert bae["master"].supported_anywhere("Windows") is False


def test_coursework_stays_one_or_more_alternatives(bae):
    coursework = _bae_requirement(bae, "Course work one or more")
    assert engine.any_of_list(coursework.original_text)
    groups = engine.requirement_groups(coursework)
    assert len(groups) == 1, groups
    match = _bae_verdict(bae, "Course work one or more")
    assert match.verdict == "strong_match"


def test_docker_never_becomes_kubernetes_for_coursework(bae):
    coursework = _bae_requirement(bae, "Course work one or more")
    assert "Kubernetes" in coursework.terms
    match = _bae_verdict(bae, "Course work one or more")
    assert "Kubernetes" not in match.evidence
    assert bae["master"].supported_anywhere("Kubernetes") is False


def test_education_evidence_exposes_both_degrees(bae):
    education = bae["education"]
    assert "Master's in Computer Science" in education
    assert "Bachelor of Engineering, Information Technology" in education
    assert "University at Buffalo" in education
    assert "Mumbai University" in education


def test_bae_degree_is_never_reported_as_absent(bae):
    match = _bae_verdict(bae, "Bachelor's of Science degree")
    assert match.verdict in ("strong_match", "partial_match")
    blob = f"{match.evidence} {match.limitation}".lower()
    for denial in ("does not list", "no bachelor", "not listed", "missing",
                   "no supporting evidence"):
        assert denial not in blob, blob


def test_bae_degree_field_mismatch_is_stated_accurately(bae):
    match = _bae_verdict(bae, "Bachelor's of Science degree")
    assert match.verdict == "partial_match"
    assert "Information Technology" in match.evidence
    assert "Information Technology" in match.limitation
    assert "Computer Science" in match.limitation
    assert match.source == "education"


def test_letter_rules_state_the_education_record_truthfully():
    import llm_client
    rules = llm_client._LETTER_RULES
    assert "IN PROGRESS" in rules
    assert 'Never write "I hold a Master\'s"' in rules
    assert "undergraduate foundation" in rules
    assert "B.E. in Information Technology" in rules
    assert 'do not claim "profiling"' in rules


def test_substantive_posting_without_a_title_cannot_track_silently():
    """Production metadata that never resolved must not be recorded as success."""
    source = Path(run_pipeline.__file__).read_text()
    assert "job title could not be resolved from a substantive posting" in source
    # the guard runs only for production runs, before status is decided
    guard = source.split("job title could not be resolved")[0]
    assert "if not mock and not smoke and not jd.job_title" in guard


# ---- T. Immutable Python verdicts; the model may only explain them ---------

def _stub_model_reply() -> dict:
    """What Groq actually returned on the live BAE run, verdicts included."""
    return {
        "fit_score": 5.8, "recommendation": "borderline",
        "summary": "Strong Python and OOP skills, but lacks documented evidence of a "
                   "required bachelor's degree and specific coursework.",
        "eligibility": {"status": "uncertain",
                        "details": ["Missing documented bachelor's degree (REQ-006)"]},
        "strong_matches": [{"requirement_id": "REQ-007",
                            "evidence": "Python development and JavaScript (React) listed "
                                        "in Technical Skills.", "source": "skills"}],
        "partial_matches": [{"requirement_id": "REQ-010",
                             "evidence": "Linux listed; Windows not mentioned.",
                             "limitation": "Missing Windows OS experience."}],
        "gaps": [{"requirement_id": "REQ-006", "importance": "required",
                  "status": "unsupported", "evidence": None,
                  "detail": "Resume shows only a Master's degree; no bachelor's degree "
                            "evidence provided."},
                 {"requirement_id": "REQ-013", "importance": "required",
                  "status": "unsupported", "evidence": None,
                  "detail": "No coursework list matching the required subjects."}],
        "manual_review": [],
        "complementary_strengths": ["Sensor data ingestion"],
        "tailoring_quality": {"score": 8.7, "notes": ["aligns well"]},
        "risk_flags": ["Missing documented bachelor's degree (REQ-006)",
                       "Missing required coursework evidence (REQ-013)"],
    }


@pytest.fixture(scope="session")
def bae_verdicts(bae) -> dict:
    return engine.deterministic_assessment(
        bae["requirements"], bae["master"], bae["evidence"])


def test_python_owns_the_verdicts_not_the_model(bae, bae_verdicts):
    """A model verdict is discarded: Python's classification is authoritative."""
    data = _stub_model_reply()
    table = engine.requirement_table(bae["requirements"])
    notes = grounding.enforce_verdicts(data, bae_verdicts, table)
    index = {e["requirement_id"]: b for b in
             ("strong_matches", "partial_matches", "gaps", "manual_review")
             for e in data[b]}
    assert index == engine.verdict_index(bae_verdicts)
    assert data["verdict_source"] == "python_deterministic"
    assert any("model said gaps" in n for n in notes)


def test_the_degree_is_never_a_gap_in_the_live_path(bae, bae_verdicts):
    """The exact live defect: REQ-006 reported as a missing bachelor's degree."""
    data = _stub_model_reply()
    grounding.enforce_verdicts(
        data, bae_verdicts, engine.requirement_table(bae["requirements"]))
    assert "REQ-006" not in [g["requirement_id"] for g in data["gaps"]]
    entry = next(e for e in data["partial_matches"] if e["requirement_id"] == "REQ-006")
    assert "Information Technology" in entry["evidence"]
    assert "Information Technology" in entry["limitation"]
    blob = f"{entry['evidence']} {entry['limitation']} {entry.get('explanation', '')}".lower()
    for denial in ("no bachelor", "does not list", "not documented", "missing"):
        assert denial not in blob, blob


def test_coursework_one_or_more_is_not_a_gap(bae, bae_verdicts):
    data = _stub_model_reply()
    grounding.enforce_verdicts(
        data, bae_verdicts, engine.requirement_table(bae["requirements"]))
    assert "REQ-013" not in [g["requirement_id"] for g in data["gaps"]]
    assert "REQ-013" in [s["requirement_id"] for s in data["strong_matches"]]


def test_no_model_prose_survives_on_a_requirement_entry(bae, bae_verdicts):
    """Python's evidence is the complete authority on every entry."""
    data = _stub_model_reply()
    grounding.enforce_verdicts(
        data, bae_verdicts, engine.requirement_table(bae["requirements"]))
    allowed = {"requirement_id", "requirement", "kind", "evidence", "source",
               "limitation", "detail", "reason", "importance", "status"}
    for bucket in ("strong_matches", "partial_matches", "gaps", "manual_review"):
        for entry in data[bucket]:
            assert "explanation" not in entry
            assert set(entry) <= allowed, (bucket, sorted(set(entry) - allowed))
    assert grounding.denies_evidence("no bachelor's degree evidence provided")


def test_eligibility_is_not_contaminated_by_capability_gaps(bae, bae_verdicts):
    """The BAE posting states no authorization requirement, so it cannot be uncertain."""
    data = _stub_model_reply()
    grounding.enforce_verdicts(
        data, bae_verdicts, engine.requirement_table(bae["requirements"]))
    assert data["eligibility"]["status"] == "not_applicable"
    assert data["eligibility"]["details"] == []


def test_the_user_facing_summary_is_always_deterministic(bae, bae_verdicts):
    """Never denial-detection: the summary is regenerated unconditionally."""
    data = _stub_model_reply()
    grounding.finalize_assessment(
        data, verdicts=bae_verdicts,
        table=engine.requirement_table(bae["requirements"]),
        requirements=bae["requirements"], scored=9, signals=5, manual=0)
    assert data["model_summary"].startswith("Strong Python and OOP skills")
    assert grounding.denies_evidence(data["summary"]) is None
    assert "directly evidenced" in data["summary"]
    assert "No separate work-authorization" in data["summary"]
    assert "Final recommendation:" in data["summary"]


def test_unsupported_technology_is_never_the_deterministic_evidence(bae, bae_verdicts):
    """Python's own evidence for the OR-group cites Python, never JavaScript."""
    entry = next(e for e in bae_verdicts["strong_matches"]
                 if e["requirement_id"] == "REQ-007")
    assert "Python" in entry["evidence"]
    for absent in ("JavaScript", "Java ", "C++"):
        assert absent not in entry["evidence"]


def test_mock_and_live_paths_share_one_verdict_authority(bae, bae_verdicts):
    """The mock no longer has its own classifier: both consume the same function."""
    again = engine.deterministic_assessment(
        bae["requirements"], bae["master"], bae["evidence"])
    assert engine.verdict_index(again) == engine.verdict_index(bae_verdicts)


def test_eligibility_is_not_applicable_without_an_eligibility_requirement(bae, epic):
    """"meets" would imply a requirement was cleared; BAE states none."""
    master = bae["master"]
    bae_status, bae_details, _ = engine.eligibility_assessment(
        bae["requirements"], master, bae["jd"].text)
    assert bae_status == "not_applicable"
    assert bae_details == []
    assert not [r for r in bae["requirements"] if r.kind == "eligibility"]
    # Epic does state one, so it keeps the conservative verdict
    epic_status, epic_details, epic_flags = engine.eligibility_assessment(
        epic["requirements"], master, epic["jd"].text)
    assert epic_status == "uncertain"
    assert epic_details and epic_flags
    assert "not_applicable" in grounding.ELIGIBILITY_STATUS


@pytest.mark.parametrize("raw", ["Not Applicable", "not-applicable", "N/A", "none"])
def test_not_applicable_eligibility_aliases_are_canonicalized(raw):
    out, events = grounding.canonicalize_assessment({"eligibility": {"status": raw}})
    assert out["eligibility"]["status"] == "not_applicable"
    if raw != "not_applicable":
        assert events


def test_not_applicable_never_caps_the_recommendation(epic):
    """Nothing to resolve means nothing to cap."""
    assert grounding.cap_recommendation(
        {"recommendation": "strong_apply", "eligibility": {"status": "not_applicable"}},
        epic["requirements"]) is None


def test_a_lying_provider_cannot_corrupt_assess(run, bae):
    """The real assess() boundary, not just the helper.

    A provider returns the exact bad BAE behaviour: the degree and coursework as
    unsupported gaps, JavaScript cited as evidence, eligibility uncertain, and a
    summary denying the degree. None of it may reach the return value.
    """
    import llm_client

    payload = json.dumps(_stub_model_reply())
    transport, client = _stub_client(run, payload)
    requirements = bae["requirements"]
    data = client.assess(
        bae["jd"], run["signals"], [run["master"].project("lms")],
        ["Python", "SQL", "OOP", "Linux", "Docker", "Docker Compose",
         "Operating Systems", "Machine Learning"],
        experience_plain=["Led delivery of platform modules end to end: gathered "
                          "requirements in client meetings.",
                          "Parallelized the pipeline via a worker pool."],
        project_bullets={"lms": ["Optimized REST behavior with Locust and asyncio, "
                                 "reducing p95 REST latency by approximately 26%."],
                         "pintos": ["Integrated synchronization primitives and byte-wise "
                                    "memory validation, passing 100% of 80 concurrency "
                                    "and memory-fault tests."]},
        requirements=requirements,
        extra_experience=[bae["education"]])

    buckets = {e["requirement_id"]: b for b in
               ("strong_matches", "partial_matches", "gaps", "manual_review")
               for e in data[b]}

    # the two fabricated gaps are gone
    assert data["gaps"] == []
    assert buckets["REQ-006"] == "partial_matches"
    assert buckets["REQ-013"] == "strong_matches"

    degree = next(e for e in data["partial_matches"] if e["requirement_id"] == "REQ-006")
    assert "Bachelor of Engineering in Information Technology" in degree["evidence"]
    assert "Information Technology" in degree["limitation"]
    assert grounding.denies_evidence(f"{degree['evidence']} {degree['limitation']}") is None

    # the OR-group is satisfied by Python, and JavaScript is never cited
    language = next(e for e in data["strong_matches"] if e["requirement_id"] == "REQ-007")
    assert "Python" in language["evidence"]
    assert "JavaScript" not in language["evidence"]
    assert "React" not in language["evidence"]

    assert data["eligibility"]["status"] == "not_applicable"
    assert data["verdict_source"] == "python_deterministic"
    assert data["model_summary"].startswith("Strong Python and OOP skills")
    assert grounding.denies_evidence(data["summary"]) is None
    assert "Final recommendation:" in data["summary"]
    assert client.calls[-1].accepted is True


# ---- U. Deterministic summary and the final integrity gate ----------------

def _finalized(jd_path: Path, master, *, recommendation="strong_apply") -> tuple[dict, dict, dict, list]:
    """A finalized assessment for one posting, as assess() would produce it."""
    jd = engine.read_jd(jd_path)
    requirements = engine.extract_jd_requirements(jd.text, master)
    evidence = engine.ResumeEvidence(
        experience_text=engine.latex_to_plain(
            " ".join(engine.load_template().blocks["Education"].split()))
        + " Led delivery of platform modules end to end: gathered requirements in "
          "client meetings. Parallelized the pipeline via a worker pool.",
        project_bullets=(("lms", "Optimized REST behavior with Locust and asyncio."),
                         ("pintos", "Integrated synchronization primitives, passing 100% "
                                    "of 80 concurrency and memory-fault tests.")),
        skills=("Python", "SQL", "OOP", "Linux", "Docker", "Docker Compose",
                "Operating Systems", "Machine Learning"))
    verdicts = engine.deterministic_assessment(requirements, master, evidence, jd.text)
    table = engine.requirement_table(requirements)
    data = {"fit_score": 8.0, "recommendation": recommendation,
            "summary": "Provider prose that will be replaced.",
            "eligibility": {"status": "meets", "details": []},
            "strong_matches": [], "partial_matches": [], "gaps": [], "manual_review": [],
            "complementary_strengths": [], "risk_flags": [],
            "tailoring_quality": {"score": 9.0, "notes": ["n"]}}
    grounding.finalize_assessment(
        data, verdicts=verdicts, table=table, requirements=requirements,
        scored=len(engine.scored_requirements(requirements)),
        signals=len(engine.role_signals(requirements)),
        manual=len(engine.manual_review_requirements(requirements)))
    return data, verdicts, table, requirements


def _terminators(text: str) -> int:
    return len(re.findall(r"[.!?]", text))


def test_bae_summary_is_three_sentences_and_states_not_applicable(bae):
    data, *_ = _finalized(BAE_JD, bae["master"], recommendation="apply")
    assert _terminators(data["summary"]) <= 3
    assert "Final recommendation: apply." in data["summary"]
    assert "No separate work-authorization" in data["summary"]
    assert data["eligibility"]["status"] == "not_applicable"
    assert grounding.denies_evidence(data["summary"]) is None


def test_epic_summary_never_says_strong_apply_after_the_cap(epic):
    data, *_ = _finalized(EPIC_JD, epic["master"], recommendation="strong_apply")
    assert _terminators(data["summary"]) <= 3
    assert data["recommendation"] == "apply"
    assert data["recommendation_capped"]["from"] == "strong_apply"
    lowered = data["summary"].lower()
    assert "strong apply" not in lowered and "strong_apply" not in lowered
    assert "Final recommendation: apply." in data["summary"]
    assert "unresolved" in lowered
    assert "manual review" in lowered


def test_newgrad_summary_reports_a_satisfied_graduation_window():
    master = engine.load_master()
    data, *_ = _finalized(ROOT / "jobs_synthetic" / "newgrad_swe.txt", master,
                          recommendation="strong_apply")
    assert _terminators(data["summary"]) <= 3
    assert data["eligibility"]["status"] == "meets"
    assert "eligibility requirement is satisfied" in data["summary"]
    assert f"Final recommendation: {data['recommendation']}." in data["summary"]


def test_final_validator_accepts_a_correct_finalized_assessment(bae):
    data, verdicts, table, requirements = _finalized(BAE_JD, bae["master"],
                                                     recommendation="apply")
    problems = grounding.errors(grounding.validate_final_assessment(
        data, verdicts=verdicts, table=table, requirements=requirements))
    assert problems == [], [p.message for p in problems]


@pytest.mark.parametrize("kind, corrupt", [
    ("final_verdict_mismatch", "move"),
    ("final_missing_requirement", "drop"),
    ("final_duplicate_requirement", "duplicate"),
    ("final_unknown_requirement", "unknown"),
    ("final_entry_fields", "explanation"),
    ("final_eligibility", "eligibility"),
    ("final_summary", "stale_summary"),
    ("final_summary", "long_summary"),
])
def test_final_validator_rejects_corrupted_assessments(bae, kind, corrupt):
    data, verdicts, table, requirements = _finalized(BAE_JD, bae["master"],
                                                     recommendation="apply")
    if corrupt == "move":
        moved = data["partial_matches"].pop()
        data["strong_matches"].append(moved)
    elif corrupt == "drop":
        data["strong_matches"].pop()
    elif corrupt == "duplicate":
        data["partial_matches"].append(dict(data["strong_matches"][0]))
    elif corrupt == "unknown":
        data["strong_matches"].append({"requirement_id": "REQ-999", "evidence": "x",
                                       "source": "skills"})
    elif corrupt == "explanation":
        data["strong_matches"][0]["explanation"] = "JavaScript experience in React."
    elif corrupt == "eligibility":
        data["eligibility"] = {"status": "meets", "details": []}
    elif corrupt == "stale_summary":
        data["summary"] = ("4 of 9 are evidenced. This is a strong apply. "
                           "Final recommendation: apply.")
    elif corrupt == "long_summary":
        data["summary"] = "One. Two. Three. Four. Final recommendation: apply."

    problems = grounding.validate_final_assessment(
        data, verdicts=verdicts, table=table, requirements=requirements)
    kinds = {p.kind for p in grounding.errors(problems)}
    assert kind in kinds, (corrupt, sorted(kinds))


# ---- V. Role-title matching is typography-insensitive ---------------------

_ROLE_TITLE = "Entry Level Software Engineer"


def _names_the_role(sentence: str, title: str = _ROLE_TITLE) -> bool:
    """True when the role-name check accepts `sentence`."""
    problems = grounding.validate_letter_quality(
        sentence, company=None, job_title=title)
    return not any("never names the role" in p.message for p in problems)


@pytest.mark.parametrize("rendering", [
    "Entry Level Software Engineer",
    "entry level software engineer",
    "Entry-Level Software Engineer",
    "entry-level Software Engineer",
    "entry‑level Software Engineer",          # U+2011 non-breaking hyphen
    "Entry–Level Software Engineer",          # U+2013 en dash
    "Entry Level   Software Engineer",             # repeated whitespace
    "Entry Level Software Engineer",          # U+00A0 non-breaking space
    "Entry‐Level Software Engineer",          # U+2010 hyphen
    "Entry—Level Software Engineer",          # U+2014 em dash (matching only)
])
def test_role_title_accepts_every_typographic_rendering(rendering):
    assert _names_the_role(f"I am applying for the {rendering} position at BAE Systems.")


@pytest.mark.parametrize("wrong", [
    "Software Engineer",
    "Entry Level Engineer",
    "Software Engineering",
    "Entry Level Software Developer",
    "entry level role",
    "software engineering position",
    "Engineer at BAE Systems",
])
def test_role_title_still_requires_the_whole_phrase(wrong):
    assert not _names_the_role(f"I am applying for the {wrong} position at BAE Systems.")


def test_the_exact_live_bae_sentence_names_the_role():
    """The sentence that failed the live run, verbatim (U+2011 included)."""
    sentence = ("I am applying for the entry‑level Software Engineer position at "
                "BAE Systems.")
    assert "‑" in sentence
    assert _names_the_role(sentence)


def test_title_normalization_does_not_relax_style_checks():
    """An em dash may satisfy the role name and still fail the style policy."""
    letter = ("Dear Hiring Manager,\n\nI am applying for the Entry—Level Software "
              "Engineer position at BAE Systems.\n\nSincerely,\nJay Niketan Pathare\n")
    problems = grounding.validate_letter_quality(
        letter, company="BAE Systems", job_title=_ROLE_TITLE)
    assert not any("never names the role" in p.message for p in problems)
    assert any("em dash" in p.message for p in problems)
    # and the letter itself is never rewritten by the matcher
    assert "—" in letter


def test_normalize_title_phrase_is_deterministic_and_word_preserving():
    normalize = grounding.normalize_title_phrase
    assert normalize("Entry Level Software Engineer") == "entry level software engineer"
    assert normalize("the entry‑level Software Engineer position") == (
        "the entry level software engineer position")
    assert normalize("Software Engineer, New Grad 2027") == (
        "software engineer new grad 2027")
    # no word is invented or dropped
    assert normalize("UI/UX Designer").split() == ["ui/ux", "designer"]


def test_a_preserved_live_bae_letter_satisfies_the_role_title_check():
    """The U+2011 regression, checked against a real shipped artifact.

    Only the role-title contract is asserted. That folder is historical output
    that a later run may have replaced, and any other validation failure in it
    is real and must stay visible rather than be masked here.
    """
    letter_path = (ROOT / "output" / "BAE_Systems_Entry_Level_Software_Engineer_2026-09-13"
                   / "cover_letter.txt")
    if not letter_path.exists():
        pytest.skip("the live BAE run folder is not present")
    letter = letter_path.read_text(encoding="utf-8")
    jd = engine.read_jd(ROOT / "jobs" / "bae.txt")
    assert "\u2011" in letter, "this artifact no longer exercises the U+2011 case"
    problems = grounding.validate_letter_quality(
        letter, company=jd.company_name, job_title=jd.job_title)
    assert not any("never names the role" in p.message for p in problems), \
        [p.message for p in problems]


# ---- W. Bounded provider repair, immutable runs, tracking idempotence ----

class _SequenceTransport:
    """Replays a different payload per call, recording every request."""

    name = "stub"

    def __init__(self, payloads: list[str], provider: str = "stub"):
        self.payloads = list(payloads)
        self.provider = provider
        self.requests: list = []

    def generate(self, request):
        import llm_client
        self.requests.append(request)
        payload = self.payloads[min(len(self.requests) - 1, len(self.payloads) - 1)]
        return llm_client.Reply(payload, self.provider)


def _sequence_client(run, payloads: list[str]):
    import logging

    import llm_client
    transport = _SequenceTransport(payloads)
    return transport, llm_client.LLMClient(transport, transport, run["master"],
                                           run["policy"], logging.getLogger("test"))


def _letter_body(transactions: str) -> str:
    """A letter that is valid apart from one transaction-count rendering."""
    return (
        "Dear Hiring Manager,\n\n"
        "I am applying for the Entry Level Software Engineer role at BAE Systems. I am "
        "completing an M.S. in Computer Science at the University at Buffalo after "
        "earning a B.E. in Information Technology.\n\n"
        f"I designed a graph neural network pipeline in PyTorch Geometric that processed "
        f"{transactions} to detect fraud, and I built a fuzzy-matching combinator engine "
        "that merged asynchronous JSON outputs and raised extraction accuracy from 57.03% "
        "to 90.81% across 47 law journals covering 28,167+ pages. I parallelized that "
        "pipeline through a worker pool so each worker pulls the next journal from a "
        "shared queue, and I added a confidence system that flags low-confidence fields "
        "for human review.\n\n"
        "In my professional work I built and scaled healthcare modules supporting 10k+ "
        "records, and I designed a real-time WebSocket and Redis pipeline that fed sensor "
        "data into PostgreSQL with ~2s live monitoring. I also architected the relational "
        "schemas and SQL queries behind those services, and I containerized the "
        "deployment with Docker so local and staging environments stayed consistent.\n\n"
        "In an academic project I implemented the user-programs layer of Pintos in C, "
        "covering process execution, argument passing and a system-call handler, and the "
        "work passed 100% of 80 concurrency, memory-fault and edge-case tests. That work "
        "is the closest I have come to low-level debugging under real correctness "
        "pressure.\n\n"
        "Sincerely,\nJay Niketan Pathare\n")


def _letter_call(client, run, bae):
    return client.cover_letter(
        bae["jd"], run["signals"],
        ["Built a fuzzy-matching combinator engine to merge asynchronous JSON outputs."],
        [run["master"].project("fraud")],
        themes=engine.jd_themes(bae["requirements"], 4))


def test_cover_letter_repairs_an_inexact_metric_on_retry(run, bae):
    """590 k is rejected; the corrected 590,540 attempt is accepted."""
    bad, good = _letter_body("590 k transactions"), _letter_body("590,540 transactions")
    transport, client = _sequence_client(run, [bad, good])
    letter, problems = _letter_call(client, run, bae)

    assert "590,540" in letter
    assert grounding.errors(problems) == [], [p.message for p in problems]
    letters = [c for c in client.calls if c.purpose == "cover_letter"]
    assert len(letters) == 2, [c.detail for c in letters]
    assert letters[0].accepted is False
    assert "validation failed" in letters[0].detail
    assert letters[1].accepted is True
    # the retry prompt carried the exact validator message. The preamble is
    # "was rejected" rather than "failed deterministic validation" because a
    # relevance retry is not a validation failure.
    retry = transport.requests[1].prompt
    assert "Previous attempt was rejected" in retry
    assert "nearest supported value is exactly 590540" in retry
    # grounding was not weakened: the bad rendering still fails on its own
    solo = grounding.validate_cover_letter(
        bad, run["master"], banned=run["policy"].banned_phrases,
        jd_text=bae["jd"].text, company="BAE Systems")
    assert grounding.errors(solo)


def test_cover_letter_gives_up_after_three_rejected_attempts(run, bae):
    bad = _letter_body("590 k transactions")
    transport, client = _sequence_client(run, [bad, bad, bad])
    letter, problems = _letter_call(client, run, bae)

    assert grounding.errors(problems), "an invalid letter must stay rejected"
    letters = [c for c in client.calls if c.purpose == "cover_letter"]
    assert len(letters) == 3
    assert all(c.accepted is False for c in letters)
    assert all(c.transport_ok is True for c in letters)
    assert letter  # the raw artifact is preserved, never discarded


def _assessment_payload(bae, *, broken: bool) -> str:
    requirement_id = None if broken else "REQ-006"
    return json.dumps({
        "fit_score": 7.5, "recommendation": "apply", "summary": "Provider prose.",
        "eligibility": {"status": "meets", "details": []},
        "strong_matches": [{"requirement_id": "REQ-007",
                            "evidence": "Python and JavaScript (React) in Skills.",
                            "source": "skills"}],
        "partial_matches": [], "gaps": [],
        "manual_review": [{"requirement_id": requirement_id, "reason": "needs a human"}],
        "complementary_strengths": [], "risk_flags": [],
        "tailoring_quality": {"score": 9.0, "notes": ["n"]}})


_ASSESS_EXPERIENCE = ["Led delivery of platform modules end to end."]
_ASSESS_BULLETS = {"lms": ["Optimized REST behavior with Locust."]}
_ASSESS_SKILLS = ["Python", "SQL", "OOP", "Linux", "Docker", "Operating Systems"]


def _assess_call(client, run, bae):
    return client.assess(
        bae["jd"], run["signals"], [run["master"].project("lms")], _ASSESS_SKILLS,
        experience_plain=_ASSESS_EXPERIENCE, project_bullets=_ASSESS_BULLETS,
        requirements=bae["requirements"], extra_experience=[bae["education"]])


def _assess_verdicts(bae):
    """The verdicts assess() derives from exactly the arguments above."""
    evidence = engine.ResumeEvidence(
        experience_text=" ".join(_ASSESS_EXPERIENCE + [bae["education"]]),
        project_bullets=tuple((pid, " ".join(b)) for pid, b in _ASSESS_BULLETS.items()),
        skills=tuple(_ASSESS_SKILLS))
    return engine.deterministic_assessment(
        bae["requirements"], bae["master"], evidence, bae["jd"].text)


def test_assessment_retries_a_null_requirement_id_then_accepts(run, bae):
    """The exact Epic failure: manual_review with requirement_id null."""
    transport, client = _sequence_client(
        run, [_assessment_payload(bae, broken=True),
              _assessment_payload(bae, broken=False)])
    data = _assess_call(client, run, bae)

    calls = [c for c in client.calls if c.purpose == "assessment"]
    assert len(calls) == 2
    assert calls[0].accepted is False and "schema validation failed" in calls[0].detail
    assert calls[1].accepted is True
    assert "failed deterministic schema validation" in transport.requests[1].prompt

    # Python's verdicts remain authoritative regardless of what the model sent
    assert data["verdict_source"] == "python_deterministic"
    verdicts = _assess_verdicts(bae)
    actual = {e["requirement_id"]: b for b in
              ("strong_matches", "partial_matches", "gaps", "manual_review")
              for e in data[b]}
    assert actual == engine.verdict_index(verdicts)
    # and no model per-requirement prose survived
    language = next(e for e in data["strong_matches"] if e["requirement_id"] == "REQ-007")
    assert "JavaScript" not in language["evidence"]
    assert "explanation" not in language
    problems = grounding.errors(grounding.validate_final_assessment(
        data, verdicts=verdicts, table=engine.requirement_table(bae["requirements"]),
        requirements=bae["requirements"]))
    assert problems == [], [p.message for p in problems]


def test_assessment_raises_after_three_malformed_attempts(run, bae):
    import llm_client

    broken = _assessment_payload(bae, broken=True)
    transport, client = _sequence_client(run, [broken, broken, broken])
    with pytest.raises(llm_client.ProviderError):
        _assess_call(client, run, bae)
    calls = [c for c in client.calls if c.purpose == "assessment"]
    assert len(calls) == 3
    assert all(c.accepted is False for c in calls)
    assert all(c.transport_ok is True for c in calls)


def test_production_run_folders_are_unique_per_execution(bae):
    """Same JD, same calendar day: two executions cannot share a directory."""
    jd = bae["jd"]
    first = run_pipeline.run_folder_for(jd, isolated=False, unique=True)
    second = run_pipeline.run_folder_for(jd, isolated=False, unique=True)
    assert first != second
    assert first.name != second.name
    today = engine.date.today().isoformat() if hasattr(engine, "date") else None
    # uniqueness does NOT come from the fingerprint, which is identical
    assert jd.fingerprint not in first.name and jd.fingerprint not in second.name
    assert jd.fingerprint[:8] not in first.name
    # both still carry the readable company/role/date prefix
    for path in (first, second):
        assert path.name.startswith("BAE_Systems_Entry_Level_Software_Engineer_")


def test_a_second_production_run_cannot_overwrite_the_first(tmp_path, monkeypatch, bae):
    """First run's artifacts stay byte-identical after a second execution."""
    monkeypatch.setattr(engine, "OUTPUT_DIR", tmp_path)
    first = run_pipeline.make_run_dir(bae["jd"], isolated=False)
    (first / "resume.pdf").write_bytes(b"FIRST RUN ARTIFACT")
    before = (first / "resume.pdf").read_bytes()

    second = run_pipeline.make_run_dir(bae["jd"], isolated=False)
    (second / "resume.pdf").write_bytes(b"SECOND RUN ARTIFACT")

    assert second != first
    assert (first / "resume.pdf").read_bytes() == before == b"FIRST RUN ARTIFACT"
    assert (second / "resume.pdf").read_bytes() == b"SECOND RUN ARTIFACT"
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted([first.name, second.name])


def _tracker_env(tmp_path, monkeypatch):
    """A Tracker over throwaway paths with a satisfied artifact set."""
    monkeypatch.setattr(engine, "OUTPUT_DIR", tmp_path / "output")
    (tmp_path / "output").mkdir()
    run_dir = tmp_path / "output" / "Acme_Corp_Backend_Engineer_2026-09-15_120000000001"
    run_dir.mkdir()
    for name in run_pipeline.REQUIRED_ARTIFACTS:
        (run_dir / name).write_text("x")
    tracker = run_pipeline.Tracker(tmp_path / "processed.csv", tmp_path / "index.json",
                                   enabled=True)
    return tracker, run_dir


def _stage_log():
    import logging
    return run_pipeline.StageLog(logging.getLogger("test-tracking"), "TEST")


def test_tracking_is_idempotent_for_an_already_recorded_fingerprint(tmp_path, monkeypatch,
                                                                    bae):
    tracker, run_dir = _tracker_env(tmp_path, monkeypatch)
    jd = bae["jd"]
    assert tracker.record(jd, run_dir, "success", _stage_log()) is True

    index = json.loads((tmp_path / "index.json").read_text())
    assert list(index) == [jd.fingerprint]
    assert index[jd.fingerprint]["run_folder"] == run_dir.name
    rows = [r for r in (tmp_path / "processed.csv").read_text().splitlines()[1:] if r]
    assert len(rows) == 1

    # a second success for the same fingerprint, from a different folder
    other = tmp_path / "output" / "Acme_Corp_Backend_Engineer_2026-09-15_999999999999"
    other.mkdir()
    for name in run_pipeline.REQUIRED_ARTIFACTS:
        (other / name).write_text("x")
    assert tracker.record(jd, other, "success", _stage_log()) is False

    assert json.loads((tmp_path / "index.json").read_text()) == index
    assert index[jd.fingerprint]["run_folder"] == run_dir.name
    rows = [r for r in (tmp_path / "processed.csv").read_text().splitlines()[1:] if r]
    assert len(rows) == 1, rows


def test_tracking_stops_when_the_indexed_run_folder_is_missing(tmp_path, monkeypatch, bae):
    tracker, run_dir = _tracker_env(tmp_path, monkeypatch)
    (tmp_path / "index.json").write_text(json.dumps(
        {bae["jd"].fingerprint: {"run_folder": "Vanished_Folder_2026-09-14",
                                 "source_file": "bae.txt"}}))
    assert tracker.record(bae["jd"], run_dir, "success", _stage_log()) is False
    # nothing invented, nothing written
    assert "Vanished_Folder_2026-09-14" in (tmp_path / "index.json").read_text()
    assert not (tmp_path / "processed.csv").exists()


def test_index_failure_leaves_the_csv_untouched(tmp_path, monkeypatch, bae):
    """The authoritative index commits first; a failure must not log a row."""
    tracker, run_dir = _tracker_env(tmp_path, monkeypatch)

    def explode(*args, **kwargs):
        raise OSError("simulated index write failure")

    monkeypatch.setattr(run_pipeline.os, "replace", explode)
    assert tracker.record(bae["jd"], run_dir, "success", _stage_log()) is False
    assert not (tmp_path / "index.json").exists()
    assert not (tmp_path / "processed.csv").exists()
    assert not list(tmp_path.glob("index.json.tmp"))


def test_csv_failure_after_a_committed_index_is_not_rolled_back(tmp_path, monkeypatch,
                                                                bae):
    tracker, run_dir = _tracker_env(tmp_path, monkeypatch)
    # make the CSV path un-appendable by turning it into a directory
    (tmp_path / "processed.csv").mkdir()
    assert tracker.record(bae["jd"], run_dir, "success", _stage_log()) is True
    index = json.loads((tmp_path / "index.json").read_text())
    assert index[bae["jd"].fingerprint]["run_folder"] == run_dir.name


# ---- X. Malformed provider JSON and corrupt tracking state ---------------

def test_unparseable_assessment_output_is_not_retried_in_a_looser_mode(run, bae):
    """Strict Structured Outputs makes the SHAPE the provider's contract.

    Under json-object mode an unparseable reply was worth one more attempt.
    Under a strict schema it means the provider broke its own guarantee, which
    a reworded prompt cannot fix - and retrying in a looser mode would hide a
    schema bug. So it fails once, clearly.
    """
    import llm_client

    transport, client = _sequence_client(
        run, ["I'm sorry, I cannot produce JSON for this request.",
              _assessment_payload(bae, broken=False)])
    with pytest.raises(llm_client.ProviderError) as raised:
        _assess_call(client, run, bae)

    assert raised.value.category == "malformed_response"
    assert "strict structured output" in str(raised.value)
    calls = [c for c in client.calls if c.purpose == "assessment"]
    assert len(calls) == 1, "there must be no second, looser attempt"
    assert calls[0].transport_ok is True
    assert calls[0].accepted is False
    assert "malformed JSON" in calls[0].detail
    # The second payload was never requested.
    assert len(transport.requests) == 1


def test_a_schema_the_endpoint_rejects_surfaces_as_its_own_error():
    """A configuration 400 must not be retried or hidden behind a fallback."""
    import llm_client

    assert llm_client.classify_http(
        400, "Failed to validate JSON. Please adjust your prompt. "
             "code=json_validate_failed") == "schema_rejected"
    assert llm_client.classify_http(400, "invalid json_schema") == "schema_rejected"
    assert "schema_rejected" in llm_client.NON_RETRYABLE_CATEGORIES
    assert "schema_rejected" not in llm_client.KEY_SPECIFIC_CATEGORIES

    attempts = []

    def reject(self, request):
        attempts.append(1)
        raise llm_client.ProviderError("schema_rejected", "json_validate_failed")

    transport = _stub_groq(generate=reject, model=QWEN)
    with pytest.raises(llm_client.ProviderError) as raised:
        transport.generate(llm_client.Request("assessment", "p", max_tokens=900))
    assert attempts == [1], "a rejected schema is never retried"
    assert raised.value.category == "schema_rejected"
    # And no Gemini fallback is reachable from the audit transport.
    assert transport.fallback is None


def test_schema_validation_does_not_replace_python_validation(run, bae):
    """The schema guarantees shape; Python still owns the business rules."""
    source = Path(_llm_client().__file__).read_text(encoding="utf-8")
    assess = source.split("    def assess(")[1].split("\n    def ")[0]
    for step in ("parse_json(reply.text", "grounding.canonicalize_assessment(",
                 "grounding.validate_assessment(",
                 "grounding.validate_application_audit(",
                 "grounding.finalize_assessment("):
        assert step in assess, step
    # Python's verdicts still overwrite the model's buckets.
    transport, client = _sequence_client(run, [_assessment_payload(bae, broken=False)])
    data = _assess_call(client, run, bae)
    assert data["verdict_source"] == "python_deterministic"
    actual = {e["requirement_id"]: b for b in
              ("strong_matches", "partial_matches", "gaps", "manual_review")
              for e in data[b]}
    assert actual == engine.verdict_index(_assess_verdicts(bae))


def test_assessment_transport_errors_are_not_retried_here(run, bae):
    """Transport retry/failover belongs to the transport layer, not assess()."""
    import llm_client

    class _FailingTransport:
        name = "stub"

        def __init__(self):
            self.calls = 0

        def generate(self, request):
            self.calls += 1
            raise llm_client.ProviderError("server_error", "503 upstream")

    import logging
    transport = _FailingTransport()
    client = llm_client.LLMClient(transport, transport, run["master"], run["policy"],
                                  logging.getLogger("test"))
    with pytest.raises(llm_client.ProviderError) as caught:
        _assess_call(client, run, bae)
    assert caught.value.category == "server_error"
    assert transport.calls == 1, "a transport failure must not be retried in assess()"
    assert [c.transport_ok for c in client.calls if c.purpose == "assessment"] == [False]


def test_missing_tracking_index_is_a_fresh_state(tmp_path):
    assert run_pipeline.load_index(tmp_path / "absent.json") == {}


@pytest.mark.parametrize("content, expected", [
    ("{}", {}),
    ('{"abc123": {"run_folder": "Acme_2026-09-15"}}',
     {"abc123": {"run_folder": "Acme_2026-09-15"}}),
])
def test_valid_tracking_index_is_accepted(tmp_path, content, expected):
    path = tmp_path / "index.json"
    path.write_text(content)
    assert run_pipeline.load_index(path) == expected


@pytest.mark.parametrize("content, reason", [
    ("", "empty"),
    ("   \n", "empty"),
    ("{not json", "not valid JSON"),
    ("[]", "expected a JSON object"),
    ('"a string"', "expected a JSON object"),
])
def test_corrupt_tracking_index_fails_closed(tmp_path, content, reason):
    """A damaged skip source must never read as "nothing processed yet"."""
    path = tmp_path / "index.json"
    path.write_text(content)
    with pytest.raises(run_pipeline.TrackingStateError) as caught:
        run_pipeline.load_index(path)
    assert reason in str(caught.value)


def test_a_corrupt_index_blocks_discovery_and_recording(tmp_path, monkeypatch, bae):
    """Both the skip path and the record path refuse to proceed."""
    index = tmp_path / "index.json"
    index.write_text("")                      # the real production state
    tracker = run_pipeline.Tracker(tmp_path / "processed.csv", index, enabled=True)
    with pytest.raises(run_pipeline.TrackingStateError):
        tracker.already_processed(bae["jd"].fingerprint)

    monkeypatch.setattr(engine, "OUTPUT_DIR", tmp_path / "output")
    (tmp_path / "output").mkdir()
    run_dir = tmp_path / "output" / "Acme_2026-09-15_120000000000"
    run_dir.mkdir()
    for name in run_pipeline.REQUIRED_ARTIFACTS:
        (run_dir / name).write_text("x")
    # record() reports the corruption and writes nothing
    assert tracker.record(bae["jd"], run_dir, "success", _stage_log()) is False
    assert not (tmp_path / "processed.csv").exists()
    assert index.read_text() == ""


def test_a_disabled_tracker_ignores_a_corrupt_index(tmp_path, bae):
    """Mock/smoke runs stay isolated from real production tracking state."""
    index = tmp_path / "index.json"
    index.write_text("")
    tracker = run_pipeline.Tracker(tmp_path / "processed.csv", index, enabled=False)
    assert tracker.already_processed(bae["jd"].fingerprint) is None


# ---- Y. Redwood/Superhuman real-JD defects -------------------------------

# The Superhuman posting was renamed on disk; locate it rather than assume.
REDWOOD_JD = next((p for p in sorted((ROOT / "jobs").glob("*.txt"))
                   if "Superhuman" in p.read_text(encoding="utf-8")),
                  ROOT / "jobs" / "redwoodq.txt")


@pytest.fixture(scope="session")
def redwood() -> dict:
    master = engine.load_master()
    jd = engine.read_jd(REDWOOD_JD)
    requirements = engine.extract_jd_requirements(jd.text, master)
    return {"master": master, "jd": jd, "requirements": requirements}


def _redwood_requirement(redwood, needle: str) -> engine.Requirement:
    found = [r for r in redwood["requirements"] if needle.lower() in r.original_text.lower()]
    assert len(found) == 1, (needle, [r.original_text[:60] for r in redwood["requirements"]])
    return found[0]


def test_redwood_company_is_superhuman_not_the_filename(redwood):
    """"redwood.txt" is not evidence that the employer is Redwood."""
    assert redwood["jd"].company_name == "Superhuman"
    assert "Redwood" not in (redwood["jd"].company_name or "")


def test_redwood_title_keeps_the_early_career_qualifier(redwood):
    assert redwood["jd"].job_title == "Software Engineer, Early Career"


def test_filename_is_never_authoritative_for_a_substantive_posting(tmp_path):
    """The stem may still rescue a scrap of a file, never a real posting."""
    real = tmp_path / "acmecorp.txt"
    real.write_text(REDWOOD_JD.read_text(encoding="utf-8"), encoding="utf-8")
    assert engine.substantive_jd(real.read_text(encoding="utf-8"))
    assert engine.read_jd(real).company_name == "Superhuman"

    scrap = tmp_path / "acmecorp2.txt"
    scrap.write_text("A short note, not a posting.")
    assert not engine.substantive_jd(scrap.read_text())
    assert engine.read_jd(scrap).company_name == "Acmecorp2"


@pytest.mark.parametrize("benefit", [
    "Disability and life insurance options",
    "401(k) and RRSP matching",
    "Paid parental leave",
    "20 days of paid time off per year",
    "Generous stipends",
    "Annual professional development budget",
    "Excellent health care",
])
def test_redwood_benefits_never_become_requirements(redwood, benefit):
    for requirement in redwood["requirements"]:
        assert benefit.lower() not in requirement.original_text.lower(), requirement


def test_benefit_sections_are_recognized_with_or_without_a_colon():
    for heading in ("Compensation and Benefits", "Benefits:", "Perks",
                    "Why You'll Love It Here", "What We Offer"):
        assert engine.suppressed_section(heading), heading
    for heading in ("Requirements:", "Must-haves:", "What You'll Do", "Education"):
        assert not engine.suppressed_section(heading), heading


def test_product_go_is_not_the_go_language_but_the_language_list_is(redwood):
    """Both Go sentences together, so neither reading can regress."""
    product = _redwood_requirement(redwood, "Build and iterate on features for Grammarly")
    assert "Go" in product.original_text
    assert product.terms == (), product.terms

    languages = _redwood_requirement(redwood, "Proficiency in at least one programming")
    assert "Go" in languages.terms
    assert "Python" in languages.terms


def test_python_satisfies_the_redwood_language_or_group(redwood):
    languages = _redwood_requirement(redwood, "Proficiency in at least one programming")
    assert engine.any_of_list(languages.original_text)
    groups = engine.requirement_groups(languages)
    assert len(groups) == 1, groups
    evidence = engine.ResumeEvidence(
        experience_text="Developed an ETL workflow using Python (Pandas, NumPy).",
        skills=("Python", "SQL"))
    match = engine.classify_requirement(languages, redwood["master"], evidence)
    assert match.verdict == "strong_match"
    assert "Python" in match.evidence
    for absent in ("Java ", "C++", "JavaScript", "TypeScript"):
        assert absent not in match.evidence


def test_redwood_onsite_cadence_is_logistics_not_a_capability(redwood):
    onsite = _redwood_requirement(redwood, "2 days per week")
    assert onsite.kind == "logistics"
    assert onsite.scored is False
    assert onsite in engine.manual_review_requirements(redwood["requirements"])
    assert onsite not in engine.scored_requirements(redwood["requirements"])
    # a location cadence must never collect technical evidence
    assert onsite.terms == (), onsite.terms


def test_redwood_graduation_year_alternatives_resolve_to_meets(redwood):
    window = engine.graduation_requirement(redwood["jd"].text)
    assert window and window["low"] == (2026, 1) and window["high"] == (2027, 12)
    status, details, _ = engine.eligibility_assessment(
        redwood["requirements"], redwood["master"], redwood["jd"].text)
    assert status == "meets"
    # Derived from the master, never hardcoded: the graduation date lives there.
    expected = engine.expected_graduation(redwood["master"])
    assert expected and window["low"] <= expected <= window["high"]
    assert any("falls inside" in d for d in details), details


@pytest.mark.parametrize("phrasing, low, high", [
    ("graduating in 2026 or 2027", (2026, 1), (2027, 12)),
    ("graduating in 2026/2027", (2026, 1), (2027, 12)),
    ("graduating in either 2026 or 2027", (2026, 1), (2027, 12)),
])
def test_graduation_year_alternative_grammars(phrasing, low, high):
    window = engine.graduation_requirement(f"Candidates {phrasing} are eligible.")
    assert window and window["low"] == low and window["high"] == high


def test_month_range_graduation_still_wins(redwood):
    """The New Grad month window must not be shadowed by the year form."""
    text = "graduating between December 2026 and June 2027"
    window = engine.graduation_requirement(text)
    assert window["low"] == (2026, 12) and window["high"] == (2027, 6)


def test_distributed_systems_cannot_be_claimed_as_experience(redwood):
    """The exact Redwood cover-letter failure."""
    evidence = engine.ResumeEvidence(
        experience_text="Built a fuzzy-matching combinator engine.",
        project_bullets=(("lms", "Implemented a real-time subsystem using Redis."),),
        skills=("Python", "SQL", "ReactJS"))
    unsupported = engine.unsupported_jd_concepts(
        redwood["requirements"], redwood["master"], evidence)
    assert any("distributed systems" == c.lower() for c in unsupported), unsupported

    failing = ("My experience building fault-tolerant pipelines and real-time distributed "
               "systems matches the need to ship production-grade code.")
    problems = grounding.unsupported_ownership(failing, unsupported, company="Superhuman")
    assert problems and "distributed systems" in problems[0].message

    grounded = ("My experience building real-time backend systems matches the need to ship "
                "production-grade code.")
    assert grounding.unsupported_ownership(grounded, unsupported,
                                           company="Superhuman") == []

    employer = "Superhuman builds large-scale distributed systems that stay highly available."
    assert grounding.unsupported_ownership(employer, unsupported,
                                           company="Superhuman") == []


def test_unsupported_gate_does_not_ban_supported_wording(redwood):
    """Only unsupported concepts are gated; Python stays claimable."""
    evidence = engine.ResumeEvidence(
        experience_text="Developed an ETL workflow using Python.", skills=("Python",))
    unsupported = engine.unsupported_jd_concepts(
        redwood["requirements"], redwood["master"], evidence)
    assert not any(c.lower() == "python" for c in unsupported), unsupported
    assert grounding.unsupported_ownership(
        "I built an ETL workflow in Python.", unsupported, company="Superhuman") == []


def test_react_in_a_jd_names_the_supported_reactjs_skill(redwood):
    """"React" and "ReactJS" are the same library, differently spelled."""
    jd_text = redwood["jd"].text
    assert "React" in jd_text and "ReactJS" not in jd_text
    assert engine._names_skill(jd_text, "ReactJS")
    web = _redwood_requirement(redwood, "Experience with web tech")
    assert "ReactJS" in web.terms
    # neighbours the master does not support are never invented
    master = redwood["master"]
    assert master.canonical_skill("Node.js") is None
    assert master.canonical_skill("TypeScript") is None


# ---- Z. C3 real-JD defects -----------------------------------------------

C3_JD = ROOT / "jobs" / "c3.txt"


@pytest.fixture(scope="session")
def c3() -> dict:
    master = engine.load_master()
    jd = engine.read_jd(C3_JD)
    return {"master": master, "jd": jd,
            "requirements": engine.extract_jd_requirements(jd.text, master)}


def test_c3_title_from_as_a_role_you_will(c3):
    """"As an Forward Deployed Engineer, you will design ..." (sic)."""
    assert c3["jd"].company_name == "C3 AI"
    assert c3["jd"].job_title == "Forward Deployed Engineer"


@pytest.mark.parametrize("sentence, expected", [
    ("As a Platform Engineer, you will build services.", "Platform Engineer"),
    ("As an Forward Deployed Engineer, you will design apps.", "Forward Deployed Engineer"),
    ("As a Data Scientist, you'll model demand.", "Data Scientist"),
    ("As an Analytics Developer, you are expected to ship.", "Analytics Developer"),
])
def test_role_comma_title_is_generic(sentence, expected):
    assert engine._role_comma_title(sentence) == expected


def test_role_comma_title_does_not_hijack_role_at_postings():
    """"As a software developer at Epic, you'll ..." belongs to the role-at rule."""
    assert engine._role_comma_title("As a software developer at Epic, you'll write code.") is None
    epic = engine.read_jd(ROOT / "jobs" / "epic.txt")
    assert epic.company_name == "Epic" and epic.job_title == "Software Developer"


@pytest.mark.parametrize("lines, fill, better_than", [
    (2, 16.0, (1, 95.0)),
    (2, 44.0, (2, 16.0)),
    (2, 16.0, (3, 50.0)),
])
def test_two_lines_always_outrank_one_line(lines, fill, better_than):
    """A thin two-line tail beats a single line, so repair cannot degrade it."""
    rank = run_pipeline._bullet_rank(lines, fill, 2, 20.0)
    worse = run_pipeline._bullet_rank(better_than[0], better_than[1], 2, 20.0)
    assert rank > worse, (rank, worse)


def test_a_thin_tail_asks_to_expand_not_shorten():
    """The exact C3 failure: fraud#2 at 2 lines / 16% chose shorten."""
    class _Measured:
        def __init__(self, lines, fill):
            self.lines, self.fill_pct = lines, fill

        def verdict(self, orphan, acceptable, ideal):
            if self.lines == 1:
                return "single_line"
            return "hard_orphan" if self.fill_pct < orphan else "ideal"

    policy = engine.load_policy()
    render = engine.ProjectRender(engine.load_master().project("fraud"), ["a", "b"])
    problems = run_pipeline._layout_problems(
        [_Measured(2, 44.0), _Measured(2, 16.0)], [render], policy, 1)
    goals = [a["goal"] for a in problems["actions"]]
    assert goals == ["lengthen"], problems["actions"]
    assert "more of this project's evidence" in problems["actions"][0]["reason"]


def test_restore_best_undoes_a_degrading_repair():
    """2L/16% -> repaired to 1L must not be the stored bullet."""
    import logging

    master = engine.load_master()
    render = engine.ProjectRender(master.project("fraud"), ["good two-line text"])
    best = {}
    run_pipeline._track_best(
        best, [type("M", (), {"lines": 2, "fill_pct": 16.0})()], [render],
        engine.load_policy())
    assert best[("fraud", 0)][1] == "good two-line text"

    # a repair degrades it to a single line
    render.bullets[0] = "degraded one-line text"
    run_pipeline._track_best(
        best, [type("M", (), {"lines": 1, "fill_pct": 95.0})()], [render],
        engine.load_policy())
    assert best[("fraud", 0)][1] == "good two-line text", "the 1-line text must not win"

    run_pipeline._restore_best(
        best, [render], run_pipeline.StageLog(logging.getLogger("t"), "T"))
    assert render.bullets[0] == "good two-line text"


@pytest.fixture(scope="session")
def capsules(c3) -> dict:
    return engine.source_capsules(c3["master"])


@pytest.mark.parametrize("paragraph, why", [
    ("I engineered an OCR pipeline with Tesseract-OCR and Qwen-VL. The solution runs on "
     "AWS EC2 and RDS.", "EC2/RDS belongs to the cloud role, not HeinOnline"),
    ("I engineered an OCR pipeline with Tesseract-OCR raising accuracy to 90.81% through "
     "unit testing.", "unit testing has no HeinOnline evidence"),
])
def test_cross_source_attribution_fails(c3, capsules, paragraph, why):
    problems = grounding.validate_source_scope(paragraph, capsules, c3["master"])
    assert problems, why


@pytest.mark.parametrize("paragraph", [
    "I engineered an OCR pipeline with Tesseract-OCR and Qwen models across CrossRef "
    "sources, raising extraction accuracy to 90.81%.",
    "I designed and deployed scalable AWS EC2 and RDS infrastructure with VPC networking.",
    "I load-tested the API with Locust, cutting p95 latency by approximately 26%.",
])
def test_single_source_paragraphs_still_pass(c3, capsules, paragraph):
    assert grounding.validate_source_scope(paragraph, capsules, c3["master"]) == []


def test_profiling_is_not_claimable_when_only_load_testing_exists(c3):
    """The master says load testing is not profiling."""
    evidence = engine.ResumeEvidence(
        experience_text="Load-tested the API with Locust.",
        skills=("Python", "Load Testing"))
    unsupported = engine.unsupported_jd_concepts(
        c3["requirements"], c3["master"], evidence)
    assert any("profil" in c.lower() for c in unsupported), unsupported
    claim = ("This project showcases my React expertise and systematic performance "
             "profiling.")
    assert grounding.unsupported_ownership(claim, unsupported, company="C3 AI")
    # load testing itself stays claimable
    assert grounding.unsupported_ownership(
        "I load-tested the API with Locust.", unsupported, company="C3 AI") == []


def test_c3_prose_graduation_gate_is_found_and_routed_to_manual(c3):
    """A standalone prose sentence, not a bullet."""
    gates = engine.graduation_gate_sentences(c3["jd"].text)
    assert gates and "December 2026 or Summer/Fall 2027" in gates[0]
    window = engine.graduation_requirement(c3["jd"].text)
    assert window and window["options"] == [((2026, 12), (2026, 12)),
                                            ((2027, 6), (2027, 12))]
    eligibility = [r for r in c3["requirements"] if r.kind == "eligibility"]
    assert eligibility, "the gate must become an eligibility requirement"
    assert all(not r.scored for r in eligibility)

    status, details, flags = engine.eligibility_assessment(
        c3["requirements"], c3["master"], c3["jd"].text)
    assert status == "uncertain", (status, details)
    assert any("conferral" in d for d in details)
    assert flags


def test_simpler_graduation_cases_keep_their_exact_resolution():
    master = engine.load_master()
    for path in (REDWOOD_JD, ROOT / "jobs_synthetic" / "newgrad_swe.txt"):
        jd = engine.read_jd(path)
        requirements = engine.extract_jd_requirements(jd.text, master)
        status, _, _ = engine.eligibility_assessment(requirements, master, jd.text)
        assert status == "meets", (path.name, status)


# ---- AA. Candidate education / work-authorization dates ------------------

@pytest.mark.parametrize("sentence", [
    "I expect to graduate in December 2026.",
    "I finish my Master's in December 2026.",
    "I can start full time in January 2027 under OPT.",
    "I can begin OPT work immediately after graduating in December 2026.",
    # the exact sentence my own C3 mock shipped
    "I finish my Master of Science in Computer Science at the University at Buffalo in "
    "December 2026 and can start full time in January 2027 under OPT.",
])
def test_unsupported_candidate_dates_are_rejected(sentence):
    master = engine.load_master()
    problems = grounding.validate_candidate_dates(sentence, master)
    assert problems, sentence
    assert all(p.kind == "candidate_date" for p in problems)


@pytest.mark.parametrize("sentence", [
    "I expect to graduate in February 2027.",
    "I am completing an M.S. in Computer Science at the University at Buffalo.",
    "I am eligible to begin post-completion OPT employment from February 2, 2027, "
    "subject to OPT/EAD authorization.",
    "I am completing an M.S. in Computer Science at the University at Buffalo, expected "
    "February 2027, after earning a B.E. in Information Technology.",
])
def test_supported_candidate_dates_pass(sentence):
    master = engine.load_master()
    assert grounding.validate_candidate_dates(sentence, master) == [], sentence


def test_employer_graduation_window_is_still_quotable():
    """The JD's own eligibility window is the employer's date, not a claim."""
    master = engine.load_master()
    jd_sentence = ("The role is open to candidates graduating in December 2026 or "
                   "Summer/Fall 2027.")
    assert grounding.validate_candidate_dates(jd_sentence, master) == []


def test_jd_and_candidate_dates_stay_separately_attributed():
    """Both dates in one paragraph: the JD's window and the candidate's own."""
    master = engine.load_master()
    paragraph = ("The role is open to candidates graduating in December 2026 or "
                 "Summer/Fall 2027. I expect to graduate in February 2027.")
    assert grounding.validate_candidate_dates(paragraph, master) == []
    # swapping the candidate's date for the JD's must fail
    swapped = ("The role is open to candidates graduating in December 2026 or "
               "Summer/Fall 2027. I expect to graduate in December 2026.")
    assert grounding.validate_candidate_dates(swapped, master)


def test_opt_without_its_authorization_qualifier_is_rejected():
    master = engine.load_master()
    bare = "I will be eligible for OPT from February 2027."
    problems = grounding.validate_candidate_dates(bare, master)
    assert any("qualifier" in p.message for p in problems), [p.message for p in problems]


def test_the_letter_prompt_forbids_deriving_candidate_dates_from_the_jd():
    import llm_client
    rules = llm_client._LETTER_RULES
    assert "CANDIDATE DATES COME ONLY FROM THE STANDING FACTS" in rules
    assert "Never infer an" in rules and "OPT start date" in rules
    assert "expected February 2027" in rules


def test_the_mock_eligibility_sentence_is_grounded():
    """The fixture was updated only after the validator rejected the old text."""
    import logging

    import llm_client
    master = engine.load_master()
    transport = llm_client.MockTransport(master, logging.getLogger("test"))
    letter = transport._letter({"job_title": "Forward Deployed Engineer",
                                "company": "C3 AI", "project_ids": ["lms"],
                                "jd_text": "graduating in December 2026 or Fall 2027",
                                "needs_eligibility": True, "themes": []})
    assert "December 2026" not in letter
    assert "January 2027" not in letter
    assert "OPT" not in letter
    assert grounding.validate_candidate_dates(letter, master) == []


# ---- AB. Role-level supplements belong to their own role -----------------

def test_role_supplements_are_not_a_global_exempt_pool():
    master = engine.load_master()
    prefixes = engine.role_bullet_prefixes(master)
    assert prefixes["Graduate Student Developer (CSE 611 Project)"] == "GSD"
    assert prefixes["Software Engineer"] == "SWE"

    capsules = engine.source_capsules(master)
    assert "role:Graduate" not in capsules and "role:Software" not in capsules
    # the supplement's own facts now live with their role
    assert "reproducible execution" in capsules["role:GSD"]
    assert "technical interviews" in capsules["role:SWE"]


def test_a_gsd_supplement_fact_cannot_be_attributed_to_thesis():
    """Anchors from a role supplement now identify that role, not "general"."""
    master = engine.load_master()
    capsules = engine.source_capsules(master)
    anchors = grounding._source_anchors(capsules, master)
    assert anchors.get("Agile") == "role:SWE", "an SWE supplement fact must anchor to SWE"

    mixed = ("I processed 47 law journals raising accuracy to 90.81% while working in "
             "Agile sprints and conducting 20+ technical interviews.")
    problems = grounding.validate_source_scope(mixed, capsules, master)
    assert problems, "a GSD metric plus an SWE supplement fact must be cross-source"
    assert "role:GSD" in problems[0].message and "role:SWE" in problems[0].message


# ---- AC. Provider default, compact assessment, artifact hygiene ----------

def test_no_groq_model_never_resolves_to_the_removed_default(monkeypatch):
    """An unset GROQ_MODEL must not resurrect openai/gpt-oss-120b."""
    import importlib
    import resume_engine

    monkeypatch.delenv("GROQ_MODEL", raising=False)
    reloaded = importlib.reload(resume_engine)
    try:
        assert reloaded.GROQ_MODEL == ""
        assert "gpt-oss" not in (reloaded.GROQ_MODEL or "")
    finally:
        importlib.reload(resume_engine)


def test_groq_transport_without_a_model_is_unavailable_and_falls_back():
    import logging

    import llm_client
    master = engine.load_master()
    credentials = engine.load_credentials()
    gemini = llm_client.GeminiTransport(credentials, logging.getLogger("t"))
    groq = llm_client.GroqTransport(credentials, logging.getLogger("t"), model="",
                                    fallback=gemini)
    assert groq.available is False
    lonely = llm_client.GroqTransport(credentials, logging.getLogger("t"), model="",
                                      fallback=None)
    with pytest.raises(llm_client.ProviderError) as caught:
        lonely.generate(llm_client.Request("assessment", "x"))
    assert "no Groq model is configured" in str(caught.value)
    assert master is not None


class _Verified:
    experience_exact = True
    issues: list = []
    pages = 1
    action_verbs_ok = True


def _research(jd_text: str, **overrides) -> dict:
    """Company research exactly as production computes it.

    The provider answers UNKNOWN (mock mode cannot browse) and Python applies
    the deterministic rules: explicit posting language, the E-Verify rule and
    the job-level override.
    """
    payload = {"company_visa_sponsorship": "UNKNOWN", "company_visa_confidence": "LOW",
               "company_stem_opt_support": "UNKNOWN", "company_stem_opt_confidence": "LOW",
               "job_posted": "UNKNOWN", "job_posted_confidence": "LOW",
               "checked_at": "2026-09-15", "sources": []}
    payload.update(overrides)
    research, _ = grounding.validate_company_research(
        payload, jd_text=jd_text, jd_posted=grounding.job_posted_date(jd_text))
    return research


def _report(jd_text: str = "A posting with no dates or sponsorship language.",
            **kwargs) -> dict:
    base = dict(application_audit={"experience_selection_score": 8.7,
                                   "resume_tailoring_score": 9.5,
                                   "project_selection_score": 8.5,
                                   "project_bullet_score": 8.5,
                                   "callback_likelihood": "MEDIUM",
                                   "fit_score": 8.2},
                research=_research(jd_text), letter_score=9.5)
    base.update(kwargs)
    return grounding.audit_report(**base)


def test_assessment_txt_holds_exactly_the_agreed_fields():
    body = grounding.render_assessment_txt(_report())
    lines = [line for line in body.splitlines() if line.strip()]
    assert [line.split(":")[0] for line in lines] == list(grounding.ASSESSMENT_FIELDS)
    for banned in ("strong_matches", "partial_matches", "gaps",
                   "complementary_strengths", "model_summary", "requirement_id", "{",
                   "confidence", "http", "component"):
        assert banned not in body


def test_unknown_sponsorship_stays_unknown_when_the_jd_is_silent():
    assert _report()["Company Visa Sponsorship"] == "UNKNOWN"


@pytest.mark.parametrize("phrase, expected", [
    ("We do not offer visa sponsorship for this position.", "NO"),
    ("Unable to sponsor visas at this time.", "NO"),
    ("Visa sponsorship is available for this role.", "YES"),
    ("Nothing about work authorization here.", "UNKNOWN"),
])
def test_explicit_sponsorship_language_only(phrase, expected):
    assert _report(phrase)["Company Visa Sponsorship"] == expected


def test_stem_opt_is_unknown_without_evidence_and_never_inferred():
    """E-Verify participation must not imply a STEM OPT conclusion."""
    assert _report()["Company STEM OPT Support"] == "UNKNOWN"
    everify = _report("We participate in the US federal E-Verify program.")
    assert everify["Company STEM OPT Support"] == "UNKNOWN"
    assert _report("This employer supports the STEM OPT extension."
                   )["Company STEM OPT Support"] == "YES"


@pytest.mark.parametrize("jd_text, expected", [
    ("No timestamp anywhere in this posting.", "UNKNOWN"),
    ("Posted on March 4, 2026 by the hiring team.", "2026-03-04"),
    ("Published: 2026-03-04", "2026-03-04"),
])
def test_job_posted_only_from_real_evidence(jd_text, expected):
    assert _report(jd_text)["Job Posted"] == expected


def test_job_posted_never_substitutes_a_local_timestamp(tmp_path):
    """No file mtime, run time or download time may stand in for a posting date."""
    report = _report("A posting with no date at all.")
    assert report["Job Posted"] == "UNKNOWN"
    assert str(engine.date.today().year) not in report["Job Posted"]
    # Nor may a researched value that is merely today's date pass as evidence
    # when the posting itself says nothing: it must still be a real claim.
    today = engine.date.today().isoformat()
    assert _research("A posting with no date at all.",
                     job_posted=today)["job_posted"] == today
    assert _research("A posting with no date at all.",
                     job_posted="not a date")["job_posted"] == "UNKNOWN"


def test_callback_likelihood_is_one_field_and_never_a_percentage():
    report = _report()
    assert report["Callback Likelihood"] in grounding.CALLBACK_LIKELIHOOD
    body = grounding.render_assessment_txt(report)
    assert "%" not in body
    assert not re.search(r"\d+\s*(?:percent|probability|chance)", body, re.IGNORECASE)
    assert body.count("Callback") == 1
    # An absent or malformed provider value degrades, never guesses.
    assert _report(application_audit={})["Callback Likelihood"] == "UNKNOWN"
    audit, _ = grounding.validate_application_audit({"callback_likelihood": "72%"})
    assert audit["callback_likelihood"] == "UNKNOWN"


def test_the_report_is_advisory_when_an_audit_is_unavailable():
    report = _report(application_audit={})
    assert report["Fit Match Score"] == "UNKNOWN"
    assert report["Resume Tailoring Score"] == "UNKNOWN"
    assert report["Experience Selection Score"] == "UNKNOWN"
    assert report["Callback Likelihood"] == "UNKNOWN"
    # The deterministic half still scores: it needs no provider at all.
    assert report["Cover Letter Score"] == "9.5 / 10"


def test_final_pdf_is_named_company_and_title():
    jd = engine.read_jd(C3_JD)
    assert run_pipeline.final_pdf_name(jd) == "C3_AI_Forward_Deployed_Engineer.pdf"
    assert engine.date.today().isoformat() not in run_pipeline.final_pdf_name(jd)


def test_success_removes_latex_junk_but_keeps_run_log(tmp_path):
    import logging

    jd = engine.read_jd(C3_JD)
    for name in ("resume.pdf", "resume.aux", "resume.log", "resume.out", "run.log",
                 "resume.tex", "resume.txt", "assessment.json"):
        (tmp_path / name).write_text("x")
    final = run_pipeline.finalize_artifacts(
        tmp_path, jd, run_pipeline.StageLog(logging.getLogger("t"), "T"))

    assert final and final.name == "C3_AI_Forward_Deployed_Engineer.pdf"
    assert final.exists()
    assert not (tmp_path / "resume.pdf").exists()
    for gone in ("resume.aux", "resume.log", "resume.out", "assessment.json"):
        assert not (tmp_path / gone).exists(), gone
    for kept in ("run.log", "resume.tex", "resume.txt"):
        assert (tmp_path / kept).exists(), kept
    assert len(list(tmp_path.glob("*.pdf"))) == 1


def test_required_artifacts_describe_the_new_contract():
    assert "assessment.txt" in run_pipeline.REQUIRED_ARTIFACTS
    assert "assessment.json" not in run_pipeline.REQUIRED_ARTIFACTS
    assert "resume.pdf" not in run_pipeline.REQUIRED_ARTIFACTS
    assert "run.log" in run_pipeline.REQUIRED_ARTIFACTS


def test_a_failed_run_keeps_latex_diagnostics(tmp_path):
    """finalize_artifacts is only reached on success, so junk survives a failure."""
    for name in ("resume.pdf", "resume.aux", "resume.log", "resume.out", "run.log"):
        (tmp_path / name).write_text("x")
    # needs_review path: finalize_artifacts is never called
    for kept in ("resume.pdf", "resume.aux", "resume.log", "resume.out", "run.log"):
        assert (tmp_path / kept).exists(), kept
    source = Path(run_pipeline.__file__).read_text()
    guard = source.split("final_pdf = finalize_artifacts")[0]
    assert guard.rstrip().endswith('if status == "success":'), \
        "cleanup must be gated on success"


def test_revalidate_accepts_the_renamed_final_pdf():
    """Revalidation must not assume the PDF is called resume.pdf."""
    source = Path(run_pipeline.__file__).read_text()
    revalidate = source.split("def revalidate(")[1]
    assert 'run_dir.glob("*.pdf")' in revalidate
    folders = sorted((ROOT / "output" / "_smoke_tests").glob("C3_AI_*"))
    if not folders:
        pytest.skip("no C3 smoke folder present")
    run_dir = folders[-1]
    if not list(run_dir.glob("*.pdf")):
        pytest.skip("the C3 smoke folder holds no PDF")
    assert not (run_dir / "resume.pdf").exists()
    result = run_pipeline.revalidate(run_dir, console=False)
    assert result.status == "success", result.issues


# ============================================================ AD. cover-letter
# distinctive priorities: choosing the MOST DISTINCTIVE supported evidence.
#
# Every letter under review was true and still led with the wrong evidence: a
# forward-deployed posting got a backend paragraph. These tests pin the
# selection behaviour AND the scope freeze around it - the resume engine must
# be untouched by all of it.

# The resume decisions recorded from the four real-JD mocks BEFORE this patch.
# Cover-letter evidence selection may not move any of them.
PRE_PATCH_RESUME_DECISIONS = {
    "BAE_Systems_Entry_Level_Software_Engineer": {
        "section_order": ["Experience", "Education", "Academic Projects",
                          "Technical Skills"],
        "shipped_ids": ["GSD-1", "GSD-2", "GSD-3", "SWE-1", "SWE-2", "SWE-3",
                        "SWE-ALT-DOCKER", "SWE-5", "CLOUD-1", "CLOUD-2", "DATA-1"],
        "selected_by_relevance": ["lms", "pintos", "tailor_pipeline"],
        "display_order": ["tailor_pipeline", "lms", "pintos"],
        "allocation": {"lms": 3, "pintos": 2, "tailor_pipeline": 2},
        "pages": 1,
    },
    "C3_AI_Forward_Deployed_Engineer": {
        "section_order": ["Education", "Experience", "Academic Projects",
                          "Technical Skills"],
        "shipped_ids": ["GSD-1", "GSD-2", "GSD-3", "SWE-1", "SWE-2", "SWE-3",
                        "SWE-ALT-DOCKER", "SWE-5", "CLOUD-1", "CLOUD-2", "DATA-1"],
        "selected_by_relevance": ["lms", "pintos", "temp"],
        "display_order": ["lms", "pintos", "temp"],
        "allocation": {"lms": 3, "pintos": 2, "temp": 2},
        "pages": 1,
    },
    "Epic_Software_Developer": {
        "section_order": ["Experience", "Education", "Academic Projects",
                          "Technical Skills"],
        "shipped_ids": ["GSD-1", "GSD-2", "GSD-3", "SWE-1", "SWE-2", "SWE-3",
                        "SWE-4", "SWE-5", "CLOUD-1", "CLOUD-2", "DATA-1"],
        "selected_by_relevance": ["pintos", "lms", "fraud"],
        "display_order": ["lms", "pintos", "fraud"],
        "allocation": {"pintos": 3, "lms": 2, "fraud": 2},
        "pages": 1,
    },
    "Superhuman_Software_Engineer_Early_Career": {
        "section_order": ["Education", "Experience", "Academic Projects",
                          "Technical Skills"],
        "shipped_ids": ["GSD-1", "GSD-2", "GSD-3", "SWE-1", "SWE-2", "SWE-3",
                        "SWE-ALT-DOCKER", "SWE-5", "CLOUD-1", "CLOUD-2", "DATA-1"],
        "selected_by_relevance": ["lms", "temp", "pintos"],
        "display_order": ["lms", "pintos", "temp"],
        "allocation": {"lms": 3, "temp": 2, "pintos": 2},
        "pages": 1,
    },
}


def _priorities_for(jd_path: Path, chosen_ids: list[str]) -> list[dict]:
    """Derive the letter priorities exactly as run_pipeline does."""
    master = engine.load_master()
    jd = engine.read_jd(jd_path)
    requirements = engine.extract_jd_requirements(jd.text, master)
    projects = [p for p in master.projects if p.project_id in chosen_ids]
    capsules = engine.source_capsules(master, projects)
    return grounding.letter_priorities(
        jd_text=jd.text, job_title=jd.job_title or "", requirements=requirements,
        capsules=capsules, master=master, on_resume=chosen_ids)


# ---- A. the resume engine is not part of this feature ----------------------

def test_a_the_priority_feature_never_reaches_the_resume_engine():
    """Scope freeze: evidence SELECTION lives outside the frozen modules."""
    frozen = {"resume_engine.py": engine, "pdf_utils.py": pdf}
    leaked = ("letter_priorities", "DISTINCTIVE_THEMES", "cover_letter_score",
              "priority_addressed", "letter_relevance", "MOCK_THEME_PARAGRAPHS")
    for name, module in frozen.items():
        source = Path(module.__file__).read_text(encoding="utf-8")
        for symbol in leaked:
            assert symbol not in source, f"{symbol} leaked into {name}"
        for symbol in leaked:
            assert not hasattr(module, symbol), f"{name} exposes {symbol}"


def test_a_priority_derivation_mutates_none_of_its_inputs():
    """Read-only over requirements and capsules, so no resume input can shift."""
    master = engine.load_master()
    jd = engine.read_jd(C3_JD)
    requirements = engine.extract_jd_requirements(jd.text, master)
    capsules = engine.source_capsules(master, list(master.projects)[:3])
    before_reqs = [dataclasses.asdict(r) for r in requirements]
    before_caps = dict(capsules)
    grounding.letter_priorities(jd_text=jd.text, job_title=jd.job_title,
                               requirements=requirements, capsules=capsules,
                               master=master, on_resume=["lms"])
    assert [dataclasses.asdict(r) for r in requirements] == before_reqs
    assert capsules == before_caps


# ---- B. the four real JDs produce the same resume as before -----------------

@pytest.mark.parametrize("folder", sorted(PRE_PATCH_RESUME_DECISIONS))
def test_b_real_jd_resume_decisions_are_unchanged(folder):
    """Section order, Experience IDs, projects, allocation, Skills, pages."""
    candidates = sorted((ROOT / "output" / "_smoke_tests").glob(folder + "_*"))
    if not candidates:
        pytest.skip(f"no recorded mock run for {folder}; run the four real-JD mocks")
    strategy = json.loads((candidates[-1] / "strategy.json").read_text(encoding="utf-8"))
    expected = PRE_PATCH_RESUME_DECISIONS[folder]

    assert strategy["section_order"] == expected["section_order"]
    assert strategy["experience"]["shipped_ids"] == expected["shipped_ids"]
    assert strategy["projects"]["selected_by_relevance"] == expected["selected_by_relevance"]
    assert strategy["projects"]["display_order"] == expected["display_order"]
    assert strategy["projects"]["allocation"] == expected["allocation"]
    assert strategy["verification"]["pages"] == expected["pages"]
    # Skills are compared as the rendered lines, which is what the PDF shows.
    assert len(strategy["skills"]["rendered"]) == 5
    assert strategy["skills"]["rendered"][0].startswith("Languages & Databases:")


# ---- C. a badly prioritized C3 letter scores below a well prioritized one ---

_C3_CHOSEN = ["lms", "pintos", "temp"]

_C3_GROUNDED_BUT_MISPRIORITIZED = """Dear Hiring Manager,

I am applying for the Forward Deployed Engineer role at C3 AI. My background is
Python backend services and databases.

At Thesis Mumbai Tech I built healthcare modules supporting 10k+ records and
designed a real-time WebSocket and Redis pipeline feeding IoT sensor data into
PostgreSQL, and I architected the relational schemas and SQL queries behind
those services with indexing and query tuning to keep reads predictable.

Outside work I implemented the user-programs layer of Pintos, an x86 teaching
kernel written in C, covering process execution and a system-call handler.

Sincerely,
Jay Niketan Pathare"""

_C3_WELL_PRIORITIZED = """Dear Hiring Manager,

I am applying for the Forward Deployed Engineer role at C3 AI. My background is
Python backend services delivered directly with the people who use them.

At Thesis Mumbai Tech I led platform module delivery end to end, gathering
requirements in client meetings, clarifying scope with stakeholders, building
prototypes to validate it and translating the result into technical
specifications the team could implement.

Outside work I implemented the user-programs layer of Pintos, an x86 teaching
kernel written in C, covering process execution and a system-call handler.

Sincerely,
Jay Niketan Pathare"""


def test_c_c3_primary_priority_is_customer_facing_delivery():
    """The Forward Deployed title and the JD's demos/training make it primary."""
    priorities = _priorities_for(C3_JD, _C3_CHOSEN)
    top = grounding.supported_priorities(priorities)[0]
    assert top["theme"] == "customer_facing"
    assert top["source"] == "role:SWE"
    assert any("client" in term or "requirements" in term
               for term in top["evidence_terms"]), top["evidence_terms"]


def test_c_ignoring_customer_facing_evidence_costs_relevance_and_score():
    priorities = _priorities_for(C3_JD, _C3_CHOSEN)
    bad = grounding.letter_relevance(_C3_GROUNDED_BUT_MISPRIORITIZED, priorities)
    good = grounding.letter_relevance(_C3_WELL_PRIORITIZED, priorities)

    assert bad and bad[0].kind == "relevance"
    assert bad[0].severity == "warning", "a true letter is never rejected for this"
    assert "customer-facing delivery" in bad[0].message
    assert good == []

    common = dict(problems=[], priorities=priorities, themes_covered=2,
                  evidence_sources=2, named_terms_covered=2,
                  company="C3 AI", job_title="Forward Deployed Engineer")
    bad_score = grounding.cover_letter_score(
        letter=_C3_GROUNDED_BUT_MISPRIORITIZED,
        words=grounding.word_count(_C3_GROUNDED_BUT_MISPRIORITIZED), **common)
    good_score = grounding.cover_letter_score(
        letter=_C3_WELL_PRIORITIZED,
        words=grounding.word_count(_C3_WELL_PRIORITIZED), **common)
    assert good_score > bad_score, (good_score, bad_score)
    assert bad_score < 9.5, "ignoring the defining requirement cannot score 9.5"


# ---- D. Superhuman prefers the AI-native evidence --------------------------

_SUPERHUMAN_CHOSEN = ["lms", "temp", "pintos"]

_SUPERHUMAN_BACKEND_ONLY = """Dear Hiring Manager,

I am applying for the Software Engineer, Early Career role at Superhuman.

At Thesis Mumbai Tech I designed a real-time WebSocket and Redis pipeline
feeding IoT sensor data into PostgreSQL, and architected the relational schemas
and SQL queries behind those services.

Outside work I built a decoupled learning management system whose real-time
layer runs on Django Channels, Daphne, Redis and WebSockets.

Sincerely,
Jay Niketan Pathare"""

_SUPERHUMAN_AI_NATIVE = """Dear Hiring Manager,

I am applying for the Software Engineer, Early Career role at Superhuman.

At Thesis Mumbai Tech I designed a real-time WebSocket and Redis pipeline
feeding IoT sensor data into PostgreSQL, and architected the relational schemas
and SQL queries behind those services.

I work with AI coding agents day to day rather than around them: GitHub Copilot
for code generation and completion, and Cursor and Claude Code for
repository-level development, debugging, test-driven iteration and code review.

Sincerely,
Jay Niketan Pathare"""


def test_d_superhuman_primary_priority_is_ai_native_development():
    priorities = _priorities_for(REDWOOD_JD, _SUPERHUMAN_CHOSEN)
    top = grounding.supported_priorities(priorities)[0]
    assert top["theme"] == "ai_native"
    # The supporting evidence is the skills bank, not an invented project.
    assert top["source"] == "general"
    assert "claude code" in top["evidence_terms"]
    assert "cursor" in top["evidence_terms"]


def test_d_ai_native_evidence_outranks_an_equally_grounded_backend_letter():
    priorities = _priorities_for(REDWOOD_JD, _SUPERHUMAN_CHOSEN)
    assert grounding.letter_relevance(_SUPERHUMAN_AI_NATIVE, priorities) == []
    assert grounding.letter_relevance(_SUPERHUMAN_BACKEND_ONLY, priorities)

    common = dict(problems=[], priorities=priorities, themes_covered=2,
                  evidence_sources=2, named_terms_covered=1,
                  company="Superhuman", job_title="Software Engineer, Early Career")
    ai = grounding.cover_letter_score(
        letter=_SUPERHUMAN_AI_NATIVE,
        words=grounding.word_count(_SUPERHUMAN_AI_NATIVE), **common)
    backend = grounding.cover_letter_score(
        letter=_SUPERHUMAN_BACKEND_ONLY,
        words=grounding.word_count(_SUPERHUMAN_BACKEND_ONLY), **common)
    assert ai > backend, (ai, backend)


def test_d_epic_still_leads_with_healthcare_and_bae_invents_nothing():
    """The other two real JDs keep their existing, correct emphasis."""
    epic_top = grounding.supported_priorities(
        _priorities_for(EPIC_JD, ["pintos", "lms", "fraud"]))[0]
    assert epic_top["theme"] == "healthcare"
    assert epic_top["source"] == "role:SWE"

    bae = _priorities_for(BAE_JD, ["lms", "pintos", "tailor_pipeline"])
    labels = " ".join(p["label"] for p in bae).lower()
    evidence = " ".join(term for p in bae for term in p["evidence_terms"]).lower()
    # None of the things the BAE posting must never be answered with.
    for invented in ("clearance", "computer vision", "profiling", "windows",
                     "java", "c++", "javascript"):
        assert invented not in labels and invented not in evidence, invented


# ---- E. an unsupported distinctive requirement stays uncovered -------------

def test_e_an_unsupported_distinctive_signal_is_never_padded(tmp_path):
    """No capsule evidence: the signal is named, then left alone."""
    master = engine.load_master()
    jd_text = ("Senior Kernel Engineer\n\nWe build device drivers and work on kernel "
               "memory management and thread synchronization in the Linux kernel every "
               "day. Systems programming and concurrency are the whole job.\n")
    requirements = engine.extract_jd_requirements(jd_text, master)
    # A capsule pool that supports nothing low-level at all.
    capsules = {"role:SWE": "Built REST APIs with Django and PostgreSQL."}
    priorities = grounding.letter_priorities(
        jd_text=jd_text, job_title="Senior Kernel Engineer", requirements=requirements,
        capsules=capsules, master=master, on_resume=[])

    assert any(p["theme"] == "low_level_systems" for p in priorities)
    low_level = next(p for p in priorities if p["theme"] == "low_level_systems")
    assert low_level["supported"] is False
    assert low_level["source"] == "" and low_level["evidence_terms"] == []
    # It reaches neither the prompt nor the relevance check, so nothing is forced.
    assert "low-level" not in grounding.render_letter_priorities(priorities)
    assert grounding.letter_relevance("Dear Hiring Manager, I write Python.",
                                      priorities) == []


def test_e_a_posting_with_nothing_distinctive_yields_no_priorities():
    master = engine.load_master()
    jd_text = ("Software Engineer\n\nYou will write Python, use Git, build REST APIs "
               "and communicate well with your team in an Agile environment.\n")
    requirements = engine.extract_jd_requirements(jd_text, master)
    priorities = grounding.letter_priorities(
        jd_text=jd_text, job_title="Software Engineer", requirements=requirements,
        capsules=engine.source_capsules(master, []), master=master)
    assert priorities == []
    assert grounding.render_letter_priorities(priorities) == ""
    # A generic posting is not punished for being generic.
    assert grounding.cover_letter_score(
        letter="Dear Hiring Manager, I build Python services. Sincerely, Jay",
        problems=[], words=200, priorities=priorities, themes_covered=2,
        evidence_sources=2, named_terms_covered=1, company="", job_title="") >= 8.0


# ---- F. generic keywords never drive any of this ---------------------------

def test_f_generic_cues_can_never_become_a_priority():
    """Python, Git, REST and communication describe every posting."""
    for theme in grounding.DISTINCTIVE_THEMES:
        overlap = grounding.GENERIC_CUES & set(theme.jd_cues)
        assert not overlap, (theme.key, overlap)


def test_f_missing_jd_keywords_alone_is_not_a_relevance_failure():
    """The check fires on ignored distinctive evidence, not absent keywords."""
    priorities = _priorities_for(C3_JD, _C3_CHOSEN)
    # Names no JD technology at all, but does address the primary priority.
    letter = ("Dear Hiring Manager,\n\nI am applying for the Forward Deployed Engineer "
              "role at C3 AI.\n\nAt Thesis Mumbai Tech I gathered requirements in client "
              "meetings, built prototypes to validate scope and owned delivery end to "
              "end.\n\nSincerely,\nJay Niketan Pathare")
    assert grounding.letter_named_terms(letter, ["Kubernetes", "Terraform", "Go"]) == []
    assert grounding.letter_relevance(letter, priorities) == []


def test_f_relevance_is_advisory_and_generation_validation_is_untouched():
    """A relevance warning can lower a score; it can never block a letter."""
    priorities = _priorities_for(C3_JD, _C3_CHOSEN)
    problems = grounding.letter_relevance(_C3_GROUNDED_BUT_MISPRIORITIZED, priorities)
    assert grounding.errors(problems) == []
    import llm_client

    source = Path(llm_client.__file__).read_text(encoding="utf-8")
    loop = source.split("def cover_letter(")[1].split("def assess(")[0]
    # Relevance only retries while attempts remain; it never reaches `blocking`.
    assert "relevance and attempt < LETTER_ATTEMPTS" in loop
    assert "return letter, problems + relevance" in loop


# ---- G. the compact score reflects the agreed rubric -----------------------

def test_g_blocking_validation_still_short_circuits_the_score():
    problems = [grounding.Problem("metric", "error", "unsupported metric")]
    assert grounding.cover_letter_score(
        letter="x", problems=problems, words=200, priorities=[],
        themes_covered=2, evidence_sources=2, named_terms_covered=1,
        company="C3 AI", job_title="Forward Deployed Engineer") == 4.0


def test_g_a_grounding_warning_costs_more_than_a_structure_warning():
    priorities = _priorities_for(C3_JD, _C3_CHOSEN)
    common = dict(letter=_C3_WELL_PRIORITIZED, words=200, priorities=priorities,
                  themes_covered=2, evidence_sources=2, named_terms_covered=2,
                  company="C3 AI", job_title="Forward Deployed Engineer")
    clean = grounding.cover_letter_score(problems=[], **common)
    structural = grounding.cover_letter_score(
        problems=[grounding.Problem("structure", "warning", "three paragraphs")], **common)
    factual = grounding.cover_letter_score(
        problems=[grounding.Problem("source_scope", "warning", "mixed sources")], **common)
    assert clean > structural > factual, (clean, structural, factual)


def test_g_the_letter_score_reaches_the_report_unchanged():
    """The role-distinctive rubric survives the audit layer intact."""
    priorities = _priorities_for(C3_JD, _C3_CHOSEN)
    score = grounding.cover_letter_score(
        letter=_C3_WELL_PRIORITIZED, problems=[], words=200, priorities=priorities,
        themes_covered=2, evidence_sources=2, named_terms_covered=2,
        company="C3 AI", job_title="Forward Deployed Engineer")
    report = grounding.audit_report(letter_score=score)
    assert list(report) == list(grounding.ASSESSMENT_FIELDS)
    assert report["Cover Letter Score"] == f"{score:.1f} / 10"
    # Only the deterministic field is populated; the rest await their audits.
    assert report["Company STEM OPT Support"] == "UNKNOWN"
    assert report["Company Visa Sponsorship"] == "UNKNOWN"
    assert report["Callback Likelihood"] == "UNKNOWN"


def test_g_no_immigration_language_is_injected_by_the_priority_block():
    """Priorities steer evidence only; visa/OPT wording is never added."""
    for jd_path, chosen in ((C3_JD, _C3_CHOSEN), (EPIC_JD, ["pintos", "lms", "fraud"]),
                            (REDWOOD_JD, _SUPERHUMAN_CHOSEN),
                            (BAE_JD, ["lms", "pintos", "tailor_pipeline"])):
        block = grounding.render_letter_priorities(_priorities_for(jd_path, chosen)).lower()
        for term in ("visa", "sponsor", "opt", "ead", "e-verify", "immigration"):
            # Whole words only: "optimization" is not an OPT reference.
            assert not re.search(r"(?<![a-z])" + term + r"(?![a-z])", block), \
                (jd_path.name, term)


# ================================================== AE. post-run Groq audit
#
# Three Groq calls run after the resume and cover letter are final. They are
# advisory: nothing they return may re-enter generation, none of them may fall
# back to Gemini, and only the company-research call may browse.

AUDIT_PURPOSES = ("assessment", "company_research")


class _RecordingTransport:
    """Records every Request and answers from a per-purpose payload map."""

    name = "recording"

    def __init__(self, payloads: dict[str, object], *, fail: tuple[str, ...] = ()):
        self.payloads = payloads
        self.fail = fail
        self.requests: list = []

    def generate(self, request):
        import llm_client

        self.requests.append(request)
        if request.purpose in self.fail:
            raise llm_client.ProviderError("server_error", f"{request.purpose} is down")
        payload = self.payloads.get(request.purpose, {})
        text = payload if isinstance(payload, str) else json.dumps(payload)
        return llm_client.Reply(text, self.name)


def _audit_client(run, payloads, *, fail=()):
    import logging

    import llm_client

    transport = _RecordingTransport(payloads, fail=fail)
    client = llm_client.LLMClient(transport, transport, run["master"], run["policy"],
                                  logging.getLogger("test"), audit=transport)
    return transport, client


def _experience_audit_payload(score: float = 8.5, **overrides) -> dict:
    payload = {
        "experience_selection_score": score,
        "components": {"signal_interpretation": score, "policy_choice": score,
                       "evidence_relevance": score},
        "verdict": "PASS" if score >= 7.0 else "REVIEW",
        "missed_signals": [],
        "better_permitted_choice_exists": False,
        "explanation": "the applied rule was the right one",
    }
    payload.update(overrides)
    return payload


# ---- transports: audits are Groq or UNKNOWN -------------------------------

def test_the_audit_transport_has_no_gemini_fallback():
    """build_client gives audits their own Groq transport with fallback=None."""
    import llm_client

    import logging
    credentials = engine.Credentials(gemini_accounts=(1,), has_groq=True,
                                    _gemini_keys={1: "k1"}, _groq_key="k2")
    client = llm_client.build_client(engine.load_master(), engine.load_policy(),
                                     logging.getLogger("test"), mock=False,
                                     credentials=credentials)
    # Two distinct Groq models, neither able to reach Gemini.
    assert client.audit is not client.research
    assert client.audit.fallback is None, "an audit may never fail over to Gemini"
    assert client.research.fallback is None, "research may never fail over to Gemini"
    assert client.audit.model == "qwen/qwen3.8-27b"
    assert client.research.model == "openai/gpt-oss-20b"
    # Generation is Gemini now, so there is no Groq generation transport left.
    assert isinstance(client.gemini, llm_client.GeminiTransport)


def test_every_audit_call_uses_the_audit_transport():
    """Source contract: the three audit purposes invoke self.audit, not self.groq."""
    source = Path(_llm_client().__file__).read_text(encoding="utf-8")
    expected = {"assessment": "self.audit", "company_research": "self.research"}
    for purpose, transport in expected.items():
        head = source.split(f'"{purpose}", ')[0]
        invoke = head.rsplit("self._invoke(", 1)[1].split(",")[0].strip()
        assert invoke == transport, f"{purpose} must be invoked on {transport}"
    # Generation is Gemini-only: the cover letter included. Selection goes
    # through _call (which wraps _invoke), so both spellings are accepted.
    for purpose in ("project_selection", "project_bullets", "cover_letter"):
        head = source.split(f'"{purpose}", ')[0]
        cut = max(head.rfind("self._invoke("), head.rfind("self._call("))
        invoke = head[cut:].split("(", 1)[1].split(",")[0].strip()
        assert invoke == "self.gemini", f"{purpose} must go to Gemini, got {invoke}"


def _llm_client():
    import llm_client

    return llm_client


def test_an_unavailable_groq_audit_transport_raises_instead_of_using_gemini():
    import logging

    import llm_client

    credentials = engine.Credentials(gemini_accounts=(1,), has_groq=False,
                                    _gemini_keys={1: "k1"})
    audit = llm_client.GroqTransport(credentials, logging.getLogger("test"), fallback=None)
    with pytest.raises(llm_client.ProviderError) as raised:
        audit.generate(llm_client.Request("assessment", "prompt"))
    assert raised.value.category == "auth_permission"


# ---- only company research may browse -------------------------------------
def test_browser_search_is_forced_and_only_for_web_search_requests():
    """tool_choice="required" stops the model answering from stale memory."""
    source = Path(_llm_client().__file__).read_text(encoding="utf-8")
    generate = source.split("def _generate(self, request: Request) -> str:")[1]
    generate = generate.split("# ================")[0]
    block = generate.split("if request.web_search and not self.no_web_search:")[1]
    block = block.split("try:")[0]
    assert '"tools"' in block and "browser_search" in block
    assert 'settings["tool_choice"] = "required"' in block
    # Nothing outside that guard may set either field.
    assert generate.count('settings["tool_choice"]') == 1
    assert generate.count('settings["tools"]') == 1


def test_no_generation_request_ever_asks_for_web_search():
    source = Path(_llm_client().__file__).read_text(encoding="utf-8")
    for purpose in ("project_selection", "project_bullets", "bullet_repair", "cover_letter"):
        head = source.split(f'"{purpose}", ')[1].split("))")[0]
        assert "web_search" not in head, f"{purpose} must never browse"


def test_unsupported_browser_search_degrades_research_to_unknown():
    """A rejected tool retries once without browsing, then reports UNKNOWN."""
    import logging

    import llm_client

    calls: list[bool] = []

    class _Rejecting(llm_client.GroqTransport):
        def _generate(self, request):
            calls.append(request.web_search and not self.no_web_search)
            if request.web_search and not self.no_web_search:
                raise llm_client.ProviderError("bad_request", "tools are not supported")
            return json.dumps({"company_visa_sponsorship": "UNKNOWN",
                               "company_stem_opt_support": "UNKNOWN",
                               "job_posted": "UNKNOWN", "checked_at": "2026-09-15",
                               "sources": []})

    credentials = engine.Credentials(has_groq=True, _groq_key="k")
    transport = _Rejecting(credentials, logging.getLogger("test"), model="m", fallback=None)
    reply = transport.generate(llm_client.Request("company_research", "p", web_search=True))
    assert calls == [True, False], calls
    assert transport.no_web_search is True
    research, _ = grounding.validate_company_research(json.loads(reply.text))
    assert research["company_visa_sponsorship"] == "UNKNOWN"
    assert research["job_posted"] == "UNKNOWN"


# ---- audit results never re-enter generation ------------------------------

def c3_signals(c3):
    return engine.classify_jd(c3["jd"].text, c3["master"].section_order)

def test_audit_results_are_never_read_back_into_generation():
    """Source contract: no generation path reads an audit value."""
    source = Path(run_pipeline.__file__).read_text(encoding="utf-8")
    # Everything before the audit layer runs is generation. It cannot mention
    # the audit names, because they do not exist yet.
    run_one = source.split("def run_one(")[1]
    generation = run_one.split("audits = run_audits(")[0]
    for name in ("experience_selection_score", "better_permitted_choice_exists",
                 "project_selection_score", "project_bullet_score",
                 "callback_likelihood", "company_visa_sponsorship"):
        assert name not in generation, f"{name} reached generation"
    # And after the audits, only reporting happens: no regeneration entry point.
    after = run_one.split("audits = run_audits(")[1].split("def assemble(")[0]
    for forbidden in ("client.select_projects", "client.write_bullets",
                      "client.cover_letter", "engine.select_experience",
                      "fit_to_page(", "pdf.compile_pdf"):
        assert forbidden not in after, f"{forbidden} runs after the audit layer"


def test_a_low_application_audit_score_does_not_regenerate_anything(run, c3):
    """2.0s across the board still produce a report, never a rewrite."""
    signals = c3_signals(c3)
    requirements = c3["requirements"]
    ids = [r.requirement_id for r in engine.scored_requirements(requirements)]
    payload = {
        "fit_score": 2.0, "recommendation": "skip",
        "summary": "Few requirements are evidenced.",
        "eligibility": {"status": "uncertain", "details": []},
        "strong_matches": [], "partial_matches": [],
        "gaps": [{"requirement_id": ids[0], "importance": "required",
                  "status": "unsupported", "detail": "no evidence", "evidence": None}],
        "manual_review": [], "complementary_strengths": [],
        "tailoring_quality": {"score": 2.0, "notes": ["weak"]},
        "project_selection": {"score": 1.5, "components": {
            "jd_relevance": 1.0, "best_available_chosen": 2.0,
            "complementary_coverage": 1.0, "ranking_and_allocation": 2.0},
            "notes": ["fraud overlapped more than lms"]},
        "project_bullets": {"score": 2.5, "components": {
            "jd_relevance": 2.0, "technical_specificity": 3.0, "evidence_fidelity": 2.0,
            "impact_ownership": 3.0, "non_redundancy": 3.0}, "notes": ["thin"]},
        "experience_selection": {"score": 3.0, "notes": ["the rule misreads the posting"]},
        "callback_likelihood": "LOW", "risk_flags": [],
    }
    audit, problems = grounding.validate_application_audit(payload)
    assert audit["project_selection_score"] == 1.5
    assert audit["project_bullet_score"] == 2.5
    assert audit["callback_likelihood"] == "LOW"
    assert audit["experience_selection_score"] == 3.0
    assert problems == []
    # The report renders the low scores and stops there.
    report = grounding.audit_report(application_audit={**audit, "fit_score": 2.0},
                                    letter_score=9.5)
    assert report["Project Selection Score"] == "1.5 / 10"
    assert report["Callback Likelihood"] == "LOW"


# ---- the audits see the COMPLETE option space -----------------------------
def test_the_project_audit_sees_every_candidate_project(run, c3):
    import llm_client

    master = run["master"]
    selection = llm_client.Selection(selected=["lms", "pintos", "temp"],
                                     ranks={"lms": 1, "pintos": 2, "temp": 3},
                                     reasons={"lms": "r1", "pintos": "r2", "temp": "r3"},
                                     considered=[])
    chosen = [master.project(p) for p in selection.selected]
    bullets = {"lms": ["b1"], "pintos": ["b2"], "temp": ["b3"]}
    catalogue = run_pipeline.project_catalogue(master, selection, chosen, bullets)

    assert {e["project_id"] for e in catalogue} == {p.project_id for p in master.projects}
    assert len(catalogue) > len(selection.selected), "the audit needs the whole catalogue"
    assert sum(1 for e in catalogue if e["selected"]) == 3
    # Passed-over projects arrive with the evidence needed to second-guess.
    for entry in catalogue:
        if not entry["selected"]:
            assert entry["evidence"], entry["project_id"]
            assert entry["tech"], entry["project_id"]


def test_the_bullet_audit_sees_each_project_own_evidence(run, c3):
    import llm_client

    master = run["master"]
    selection = llm_client.Selection(selected=["lms", "pintos", "temp"], ranks={},
                                     reasons={}, considered=[])
    chosen = [master.project(p) for p in selection.selected]
    bullets = {"lms": ["Implemented the real-time subsystem with Django Channels."],
               "pintos": ["Engineered the user-programs layer of Pintos in C."],
               "temp": ["Built an IoT monitoring backend in Django REST Framework."]}
    catalogue = run_pipeline.project_catalogue(master, selection, chosen, bullets)
    by_id = {e["project_id"]: e for e in catalogue}
    for pid, written in bullets.items():
        assert by_id[pid]["final_bullets"] == written
        assert by_id[pid]["evidence"] == " | ".join(master.project(pid).evidence)
    # Pintos evidence must not leak into the LMS entry, or fidelity checks lie.
    assert "Pintos" not in by_id["lms"]["evidence"]


# ---- failure isolation ----------------------------------------------------
def _selection(ids):
    import llm_client

    return llm_client.Selection(selected=list(ids), ranks={p: i + 1 for i, p in enumerate(ids)},
                                reasons={p: "reason" for p in ids}, considered=[])


def test_a_failed_audit_leaves_a_successful_run_successful():
    """Status is decided by verification and the letter, never by an audit."""
    source = Path(run_pipeline.__file__).read_text(encoding="utf-8")
    block = source.split("audits = run_audits(")[1].split('status = "success"')[0]
    # The audit errors are logged as warnings and never appended to `issues`.
    assert 'audits["errors"]' in block
    assert 'issues.append(f"audit' not in block
    assert 'issues += [f"audit' not in block
    assert 'status = "success" if not issues else "needs_review"' in source, \
        "status must still be computed from issues alone"


# ---- company research rules -----------------------------------------------

def test_everify_alone_can_never_produce_stem_opt_yes():
    data = {
        "company_visa_sponsorship": "YES", "company_visa_confidence": "HIGH",
        "company_stem_opt_support": "YES", "company_stem_opt_confidence": "HIGH",
        "job_posted": "2026-09-01", "job_posted_confidence": "HIGH",
        "checked_at": "2026-09-15",
        "sources": [{"title": "Careers FAQ", "url": "https://example.com/faq",
                     "scope": "company_policy",
                     "evidence": "The company is an E-Verify participating employer."}],
    }
    research, _ = grounding.validate_company_research(data, jd_text="We are hiring.")
    assert research["company_stem_opt_support"] == "UNKNOWN"
    assert research["company_stem_opt_confidence"] == "LOW"
    assert any("E-Verify" in note for note in research["deterministic_overrides"])
    # Real STEM OPT evidence is still allowed through.
    data["sources"].append({"title": "Immigration policy", "url": "https://example.com/i",
                            "scope": "company_policy",
                            "evidence": "We support the STEM OPT 24-month extension and "
                                        "sign the I-983 training plan."})
    data["company_stem_opt_support"] = "YES"
    research, _ = grounding.validate_company_research(data, jd_text="We are hiring.")
    assert research["company_stem_opt_support"] == "YES"


def test_historical_h1b_alone_lowers_sponsorship_confidence():
    data = {
        "company_visa_sponsorship": "YES", "company_visa_confidence": "HIGH",
        "company_stem_opt_support": "UNKNOWN", "company_stem_opt_confidence": "LOW",
        "job_posted": "UNKNOWN", "job_posted_confidence": "LOW",
        "checked_at": "2026-09-15",
        "sources": [{"title": "H-1B disclosure data", "url": "https://example.gov/lca",
                     "scope": "company_policy",
                     "evidence": "14 H-1B LCA filings were certified in 2024."}],
    }
    research, _ = grounding.validate_company_research(data, jd_text="We are hiring.")
    assert research["company_visa_sponsorship"] == "YES"
    assert research["company_visa_confidence"] == "MEDIUM"
    assert any("H-1B" in note for note in research["deterministic_overrides"])


def test_a_job_level_no_sponsorship_restriction_overrides_company_evidence():
    data = {
        "company_visa_sponsorship": "YES", "company_visa_confidence": "HIGH",
        "company_stem_opt_support": "YES", "company_stem_opt_confidence": "HIGH",
        "job_posted": "2026-09-01", "job_posted_confidence": "HIGH",
        "checked_at": "2026-09-15",
        "sources": [{"title": "Careers", "url": "https://example.com",
                     "scope": "company_policy",
                     "evidence": "We sponsor visas for engineering roles."}],
    }
    jd_text = ("Software Engineer\n\nWe do not offer visa sponsorship for this position.\n")
    assert grounding._explicit(jd_text, grounding._SPONSOR_YES,
                               grounding._SPONSOR_NO) == "NO"
    research, _ = grounding.validate_company_research(data, jd_text=jd_text)
    assert research["company_visa_sponsorship"] == "NO"
    assert research["company_visa_confidence"] == "HIGH"
    assert research["company_stem_opt_support"] == "UNKNOWN"
    assert any("overrides" in note for note in research["deterministic_overrides"])


def test_research_never_substitutes_today_for_the_posting_date():
    today = engine.date.today().isoformat()
    research, _ = grounding.validate_company_research(
        {"job_posted": "", "checked_at": today, "sources": []}, jd_text="We are hiring.")
    assert research["job_posted"] == "UNKNOWN"
    # A trustworthy date in the posting itself outranks a researched one.
    research, _ = grounding.validate_company_research(
        {"job_posted": "2020-01-01", "checked_at": today, "sources": []},
        jd_text="We are hiring.", jd_posted="2026-09-01")
    assert research["job_posted"] == "2026-09-01"
    assert research["job_posted_confidence"] == "HIGH"


# ---- the report's field contract ------------------------------------------

def test_callback_likelihood_appears_exactly_once_and_the_old_fields_are_gone():
    report = grounding.audit_report(
        application_audit={"callback_likelihood": "HIGH", "fit_score": 8.0},
        letter_score=9.5)
    rendered = grounding.render_assessment_txt(report)
    assert rendered.count("Callback Likelihood") == 1
    assert "Can Expect Callback" not in rendered
    assert "Callback Confidence" not in rendered
    assert "Resume Score:" not in rendered
    assert list(report) == list(grounding.ASSESSMENT_FIELDS)
    assert len(grounding.ASSESSMENT_FIELDS) == 10
    for retired in grounding.RETIRED_ASSESSMENT_FIELDS:
        assert retired not in grounding.ASSESSMENT_FIELDS


def test_the_retired_callback_fields_are_gone_from_the_whole_pipeline():
    for module in (grounding, run_pipeline, _llm_client()):
        source = Path(module.__file__).read_text(encoding="utf-8")
        body = source.replace("RETIRED_ASSESSMENT_FIELDS", "")
        # The only surviving mention is the retired-names tuple itself.
        for retired in ("Can Expect Callback", "Callback Confidence"):
            occurrences = body.count(retired)
            allowed = 1 if module is grounding else 0
            assert occurrences <= allowed, f"{retired} still in {module.__name__}"


def test_the_terminal_block_matches_the_agreed_shape():
    report = grounding.audit_report(
        application_audit={"experience_selection_score": 8.7,
                           "resume_tailoring_score": 9.5, "project_selection_score": 8.5,
                           "project_bullet_score": 8.5, "callback_likelihood": "MEDIUM",
                           "fit_score": 6.7},
        research={"company_visa_sponsorship": "UNKNOWN",
                  "company_stem_opt_support": "UNKNOWN", "job_posted": "UNKNOWN"},
        letter_score=9.5)
    lines = grounding.render_assessment_terminal(report, company="C3 AI",
                                                 job_title="Forward Deployed Engineer")
    assert lines[0] == "[ASSESSMENT] C3 AI | Forward Deployed Engineer"
    assert lines[1] == "  Resume Tailoring Score: 9.5 / 10"
    assert lines[-1] == "  Job Posted: UNKNOWN"
    assert len(lines) == 11
    for line in lines[1:]:
        assert line.startswith("  "), line


def test_batch_prints_the_assessment_before_the_next_job():
    """run_one prints its own block, so it lands before the next [BATCH] line."""
    source = Path(run_pipeline.__file__).read_text(encoding="utf-8")
    assert "print_assessment(compact, company=" in source
    batch = source.split("def run_batch(")[1]
    assert '[BATCH] processing' in batch
    # The batch loop itself must not print an assessment, or it would appear twice.
    assert "print_assessment" not in batch.split("[BATCH] complete")[0]
    assert "[BATCH] complete" in batch, "the final batch summary must stay"


# ---- standalone --assess --------------------------------------------------

def _c3_smoke_folder() -> Path:
    folders = sorted((ROOT / "output" / "_smoke_tests").glob("C3_AI_*"))
    if not folders:
        pytest.skip("no C3 smoke folder; run the C3 mock first")
    return folders[-1]


def test_assess_makes_two_groq_calls_and_zero_gemini_calls():
    run_dir = _c3_smoke_folder()
    result = run_pipeline.assess_run(run_dir, mock=True, console=False)
    assert result.status == "success", result.issues
    calls = (result.strategy.get("audit") or {}).get("provider_calls") or []
    purposes = [c["purpose"] for c in calls]
    assert sorted(purposes) == sorted(AUDIT_PURPOSES), purposes
    assert len(calls) == 2
    assert all(c["provider"] != "gemini" for c in calls), calls


def test_assess_preserves_every_generation_artifact():
    import hashlib

    run_dir = _c3_smoke_folder()
    watched = ["resume.tex", "resume.txt", "cover_letter.txt", "job_description.txt"]
    watched += [p.name for p in run_dir.glob("*.pdf")]
    before = {name: hashlib.sha256((run_dir / name).read_bytes()).hexdigest()
              for name in watched if (run_dir / name).exists()}
    assert len(before) == 5, sorted(before)

    result = run_pipeline.assess_run(run_dir, mock=True, console=False)
    assert result.status == "success", result.issues
    after = {name: hashlib.sha256((run_dir / name).read_bytes()).hexdigest()
             for name in before}
    assert after == before


def test_assess_regenerates_assessment_txt_and_only_the_audit_block():
    run_dir = _c3_smoke_folder()
    strategy_before = json.loads((run_dir / "strategy.json").read_text(encoding="utf-8"))
    (run_dir / "assessment.txt").write_text("stale\n", encoding="utf-8")

    result = run_pipeline.assess_run(run_dir, mock=True, console=False)
    assert result.status == "success", result.issues
    rendered = (run_dir / "assessment.txt").read_text(encoding="utf-8")
    assert "stale" not in rendered
    assert rendered.splitlines()[0].startswith("Resume Tailoring Score:")
    assert len(rendered.strip().splitlines()) == 10

    strategy_after = json.loads((run_dir / "strategy.json").read_text(encoding="utf-8"))
    changed = {key for key in set(strategy_before) | set(strategy_after)
               if strategy_before.get(key) != strategy_after.get(key)}
    # Only the audit block may move. It can also be byte-identical when a
    # previous --assess in this session produced the same values.
    assert changed <= {"audit"}, changed
    assert strategy_after["audit"]["mode"] == "standalone --assess"
    assert strategy_after["audit"]["report"] == result.assessment
    assert set(strategy_after) >= set(strategy_before)


def test_assess_never_compiles_or_regenerates(tmp_path):
    source = Path(run_pipeline.__file__).read_text(encoding="utf-8")
    block = source.split("def assess_run(")[1].split("def revalidate(")[0]
    for forbidden in ("compile_pdf", "fit_to_page", "select_projects", "write_bullets",
                      "cover_letter(", "finalize_artifacts", "assemble("):
        assert forbidden not in block, f"--assess must not call {forbidden}"
    # Only two writes are permitted, plus its own log.
    writes = re.findall(r'\(run_dir / "([^"]+)"\)\.write_text', block)
    writes += re.findall(r'(\w+)_path\.write_text', block)
    assert set(writes) <= {"assessment.txt", "strategy"}, writes


def test_assess_never_touches_production_tracking():
    import hashlib

    source = Path(run_pipeline.__file__).read_text(encoding="utf-8")
    block = source.split("def assess_run(")[1].split("def revalidate(")[0]
    for forbidden in ("Tracker(", "PROCESSED_CSV", "PROCESSED_INDEX", "tracker.record"):
        assert forbidden not in block, f"--assess must not reference {forbidden}"

    tracking = [ROOT / "processed_jobs.csv", ROOT / ".processed_index.json"]
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in tracking if p.exists()}
    run_pipeline.assess_run(_c3_smoke_folder(), mock=True, console=False)
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in tracking if p.exists()}
    assert after == before, "--assess changed production tracking"


def test_assess_reconstructs_experience_without_reselecting_it():
    """The ids come from strategy.json; no swap rule is re-evaluated."""
    run_dir = _c3_smoke_folder()
    strategy = json.loads((run_dir / "strategy.json").read_text(encoding="utf-8"))
    policy = engine.load_policy()
    decision = run_pipeline._decision_from_strategy(strategy, policy)
    assert decision.shipped_ids == strategy["experience"]["shipped_ids"]
    assert decision.rule == strategy["experience"]["rule"]
    # Wording comes from the approved library by id, never from the model.
    by_id = {e.bullet_id: e for e in policy.experience_library}
    by_id.update({e.bullet_id: e for e in policy.alternate_library})
    for bullet_id, _action, latex in decision.shipped:
        if bullet_id in by_id:
            assert latex == by_id[bullet_id].latex


# ---- BullsAI stays dormant -------------------------------------------------

def test_no_bullsai_behaviour_is_added_by_the_audit_layer():
    """resume_engine.py is frozen, so its dormant field stays; nothing new."""
    for module in (grounding, run_pipeline, _llm_client()):
        source = Path(module.__file__).read_text(encoding="utf-8").lower()
        assert "bullsai" not in source, module.__name__
    # The dormant credential field is never wired to a transport.
    engine_source = Path(engine.__file__).read_text(encoding="utf-8")
    assert "BullsAITransport" not in engine_source
    assert "has_bullsai" in engine_source, "unchanged frozen file"


# ============================================ AF. GROQ_MODEL config loading
#
# resume_engine.GROQ_MODEL is bound at import time from os.environ only, while
# load_credentials() separately merges .env. A project whose model lives in
# .env alone therefore had a working key and an empty model, and Groq called
# itself unavailable. The model is resolved at call time instead.

def _credentials(groq: bool = True):
    return engine.Credentials(gemini_accounts=(1,), has_groq=groq,
                              _gemini_keys={1: "k1"},
                              _groq_key="k2" if groq else None)


def _build(monkeypatch, tmp_path, *, environ=None, dotenv=None):
    """build_client with a controlled process env and .env file."""
    import logging

    import llm_client

    monkeypatch.delenv("GROQ_MODEL", raising=False)
    if environ is not None:
        monkeypatch.setenv("GROQ_MODEL", environ)
    lines = [f"{key}={value}" for key, value in (dotenv or {}).items()]
    (tmp_path / ".env").write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setattr(engine, "PROJECT_ROOT", tmp_path)
    return llm_client.build_client(engine.load_master(), engine.load_policy(),
                                    logging.getLogger("test"), mock=False,
                                    credentials=_credentials())


def test_groq_model_comes_from_dotenv_when_the_environment_lacks_it(monkeypatch, tmp_path):
    """The reported bug: key found in .env, model constant left empty."""
    import llm_client

    monkeypatch.delenv("GROQ_MODEL", raising=False)
    (tmp_path / ".env").write_text(
        'GROQ_API_KEY=secret\nGROQ_MODEL=openai/gpt-oss-120b\n', encoding="utf-8")
    monkeypatch.setattr(engine, "PROJECT_ROOT", tmp_path)
    assert llm_client.configured_groq_model() == "openai/gpt-oss-120b"
    # The stale import-time constant is exactly what this works around.
    assert os.getenv("GROQ_MODEL") is None


def test_the_process_environment_overrides_dotenv(monkeypatch, tmp_path):
    import llm_client

    monkeypatch.setenv("GROQ_MODEL", "env/wins")
    (tmp_path / ".env").write_text("GROQ_MODEL=dotenv/loses\n", encoding="utf-8")
    monkeypatch.setattr(engine, "PROJECT_ROOT", tmp_path)
    assert llm_client.configured_groq_model() == "env/wins"


def test_surrounding_whitespace_is_stripped(monkeypatch, tmp_path):
    import llm_client

    monkeypatch.setenv("GROQ_MODEL", "  openai/gpt-oss-120b  ")
    monkeypatch.setattr(engine, "PROJECT_ROOT", tmp_path)
    (tmp_path / ".env").write_text("", encoding="utf-8")
    assert llm_client.configured_groq_model() == "openai/gpt-oss-120b"


def test_no_hardcoded_model_is_ever_restored(monkeypatch, tmp_path):
    """Neither source configured: empty, and Groq says exactly that."""
    import llm_client

    for name in ("GROQ_MODEL", "GROQ_ASSESSMENT_MODEL", "GROQ_RESEARCH_MODEL"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / ".env").write_text("GROQ_API_KEY=secret\n", encoding="utf-8")
    monkeypatch.setattr(engine, "PROJECT_ROOT", tmp_path)
    assert llm_client.assessment_model() == ""
    assert llm_client.research_model() == ""

    client = _build(monkeypatch, tmp_path, dotenv={"GROQ_API_KEY": "secret"})
    assert client.audit.model == "" and client.research.model == ""
    assert client.audit.available is False and client.research.available is False
    with pytest.raises(llm_client.ProviderError) as raised:
        client.audit.generate(llm_client.Request("assessment", "p"))
    assert "no Groq model is configured" in str(raised.value)
    # No model name may act as a DEFAULT. A model may be named in a
    # per-model calibration or rate table - that is tuning, not a fallback -
    # but never as the value either resolver falls back to.
    source = Path(llm_client.__file__).read_text(encoding="utf-8")
    assert "gpt-oss-120b" not in source, "the retired model must be gone entirely"
    for resolver in ("def assessment_model()", "def research_model()"):
        body = source.split(resolver)[1].split("\n\n\n")[0]
        for named in ("qwen", "gpt-oss"):
            assert named not in body.lower(), f"{resolver} must have no default"
    # And the calibration tables never supply a model to call.
    for table in ("_MODEL_INPUT_CHARS_PER_TOKEN", "_MODEL_INPUT_SAFETY"):
        assert f"{table}: dict" in source or f"{table} = " in source, table
    assert llm_client.assessment_model() == ""
    assert llm_client.research_model() == ""


def test_each_audit_transport_gets_its_own_configured_model(monkeypatch, tmp_path):
    client = _build(monkeypatch, tmp_path,
                    dotenv={"GROQ_ASSESSMENT_MODEL": "qwen/qwen3.8-27b",
                            "GROQ_RESEARCH_MODEL": "openai/gpt-oss-20b"})
    assert client.audit.model == "qwen/qwen3.8-27b"
    assert client.research.model == "openai/gpt-oss-20b"
    assert client.audit.available and client.research.available
    # GPT-OSS-120B has no normal role any more.
    assert "120b" not in f"{client.audit.model}{client.research.model}"


def test_model_settings_follow_the_same_env_then_dotenv_precedence(monkeypatch, tmp_path):
    import llm_client

    monkeypatch.setattr(engine, "PROJECT_ROOT", tmp_path)
    (tmp_path / ".env").write_text(
        "GROQ_ASSESSMENT_MODEL=dotenv/assessor\nGROQ_RESEARCH_MODEL=dotenv/researcher\n",
        encoding="utf-8")
    monkeypatch.delenv("GROQ_ASSESSMENT_MODEL", raising=False)
    monkeypatch.delenv("GROQ_RESEARCH_MODEL", raising=False)
    assert llm_client.assessment_model() == "dotenv/assessor"
    assert llm_client.research_model() == "dotenv/researcher"
    monkeypatch.setenv("GROQ_ASSESSMENT_MODEL", "env/assessor")
    assert llm_client.assessment_model() == "env/assessor"
    assert llm_client.research_model() == "dotenv/researcher"


def test_neither_audit_transport_can_reach_gemini(monkeypatch, tmp_path):
    client = _build(monkeypatch, tmp_path,
                    dotenv={"GROQ_ASSESSMENT_MODEL": "a", "GROQ_RESEARCH_MODEL": "b"})
    assert client.audit.fallback is None
    assert client.research.fallback is None
    assert client.audit is not client.research
    # And generation has no Groq transport at all to fall back FROM.
    assert client.gemini is client.groq


def test_each_model_has_its_own_pacing_budget(monkeypatch, tmp_path):
    """Qwen's reservation must not stall GPT-OSS, or parallelism buys nothing."""
    import llm_client

    client = _build(monkeypatch, tmp_path,
                    dotenv={"GROQ_ASSESSMENT_MODEL": "qwen/qwen3.8-27b",
                            "GROQ_RESEARCH_MODEL": "openai/gpt-oss-20b"})
    assert client.audit.budget is not client.research.budget
    assert llm_client.budget_for("qwen/qwen3.8-27b") is client.audit.budget
    assert llm_client.budget_for("openai/gpt-oss-20b") is client.research.budget

    # Fill the assessor's window completely; research must still not wait.
    client.audit.budget.record(client.audit.budget.limit * 2)
    slept = []
    client.research.budget._sleep = lambda s: slept.append(s)
    assert client.research.budget.wait_for(1000) == 0.0
    assert slept == [], "research paced on the assessor's budget"


def test_per_model_tpm_limits_are_configurable(monkeypatch, tmp_path):
    import llm_client

    monkeypatch.setattr(engine, "PROJECT_ROOT", tmp_path)
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.delenv("GROQ_TPM_LIMIT", raising=False)
    monkeypatch.setenv("GROQ_TPM_LIMIT__QWEN_QWEN3_8_27B", "30000")
    assert llm_client.model_tpm_limit("qwen/qwen3.8-27b") == 30000
    # Unset models fall back to the global setting, then the default.
    monkeypatch.setenv("GROQ_TPM_LIMIT", "12000")
    assert llm_client.model_tpm_limit("openai/gpt-oss-20b") == 12000


def test_mock_mode_is_untouched_by_model_resolution(monkeypatch, tmp_path):
    """--mock never reads GROQ_MODEL: there is no transport to configure."""
    import logging

    import llm_client

    monkeypatch.delenv("GROQ_MODEL", raising=False)
    monkeypatch.setattr(engine, "PROJECT_ROOT", tmp_path)
    client = llm_client.build_client(engine.load_master(), engine.load_policy(),
                                      logging.getLogger("test"), mock=True)
    assert isinstance(client.groq, llm_client.MockTransport)
    assert client.audit is client.groq


def test_model_resolution_touches_no_generation_behaviour():
    """The resolvers are read only by build_client, never by a prompt path."""
    source = Path(_llm_client().__file__).read_text(encoding="utf-8")
    for resolver in ("assessment_model()", "research_model()"):
        # One call site (build_client), plus its own definition.
        assert source.count(resolver) - source.count(f"def {resolver}") == 1, resolver
    builder = source.split("def build_client(")[1]
    assert "assessment_model()" in builder and "research_model()" in builder
    for method in ("def select_projects(", "def write_bullets(", "def cover_letter(",
                   "def assess(", "def research_company("):
        # Stop at the next method OR the next module-level definition, so a
        # trailing method does not swallow build_client.
        body = source.split(method)[1].split("\n    def ")[0].split("\n\ndef ")[0]
        assert "configured_groq_model" not in body, method


# ====================================== AG. Groq token budget / 413 handling
#
# A live --assess run failed three ways at once: the application audit asked
# for more tokens than the whole per-minute budget and was retried three times
# (each retry spending the budget again), and the research call was rejected
# because Groq refuses JSON mode alongside tool calling.

def _captured_audit_requests(run, c3):
    """Every audit Request the client would send, without any network call."""
    import logging

    import llm_client

    class _Capture:
        name = "capture"

        def __init__(self):
            self.requests = []

        def generate(self, request):
            self.requests.append(request)
            raise llm_client.ProviderError("transport", "captured")

    transport = _Capture()
    client = llm_client.LLMClient(transport, transport, run["master"], run["policy"],
                                   logging.getLogger("test"), audit=transport)
    signals = c3_signals(c3)
    decision = engine.select_experience(run["template"], run["policy"], signals)
    master = run["master"]
    ids = ["lms", "pintos", "temp"]
    chosen = [master.project(p) for p in ids]
    bullets = {p: [f"A bullet about {p}."] for p in ids}
    catalogue = run_pipeline.project_catalogue(master, _selection(ids), chosen, bullets)
    for call in (
        lambda: client.assess(
            c3["jd"], signals, chosen, ["Python"],
            experience_plain=["x"], project_bullets=bullets,
            requirements=c3["requirements"], tailoring={}, extra_experience=[],
            project_catalogue=catalogue,
            selection_detail={"selected": ids, "display_order": ids,
                              "allocation": {"lms": 3, "pintos": 2, "temp": 2}}),
        lambda: client.research_company(c3["jd"], jd_posted="UNKNOWN", today="2026-09-15"),
    ):
        try:
            call()
        except Exception:                    # noqa: BLE001 - the capture raises
            pass
    return {r.purpose: r for r in transport.requests}


# ---- 413 is never retried -------------------------------------------------

@pytest.mark.parametrize("status, body", [
    (413, "Request too large for model"),
    (200, "Error code: 413 - Request too large for model on tokens per minute (TPM)"),
])
def test_a_request_too_large_is_its_own_category(status, body):
    import llm_client

    assert llm_client.classify_http(status, body) == "request_too_large"
    assert llm_client.classify_exception(
        Exception("Error code: 413 - {'message': 'Request too large ...'}")
    ) == "request_too_large"


def test_a_request_too_large_is_not_retried():
    """Three attempts at an oversize request spend the budget three times."""
    import logging

    import llm_client

    attempts = []

    class _TooLarge(llm_client.GroqTransport):
        def _generate(self, request):
            attempts.append(request.purpose)
            raise llm_client.ProviderError("request_too_large",
                                           "Request too large for model")

    transport = _TooLarge(_credentials(), logging.getLogger("test"), model="m",
                          fallback=None, budget=llm_client.TokenBudget(100000))
    with pytest.raises(llm_client.ProviderError) as raised:
        transport.generate(llm_client.Request("assessment", "p" * 100))
    assert len(attempts) == 1, f"retried {len(attempts)} times"
    assert raised.value.category == "request_too_large"
    # A retryable category still gets its attempts.
    assert llm_client.GROQ_MAX_ATTEMPTS == 3


# ---- JSON mode and tool calling are mutually exclusive on Groq ------------

def test_company_research_asks_for_tools_instead_of_json_mode(run, c3):
    """Groq: "json mode cannot be combined with tool/function calling"."""
    requests = _captured_audit_requests(run, c3)
    research = requests["company_research"]
    assert research.web_search is True
    assert research.json is False, "JSON mode would make Groq reject the browsing tool"
    # The shape is still demanded, so parse_json has something to read.
    assert "Return ONLY valid JSON" in research.prompt
    # The other two audits keep JSON mode and never browse.
    assert requests["assessment"].json is True
    assert requests["assessment"].web_search is False


def test_a_prose_mode_research_reply_still_parses():
    import llm_client

    fenced = ('```json\n{"company_visa_sponsorship": "NO", "company_visa_confidence": '
              '"HIGH", "company_stem_opt_support": "UNKNOWN", "job_posted": "2026-09-01", '
              '"checked_at": "2026-09-15", "sources": []}\n```')
    data = llm_client.parse_json(fenced, "company_research")
    research, _ = grounding.validate_company_research(data, jd_text="We are hiring.")
    assert research["company_visa_sponsorship"] == "NO"
    assert research["job_posted"] == "2026-09-01"


# ---- every audit request fits the per-minute budget -----------------------

def test_each_audit_request_fits_the_token_budget(run, c3):
    import llm_client

    requests = _captured_audit_requests(run, c3)
    for purpose, request in requests.items():
        estimate = llm_client.estimate_tokens(request)
        assert estimate < llm_client.GROQ_TPM_LIMIT, (
            f"{purpose} reserves ~{estimate} tokens, over the "
            f"{llm_client.GROQ_TPM_LIMIT} TPM budget; a 413 cannot be retried away")


def test_audits_reserve_only_the_output_they_need():
    import llm_client

    assert set(llm_client.AUDIT_OUTPUT_TOKENS) == set(AUDIT_PURPOSES)
    for purpose, reserved in llm_client.AUDIT_OUTPUT_TOKENS.items():
        assert reserved < engine.GROQ_MAX_OUTPUT_TOKENS, purpose
    assert sum(llm_client.AUDIT_OUTPUT_TOKENS.values()) < llm_client.GROQ_TPM_LIMIT


def test_the_catalogue_still_names_every_project_after_trimming(run, c3):
    """Token trimming may abridge evidence; it may not hide a project."""
    requests = _captured_audit_requests(run, c3)
    prompt = requests["assessment"].prompt
    for project in run["master"].projects:
        assert project.project_id in prompt, project.project_id
    # The three that shipped still carry their FULL evidence for fidelity.
    for pid in ("lms", "pintos", "temp"):
        evidence = run["master"].project(pid).evidence
        assert evidence[0][:60] in prompt, pid
    assert "FINAL BULLETS" in prompt

def _budget(limit=8000):
    import llm_client

    state = {"now": 0.0, "slept": []}

    def sleep(seconds):
        state["slept"].append(seconds)
        state["now"] += seconds

    budget = llm_client.TokenBudget(limit, sleep=sleep, clock=lambda: state["now"])
    return budget, state


def test_the_budget_waits_for_the_window_to_clear():
    budget, state = _budget(8000)
    assert budget.wait_for(6000) == 0.0
    budget.record(6000)
    # 6000 + 3000 exceeds 8000, so the oldest usage must age out first.
    waited = budget.wait_for(3000)
    assert waited == pytest.approx(60.0)
    assert state["slept"] == [pytest.approx(60.0)]
    # The window is empty again, so the call proceeds.
    assert sum(t for _, t in budget.used) == 0


def test_the_budget_never_blocks_a_request_bigger_than_the_whole_limit():
    budget, state = _budget(8000)
    budget.record(100)
    # Clamped to the limit, so it waits once for the window and then proceeds
    # rather than looping forever on something that can never fit.
    waited = budget.wait_for(50000)
    assert waited <= 60.0
    assert state["slept"] and sum(state["slept"]) <= 60.0


def test_an_empty_window_never_waits():
    budget, state = _budget(8000)
    assert budget.wait_for(50000) == 0.0
    assert state["slept"] == []


def test_actual_usage_replaces_the_pre_call_estimate():
    budget, _ = _budget(8000)
    budget.record(5000)                       # the estimate
    budget.record(1200, replaces=5000)        # what the provider billed
    assert sum(t for _, t in budget.used) == 1200


def test_the_transport_records_provider_usage_when_it_is_reported():
    import logging

    import llm_client

    class _Reporting(llm_client.GroqTransport):
        def _generate(self, request):
            self.last_usage = 1500
            return "{}"

    budget, _ = _budget(100000)
    transport = _Reporting(_credentials(), logging.getLogger("test"), model="m",
                           fallback=None, budget=budget)
    transport.generate(llm_client.Request("assessment", "p" * 6000, max_tokens=2600))
    assert sum(t for _, t in budget.used) == 1500, budget.used


def test_budgets_are_per_model_not_per_account(monkeypatch, tmp_path):
    """The two audits are deliberately parallel, so they cannot share a bucket."""
    import llm_client

    client = _build(monkeypatch, tmp_path,
                    dotenv={"GROQ_ASSESSMENT_MODEL": "model/one",
                            "GROQ_RESEARCH_MODEL": "model/two"})
    assert client.audit.budget is not client.research.budget
    assert client.audit.budget is llm_client.budget_for("model/one")
    assert client.research.budget is llm_client.budget_for("model/two")
    assert llm_client.budget_for("model/one").limit == llm_client.model_tpm_limit("model/one")


def test_the_budget_is_advisory_and_paces_only_groq():
    """The mock transport has no budget: --mock never waits."""
    import logging

    import llm_client

    mock = llm_client.MockTransport(engine.load_master(), logging.getLogger("test"))
    assert not hasattr(mock, "budget")
    source = Path(llm_client.__file__).read_text(encoding="utf-8")
    # Pacing lives in the Groq transport only.
    assert source.count("self.budget.wait_for(") == 1
    # TokenBudget is defined between the two transports, so stop there.
    gemini = source.split("class GeminiTransport")[1].split("class TokenBudget")[0]
    assert "budget" not in gemini, "the Gemini transport is not paced by the Groq budget"


# ================================ AH. reasoning effort and truncated output
#
# The second live --assess run truncated two audits: GPT-OSS at its default
# medium reasoning effort spent the reserved output thinking, hit
# finish_reason=length, and the identical retry then blew the TPM window. The
# audits now ask for low effort, promise compact output, and never resend a
# request that truncated.

def _stub_groq(*, generate, budget=None, model="m"):
    """A GroqTransport whose only live method is replaced."""
    import logging

    import llm_client

    class _Stub(llm_client.GroqTransport):
        def _generate(self, request):
            return generate(self, request)

    return _Stub(_credentials(), logging.getLogger("test"), model=model,
                 fallback=None, budget=budget or llm_client.TokenBudget(1_000_000))


# ---- 1. low reasoning on the audits only ----------------------------------

def test_the_audits_keep_reasoning_out_of_their_output_budgets(run, c3):
    requests = _captured_audit_requests(run, c3)
    assert sorted(requests) == sorted(AUDIT_PURPOSES)
    # The assessment turns reasoning OFF: hidden reasoning still spent the 900
    # completion tokens and truncated the document before it closed.
    assert requests["assessment"].reasoning_effort == "none"
    assert requests["assessment"].reasoning_format is None
    assert requests["assessment"].include_reasoning is None
    # Research is unchanged: low effort, reasoning not returned.
    assert requests["company_research"].reasoning_effort == "low"
    assert requests["company_research"].include_reasoning is False
    assert requests["company_research"].reasoning_format is None


def test_generation_requests_carry_no_reasoning_settings(run, bae):
    """Generation keeps the provider default: the fields are not sent at all."""
    import llm_client

    transport = _SequenceTransport([json.dumps({
        "role_family": "backend", "career_stage": "early_career",
        "selected": [{"project_id": pid, "llm_rank": rank, "reason": "r"}
                     for rank, pid in enumerate(("lms", "pintos", "temp"), start=1)],
        "considered": []})])
    import logging
    client = llm_client.LLMClient(transport, transport, run["master"], run["policy"],
                                   logging.getLogger("test"), audit=transport)
    client.select_projects(bae["jd"].text, c3_signals(bae), 3)
    request = transport.requests[0]
    assert request.purpose == "project_selection"
    assert request.reasoning_effort is None
    assert request.include_reasoning is None
    # Defaults on the dataclass, so no generation path can leak them.
    blank = llm_client.Request("project_bullets", "p")
    assert blank.reasoning_effort is None and blank.include_reasoning is None


def test_reasoning_settings_reach_the_sdk_only_when_asked():
    import llm_client

    captured = {}

    def fake(self, request):
        # Mirror what _generate assembles, without the SDK.
        settings = {}
        if request.json:
            settings["response_format"] = {"type": "json_object"}
        if request.reasoning_effort:
            settings["reasoning_effort"] = request.reasoning_effort
        if request.include_reasoning is not None:
            settings["include_reasoning"] = request.include_reasoning
        captured[request.purpose] = settings
        return "{}"

    source = Path(llm_client.__file__).read_text(encoding="utf-8")
    generate = source.split("def _generate(self, request: Request) -> str:")[1]
    generate = generate.split("\n\n\n")[0]
    assert 'settings["reasoning_effort"] = request.reasoning_effort' in generate
    assert 'settings["include_reasoning"] = request.include_reasoning' in generate
    # Guarded, so an unset field is never transmitted.
    assert "if request.reasoning_effort:" in generate
    assert "if request.include_reasoning is not None:" in generate

    transport = _stub_groq(generate=fake)
    transport.generate(llm_client.Request("assessment", "p", **llm_client.AUDIT_REASONING))
    transport.generate(llm_client.Request("project_bullets", "p"))
    assert captured["assessment"]["reasoning_effort"] == "low"
    assert captured["assessment"]["include_reasoning"] is False
    assert "reasoning_effort" not in captured["project_bullets"]
    assert "include_reasoning" not in captured["project_bullets"]


# ---- 2. the output caps ---------------------------------------------------

def test_the_output_caps_are_exactly_as_agreed():
    import llm_client

    assert llm_client.AUDIT_OUTPUT_TOKENS == {
        # Qwen's tier caps OUTPUT tokens per minute at 1000, so the assessment
        # reserves 900; research is unchanged.
        "assessment": 900,
        "company_research": 1000,
    }


def test_each_audit_request_still_fits_after_the_cap_change(run, c3):
    import llm_client

    requests = _captured_audit_requests(run, c3)
    for purpose, request in requests.items():
        assert request.max_tokens == llm_client.AUDIT_OUTPUT_TOKENS[purpose], purpose
        assert llm_client.estimate_tokens(request) < llm_client.GROQ_TPM_LIMIT, purpose


# ---- 3. compact output, same information ---------------------------------

def test_the_audits_demand_compact_output_without_dropping_inputs(run, c3):
    requests = _captured_audit_requests(run, c3)
    assessment = requests["assessment"].prompt
    assert "OUTPUT LENGTH" in assessment
    for rule in ("2 entries each", "at most 35 words",
                 "HARD 900-token completion budget"):
        assert rule in assessment, rule
    # The schema now enforces the shape, so its prose is gone from the prompt.
    for redundant in ("Return ONLY valid JSON", "ENUM CONTRACT", '"fit_score": 8.2'):
        assert redundant not in assessment, redundant
    assert "HOW THE SHIPPED PROFESSIONAL EXPERIENCE WAS CHOSEN" in assessment
    assert "at most 15 words" in assessment
    # Inputs are untouched: every project and every approved bullet is still
    # offered, with the selected projects' full evidence.
    for project in run["master"].projects:
        assert project.project_id in assessment, project.project_id
    for pid in ("lms", "pintos", "temp"):
        assert run["master"].project(pid).evidence[0][:60] in assessment, pid



def test_the_audit_schemas_are_unchanged():
    """Compact prose must not become a different contract."""
    assert grounding.EXPERIENCE_AUDIT_COMPONENTS == (
        "signal_interpretation", "policy_choice", "evidence_relevance")
    assert grounding.PROJECT_SELECTION_COMPONENTS == (
        "jd_relevance", "best_available_chosen", "complementary_coverage",
        "ranking_and_allocation")
    assert grounding.PROJECT_BULLET_COMPONENTS == (
        "jd_relevance", "technical_specificity", "evidence_fidelity",
        "impact_ownership", "non_redundancy")
    assert grounding.AUDIT_VERDICTS == ("PASS", "REVIEW")
    assert grounding.CALLBACK_LIKELIHOOD == ("LOW", "MEDIUM", "HIGH", "UNKNOWN")
    audit, _ = grounding.validate_application_audit(
        {"experience_selection": {"score": 8.0, "notes": ["fits"]},
         "tailoring_quality": {"score": 9.0},
         "project_selection": {"score": 8.0, "components": {}},
         "project_bullets": {"score": 8.0, "components": {}},
         "callback_likelihood": "MEDIUM"})
    assert audit["experience_selection_score"] == 8.0
    assert audit["callback_likelihood"] == "MEDIUM"


# ---- 4. a truncated answer is never resent -------------------------------

def test_finish_reason_length_becomes_output_truncated():
    import llm_client

    def truncate(self, request):
        raise ProviderTruncation(request)

    class ProviderTruncation(llm_client.ProviderError):
        def __init__(self, request):
            super().__init__("output_truncated",
                             f"Groq hit the {request.max_tokens}-token output cap "
                             f"(finish_reason=length)")

    # The category is raised by _generate itself, at the finish_reason check.
    source = Path(llm_client.__file__).read_text(encoding="utf-8")
    block = source.split('if choice.finish_reason == "length":')[1].split("return")[0]
    assert 'ProviderError("output_truncated"' in block
    assert "malformed_response" not in block

    transport = _stub_groq(generate=truncate)
    with pytest.raises(llm_client.ProviderError) as raised:
        transport.generate(llm_client.Request("assessment", "p"))
    assert raised.value.category == "output_truncated"


@pytest.mark.parametrize("category", ["output_truncated", "request_too_large"])
def test_a_non_retryable_category_gets_exactly_one_attempt(category):
    import llm_client

    attempts = []

    def fail(self, request):
        attempts.append(request.purpose)
        raise llm_client.ProviderError(category, f"simulated {category}")

    transport = _stub_groq(generate=fail)
    with pytest.raises(llm_client.ProviderError) as raised:
        transport.generate(llm_client.Request("assessment", "p"))
    assert attempts == ["assessment"], attempts
    assert raised.value.category == category


@pytest.mark.parametrize("category", ["rate_limited", "server_error"])
def test_transient_categories_still_retry(category, monkeypatch):
    import llm_client

    monkeypatch.setattr(llm_client.time, "sleep", lambda _s: None)
    attempts = []

    def fail(self, request):
        attempts.append(category)
        raise llm_client.ProviderError(category, f"simulated {category}")

    transport = _stub_groq(generate=fail)
    with pytest.raises(llm_client.ProviderError):
        transport.generate(llm_client.Request("assessment", "p"))
    assert len(attempts) == llm_client.GROQ_MAX_ATTEMPTS, attempts


def test_a_truncated_audit_becomes_unavailable_not_a_failed_run(run, c3):
    """The audit degrades to UNKNOWN; the run is untouched."""
    import logging

    class _Truncating:
        name = "truncating"

        def generate(self, request):
            import llm_client

            if request.purpose == "assessment":
                raise llm_client.ProviderError("output_truncated", "output cap reached")
            assert request.purpose == "company_research", request.purpose
            return llm_client.Reply(json.dumps(
                {"company_visa_sponsorship": "UNKNOWN",
                 "company_stem_opt_support": "UNKNOWN", "job_posted": "UNKNOWN",
                 "checked_at": "2026-09-15", "sources": []}), self.name)

    import llm_client
    transport = _Truncating()
    client = llm_client.LLMClient(transport, transport, run["master"], run["policy"],
                                   logging.getLogger("test"), audit=transport)
    signals = c3_signals(c3)
    decision = engine.select_experience(run["template"], run["policy"], signals)
    ids = ["lms", "pintos", "temp"]
    audits = run_pipeline.run_audits(
        client, c3["jd"], signals, decision, policy=run["policy"], master=run["master"],
        selection=_selection(ids), chosen=[run["master"].project(p) for p in ids],
        project_bullets={p: ["b"] for p in ids}, display_order=ids,
        allocation={"lms": 3, "pintos": 2, "temp": 2}, skills=["Python"],
        experience_plain=["x"], requirements=c3["requirements"], tailoring={},
        extra_experience=[], log=run_pipeline.StageLog(logging.getLogger("t"), "T"))

    assert "output_truncated" in audits["errors"]["application_audit"]
    report = run_pipeline.assessment_report(audits, 9.5)
    assert report["Resume Tailoring Score"] == "UNKNOWN"
    assert report["Fit Match Score"] == "UNKNOWN"
    # The Experience score lives in the failed audit now, so it degrades with
    # it; research and the deterministic letter score are untouched.
    assert report["Experience Selection Score"] == "UNKNOWN"
    assert report["Company Visa Sponsorship"] == "UNKNOWN"
    assert report["Cover Letter Score"] == "9.5 / 10"


# ---- 5. budget accounting on the failure paths ---------------------------

def test_usage_from_a_truncated_completion_replaces_the_estimate():
    """The tokens were spent even though the answer was unusable."""
    import llm_client

    budget, _ = _budget(100000)

    def truncate(self, request):
        self.last_usage = 2450          # what Groq actually billed
        raise llm_client.ProviderError("output_truncated", "output cap reached")

    transport = _stub_groq(generate=truncate, budget=budget)
    request = llm_client.Request("assessment", "p" * 6000, max_tokens=2600)
    estimate = llm_client.estimate_tokens(request)
    with pytest.raises(llm_client.ProviderError):
        transport.generate(request)
    assert sum(t for _, t in budget.used) == 2450, budget.used
    assert estimate not in [t for _, t in budget.used]


def test_a_413_rejected_before_generation_releases_its_reservation():
    """Nothing was consumed, so the window must not hold the estimate."""
    import llm_client

    budget, _ = _budget(100000)

    def reject(self, request):
        raise llm_client.ProviderError("request_too_large", "Request too large")

    transport = _stub_groq(generate=reject, budget=budget)
    with pytest.raises(llm_client.ProviderError):
        transport.generate(llm_client.Request("assessment", "p" * 6000, max_tokens=2600))
    assert budget.used == [], budget.used


def test_a_transient_failure_keeps_its_reservation(monkeypatch):
    """A 5xx may well have cost tokens, so the estimate stands."""
    import llm_client

    monkeypatch.setattr(llm_client.time, "sleep", lambda _s: None)
    budget, _ = _budget(1_000_000)

    def fail(self, request):
        raise llm_client.ProviderError("server_error", "500")

    transport = _stub_groq(generate=fail, budget=budget)
    request = llm_client.Request("assessment", "p" * 6000, max_tokens=2600)
    with pytest.raises(llm_client.ProviderError):
        transport.generate(request)
    # One reservation per attempt, none released.
    assert len(budget.used) == llm_client.GROQ_MAX_ATTEMPTS
    assert all(t == llm_client.estimate_tokens(request) for _, t in budget.used)


def test_release_only_drops_a_matching_reservation():
    budget, _ = _budget(100000)
    budget.record(1000)
    budget.record(2000)
    budget.release(2000)
    assert [t for _, t in budget.used] == [1000]
    budget.release(9999)                 # nothing matches; nothing is dropped
    assert [t for _, t in budget.used] == [1000]


# ---- unchanged guarantees -------------------------------------------------

def test_company_research_path_is_functionally_unchanged(run, c3):
    """Browsing works live now: json off, web search on, tool use forced."""
    import llm_client

    research = _captured_audit_requests(run, c3)["company_research"]
    assert research.json is False
    assert research.web_search is True
    assert research.max_tokens == 1000
    assert research.reasoning_effort == "low"
    source = Path(llm_client.__file__).read_text(encoding="utf-8")
    generate = source.split("def _generate(self, request: Request) -> str:")[1]
    generate = generate.split("\n\n\n")[0]
    assert 'settings["tool_choice"] = "required"' in generate
    assert 'settings["tools"] = [{"type": "browser_search"}]' in generate


def test_the_audits_are_still_groq_only_after_these_changes():
    source = Path(_llm_client().__file__).read_text(encoding="utf-8")
    for purpose, transport in (("assessment", "self.audit"),
                               ("company_research", "self.research")):
        head = source.split(f'"{purpose}", ')[0]
        assert head.rsplit("self._invoke(", 1)[1].split(",")[0].strip() == transport
    builder = source.split("def build_client(")[1]
    assert "assessor = GroqTransport(credentials, log, model=assessment_model(), " \
           "fallback=None)" in builder
    assert "researcher = GroqTransport(credentials, log, model=research_model(), " \
           "fallback=None)" in builder


# ============================= AI. semantic JD signals, one Gemini call
#
# The deterministic classifier is keyword-driven and misses semantics: a
# posting that says "participate in design and code reviews" is
# code-quality-heavy without using a term it matches. Gemini reports OUR
# taxonomy from the JD in the SAME call that picks projects, Python validates
# the evidence, merges by OR, and then applies its own Experience policy.

# A posting whose code-quality emphasis is purely semantic: no term the
# deterministic detector looks for, but unmistakable to a reader.
SEMANTIC_JD = """Company: Meridian Systems
Job Title: Software Engineer

About the role
You will join a small platform group that ships Python services. We care a great deal
about how work gets done: you will participate in design and code reviews, help
establish engineering best practices across the group, and pair with teammates when a
change touches unfamiliar ground. We expect engineers to leave the codebase clearer
than they found it and to give thoughtful written feedback on each other's changes.

Requirements
Experience building services in Python. Comfort with relational databases and REST
interfaces. Strong written communication.
"""


def _semantic_payload(signal: str, evidence: list[str]) -> dict:
    return {signal: {"present": True, "evidence": evidence}}


# ---- one call, two jobs ---------------------------------------------------

def test_project_selection_is_still_exactly_one_gemini_call(run, c3):
    import logging

    import llm_client

    transport = _RecordingTransport({"project_selection": {
        "role_family": "backend", "career_stage": "early_career",
        "selected": [{"project_id": pid, "llm_rank": rank, "reason": "r"}
                     for rank, pid in enumerate(("lms", "pintos", "temp"), start=1)],
        "considered": [],
        "semantic_jd_signals": {"backend": {"present": False, "evidence": []}},
    }})
    client = llm_client.LLMClient(transport, transport, run["master"], run["policy"],
                                   logging.getLogger("test"), audit=transport)
    selection = client.select_projects(c3["jd"].text, c3_signals(c3), 3)
    assert len(transport.requests) == 1, "semantic signals must not cost a second call"
    assert transport.requests[0].purpose == "project_selection"
    assert selection.selected == ["lms", "pintos", "temp"]
    assert selection.semantic_signals_raw == {
        "backend": {"present": False, "evidence": []}}


def test_the_selection_prompt_asks_for_the_fixed_taxonomy_only(run, c3):
    import logging

    import llm_client

    transport = _RecordingTransport({})
    client = llm_client.LLMClient(transport, transport, run["master"], run["policy"],
                                   logging.getLogger("test"), audit=transport)
    try:
        client.select_projects(c3["jd"].text, c3_signals(c3), 3)
    except Exception:                        # noqa: BLE001 - empty payload
        pass
    prompt = transport.requests[0].prompt
    assert "SEMANTIC JD SIGNALS" in prompt
    for name in grounding.SEMANTIC_SIGNAL_NAMES:
        assert name in prompt, name
    assert "COPIED VERBATIM" in prompt
    # It must be told it has no authority over Experience at all.
    assert "bullet ids" in prompt
    assert "yours to choose" in prompt


# ---- validation -----------------------------------------------------------

def test_only_predefined_signal_names_are_accepted():
    validated, problems = grounding.validate_semantic_signals(
        {"backend": {"present": False, "evidence": []},
         "vibes": {"present": True, "evidence": ["you will join a small platform group"]}},
        SEMANTIC_JD)
    assert "vibes" not in validated
    assert any("unknown semantic signal" in p.message for p in problems)
    assert set(validated) <= set(grounding.SEMANTIC_SIGNAL_NAMES)


def test_a_positive_signal_requires_grounded_jd_evidence():
    grounded = grounding.validate_semantic_signals(
        _semantic_payload("code_quality_collaboration_heavy",
                          ["participate in design and code reviews"]), SEMANTIC_JD)[0]
    assert grounded["code_quality_collaboration_heavy"] is True

    # No evidence at all.
    bare, problems = grounding.validate_semantic_signals(
        {"code_quality_collaboration_heavy": {"present": True, "evidence": []}},
        SEMANTIC_JD)
    assert bare["code_quality_collaboration_heavy"] is False
    assert any("requires 1-" in p.message for p in problems)


def test_hallucinated_evidence_is_rejected():
    validated, problems = grounding.validate_semantic_signals(
        _semantic_payload("healthcare",
                          ["we build clinical decision support for hospitals"]),
        SEMANTIC_JD)
    assert validated["healthcare"] is False
    assert any("none of its evidence appears" in p.message for p in problems)
    # A paraphrase of something real is still not a quote.
    paraphrase = grounding.validate_semantic_signals(
        _semantic_payload("code_quality_collaboration_heavy",
                          ["the team values reviewing code carefully"]), SEMANTIC_JD)[0]
    assert paraphrase["code_quality_collaboration_heavy"] is False


@pytest.mark.parametrize("payload", [
    None, [], "text", {"backend": "yes"}, {"backend": {"present": "maybe"}},
    {"backend": {"present": True, "evidence": [7]}},
])
def test_a_malformed_payload_never_crashes_generation(payload):
    validated, problems = grounding.validate_semantic_signals(payload, SEMANTIC_JD)
    assert isinstance(validated, dict)
    assert all(isinstance(v, bool) for v in validated.values())
    assert all(p.severity == "warning" for p in problems)


# ---- merge policy ---------------------------------------------------------

def test_gemini_false_can_never_withdraw_a_deterministic_true():
    master = engine.load_master()
    signals = engine.classify_jd(engine.read_jd(EPIC_JD).text, master.section_order)
    assert signals.healthcare is True, "fixture assumption"
    merged, overrides = grounding.merge_semantic_signals(
        signals, {"healthcare": False, "code_quality_collaboration_heavy": False})
    assert merged["healthcare"] is True
    assert overrides["added"] == []
    # And the Signals object handed to the policy is the untouched original.
    assert grounding.apply_semantic_overrides(signals, overrides) is signals
    assert engine.experience_rule_for(signals)[0].startswith("Healthcare")


def test_validated_gemini_true_supplements_a_missed_deterministic_signal():
    master = engine.load_master()
    signals = engine.classify_jd(SEMANTIC_JD, master.section_order)
    # The deterministic detector genuinely misses it: that is the premise.
    assert signals.code_quality is False, "fixture must be a real recall gap"

    validated, problems = grounding.validate_semantic_signals(
        _semantic_payload("code_quality_collaboration_heavy",
                          ["participate in design and code reviews",
                           "establish engineering best practices"]), SEMANTIC_JD)
    assert problems == [] and validated["code_quality_collaboration_heavy"] is True

    merged, overrides = grounding.merge_semantic_signals(signals, validated)
    assert merged["code_quality_collaboration_heavy"] is True
    assert overrides["added"] == ["code_quality_collaboration_heavy"]
    assert overrides["fields"] == {"code_quality": True}


def test_the_recovered_signal_fires_the_existing_python_pr_review_rule():
    """End of the chain: the EXISTING approved swap, chosen by Python."""
    master, policy = engine.load_master(), engine.load_policy()
    template = engine.load_template()
    signals = engine.classify_jd(SEMANTIC_JD, master.section_order)
    validated = grounding.validate_semantic_signals(
        _semantic_payload("code_quality_collaboration_heavy",
                          ["participate in design and code reviews"]), SEMANTIC_JD)[0]
    _merged, overrides = grounding.merge_semantic_signals(signals, validated)
    merged_signals = grounding.apply_semantic_overrides(signals, overrides)

    before = engine.select_experience(template, policy, signals)
    after = engine.select_experience(template, policy, merged_signals)

    assert engine.experience_rule_for(merged_signals) == (
        "Non-healthcare + code-quality/collaboration-heavy", "jd_signal")
    assert after.rule != before.rule
    # The EXISTING approved PR-review swap, by id, from the spreadsheet.
    targets = {target for _source, target, _rule in after.swaps}
    assert any("PR-REVIEW" in t for t in targets), after.swaps
    assert after.shipped_ids != before.shipped_ids
    # Every shipped line is approved wording, byte for byte. A swapped bullet
    # ships its TARGET's approved text, so the whole approved pool is the
    # contract rather than the source id's own wording.
    approved = {e.latex for e in policy.experience_library}
    approved |= {e.latex for e in policy.alternate_library}
    for bullet_id, _action, latex in after.shipped:
        assert latex in approved, f"{bullet_id} shipped unapproved wording"


def test_gemini_never_selects_or_writes_experience():
    """Nothing in the semantic path can name a bullet or supply wording."""
    import llm_client

    master, policy = engine.load_master(), engine.load_policy()
    ids = {e.bullet_id for e in policy.experience_library}
    ids |= {e.bullet_id for e in policy.alternate_library}

    # 1. Even a payload that tries to name bullets cannot: names are rejected.
    validated, problems = grounding.validate_semantic_signals(
        {"SWE-ALT-PR-REVIEW": {"present": True, "evidence": ["code reviews"]},
         "experience": {"present": True, "evidence": ["SWE-4"]}}, SEMANTIC_JD)
    assert validated == {}
    assert len(problems) == 2

    # 2. The merge output is booleans only - there is no channel for text.
    signals = engine.classify_jd(SEMANTIC_JD, master.section_order)
    merged, overrides = grounding.merge_semantic_signals(
        signals, {"code_quality_collaboration_heavy": True})
    assert all(isinstance(v, bool) for v in merged.values())
    assert set(overrides) == {"fields", "role_family", "added"}
    assert not (ids & set(overrides["fields"]))

    # 3. The selection prompt never shows Experience, so it cannot echo it.
    source = Path(llm_client.__file__).read_text(encoding="utf-8")
    selection = source.split("def select_projects(")[1].split("    def write_bullets(")[0]
    for bullet_id in ids:
        assert bullet_id not in selection, bullet_id
    assert "experience_library" not in selection
    assert "alternate_library" not in selection


def test_gemini_may_only_promote_a_role_family_never_replace_one():
    master = engine.load_master()
    signals = engine.classify_jd(engine.read_jd(C3_JD).text, master.section_order)
    assert signals.domain_scores, "fixture assumption: a family was detected"
    _merged, overrides = grounding.merge_semantic_signals(
        signals, {"data_engineering": True})
    # A detected family is never displaced by a model's opinion.
    assert overrides["role_family"] is None
    assert grounding.apply_semantic_overrides(signals, overrides).role_family == \
        signals.role_family


def test_the_merge_only_reaches_the_experience_policy():
    """Project selection, bullets, skills and order use deterministic signals."""
    source = Path(run_pipeline.__file__).read_text(encoding="utf-8")
    run_one = source.split("def run_one(")[1].split("\ndef ")[0]
    assert run_one.count("merged_signals") >= 1
    # The only consumer is select_experience.
    for line in run_one.splitlines():
        if "merged_signals" in line and "=" not in line.split("merged_signals")[0]:
            continue
    assert "engine.select_experience(template, policy, merged_signals)" in run_one
    # Everything else still takes `signals`.
    assert "client.select_projects(jd.text, signals," in run_one
    assert "master.section_order.order_for(signals.section_mode)" in run_one


def test_strategy_records_every_signal_stage_separately():
    folders = sorted((ROOT / "output" / "_smoke_tests").glob("C3_AI_*"))
    if not folders:
        pytest.skip("no C3 smoke folder; run the C3 mock first")
    strategy = json.loads((folders[-1] / "strategy.json").read_text(encoding="utf-8"))
    block = strategy["jd_signals"]
    for key in ("deterministic_signals", "gemini_semantic_signals_raw",
                "gemini_semantic_signals_validated", "merged_signals",
                "recovered_by_gemini"):
        assert key in block, key
    assert strategy["experience"]["rule_fired"]
    assert strategy["experience"]["shipped_ids"]
    assert "resolved_from" in strategy["experience"]


# ---- cover letter is Gemini's --------------------------------------------

def test_the_cover_letter_goes_to_gemini_and_never_to_groq(run, bae):
    import logging

    import llm_client

    gemini = _SequenceTransport([_letter_body("590,540 transactions")])
    groq = _RecordingTransport({})
    client = llm_client.LLMClient(gemini, groq, run["master"], run["policy"],
                                   logging.getLogger("test"), audit=groq, research=groq)
    letter, _problems = _letter_call(client, run, bae)
    assert letter
    assert [r.purpose for r in gemini.requests] == ["cover_letter"]
    assert groq.requests == [], "Groq must never receive a cover-letter request"
    assert [c.purpose for c in client.calls] == ["cover_letter"]


def test_no_groq_transport_can_receive_a_cover_letter_request():
    source = Path(_llm_client().__file__).read_text(encoding="utf-8")
    head = source.split('"cover_letter", prompt')[0]
    assert head.rsplit("self._invoke(", 1)[1].split(",")[0].strip() == "self.gemini"


# ================== AJ. parallel audits, single writer, atomic persistence
#
# The two Groq audits hit different models and run concurrently. That makes
# the write path the dangerous part: two threads updating strategy.json would
# lose one another's work. The workers are therefore pure - they return result
# objects - and the coordinator performs exactly one atomic write.

def _code_only(text: str) -> str:
    """Source with docstrings and comments stripped, for contract scans.

    The prose in this pipeline names the things it promises NOT to do, so a
    substring scan has to look at the code alone.
    """
    quote = chr(34) * 3
    parts = text.split(quote)
    # Keep the even-indexed parts: the odd ones are docstring bodies.
    code = "".join(parts[::2])
    return "\n".join(line.split("#")[0] for line in code.splitlines())


def _research_payload(**overrides) -> dict:
    payload = {"company_visa_sponsorship": "UNKNOWN", "company_visa_confidence": "LOW",
               "company_stem_opt_support": "UNKNOWN", "company_stem_opt_confidence": "LOW",
               "job_posted": "UNKNOWN", "job_posted_confidence": "LOW",
               "checked_at": "2026-09-16", "sources": []}
    payload.update(overrides)
    return payload


def _assessment_payload_for(c3) -> dict:
    ids = [r.requirement_id for r in engine.scored_requirements(c3["requirements"])]
    return {
        "fit_score": 7.4, "recommendation": "apply",
        "summary": "Most scored requirements are evidenced.",
        "eligibility": {"status": "uncertain", "details": []},
        "strong_matches": [{"requirement_id": ids[0], "evidence": "shipped evidence",
                            "source": "experience"}],
        "partial_matches": [], "gaps": [], "manual_review": [],
        "complementary_strengths": [],
        "tailoring_quality": {"score": 9.0, "notes": ["well tailored"]},
        "experience_selection": {"score": 8.4, "notes": ["the rule fits"]},
        "project_selection": {"score": 8.5, "components": {}, "notes": []},
        "project_bullets": {"score": 8.6, "components": {}, "notes": []},
        "callback_likelihood": "MEDIUM", "risk_flags": [],
    }


class _AuditTransport:
    """Serves the two audit purposes, optionally slowly or by failing."""

    name = "audit-stub"

    def __init__(self, payloads: dict, *, delays: dict | None = None,
                 fail: dict | None = None):
        self.payloads = payloads
        self.delays = delays or {}
        self.fail = fail or {}
        self.started: list[str] = []
        self.finished: list[str] = []
        import threading

        self._lock = threading.Lock()

    def generate(self, request):
        import llm_client

        with self._lock:
            self.started.append(request.purpose)
        if self.delays.get(request.purpose):
            time.sleep(self.delays[request.purpose])
        with self._lock:
            self.finished.append(request.purpose)
        if request.purpose in self.fail:
            raise llm_client.ProviderError("server_error", self.fail[request.purpose])
        return llm_client.Reply(json.dumps(self.payloads[request.purpose]), self.name)


def _run_audits(run, c3, transport, *, log_name="T"):
    import logging

    import llm_client

    client = llm_client.LLMClient(transport, transport, run["master"], run["policy"],
                                   logging.getLogger(log_name), audit=transport,
                                   research=transport)
    signals = c3_signals(c3)
    decision = engine.select_experience(run["template"], run["policy"], signals)
    ids = ["lms", "pintos", "temp"]
    return client, run_pipeline.run_audits(
        client, c3["jd"], signals, decision, policy=run["policy"], master=run["master"],
        selection=_selection(ids), chosen=[run["master"].project(p) for p in ids],
        project_bullets={p: ["a bullet"] for p in ids}, display_order=ids,
        allocation={"lms": 3, "pintos": 2, "temp": 2}, skills=["Python"],
        experience_plain=["x"], requirements=c3["requirements"], tailoring={},
        extra_experience=[],
        log=run_pipeline.StageLog(logging.getLogger(log_name), log_name),
        experience_context={"merged_signals": {"backend": True}, "rule": "R",
                            "swaps": [], "shipped_ids": list(decision.shipped_ids)})


# ---- the two audits really do overlap ------------------------------------

def test_both_audits_are_submitted_before_either_is_awaited(run, c3):
    """If they were sequential, the slow one would finish before the fast one starts."""
    transport = _AuditTransport(
        {"assessment": _assessment_payload_for(c3), "company_research": _research_payload()},
        delays={"assessment": 0.35})
    _client, audits = _run_audits(run, c3, transport)

    assert audits["errors"] == {}
    # research STARTED while the slow assessment was still running, and
    # finished first - impossible if the calls were serialized.
    assert set(transport.started) == {"assessment", "company_research"}
    assert transport.finished[0] == "company_research", transport.finished
    assert transport.started.index("company_research") <= 1


def test_the_coordinator_uses_two_workers():
    source = Path(run_pipeline.__file__).read_text(encoding="utf-8")
    block = source.split("def run_audits(")[1].split("\ndef ")[0]
    assert "ThreadPoolExecutor(max_workers=2)" in block
    # Submitted first, awaited afterwards.
    submit = block.index("pool.submit(")
    await_at = block.index(".result()")
    assert submit < await_at, "futures must be submitted before either is awaited"
    assert block.count("pool.submit(") == 2


def test_output_order_is_deterministic_whichever_future_finishes_first(run, c3):
    fast_research = _AuditTransport(
        {"assessment": _assessment_payload_for(c3), "company_research": _research_payload()},
        delays={"assessment": 0.25})
    slow_research = _AuditTransport(
        {"assessment": _assessment_payload_for(c3), "company_research": _research_payload()},
        delays={"company_research": 0.25})
    _c1, first = _run_audits(run, c3, fast_research)
    _c2, second = _run_audits(run, c3, slow_research)

    assert list(first) == list(second), "key order must not depend on timing"
    assert first["application_audit"] == second["application_audit"]
    assert first["research"] == second["research"]
    assert run_pipeline.assessment_report(first, 9.5) == \
        run_pipeline.assessment_report(second, 9.5)


@pytest.mark.parametrize("failing, surviving", [
    ("assessment", "company_research"),
    ("company_research", "assessment"),
])
def test_one_failing_future_never_cancels_the_other(run, c3, failing, surviving):
    transport = _AuditTransport(
        {"assessment": _assessment_payload_for(c3), "company_research": _research_payload()},
        fail={failing: "simulated outage"})
    _client, audits = _run_audits(run, c3, transport)

    # Both calls were still MADE; only one failed.
    assert sorted(transport.started) == ["assessment", "company_research"]
    key = {"assessment": "application_audit", "company_research": "company_research"}
    assert key[failing] in audits["errors"]
    assert key[surviving] not in audits["errors"]
    report = run_pipeline.assessment_report(audits, 9.5)
    if failing == "assessment":
        assert report["Fit Match Score"] == "UNKNOWN"
        assert report["Company Visa Sponsorship"] == "UNKNOWN"   # honest UNKNOWN
    else:
        assert report["Fit Match Score"] == "7.4 / 10"
    # Advisory either way: the letter score never depended on a provider.
    assert report["Cover Letter Score"] == "9.5 / 10"


# ---- single writer --------------------------------------------------------

def test_the_audit_workers_write_no_artifact_at_all():
    """Source contract: nothing inside run_audits touches the filesystem."""
    source = Path(run_pipeline.__file__).read_text(encoding="utf-8")
    block = source.split("def run_audits(")[1].split("\ndef ")[0]
    # Scan the CODE, not the prose that explains the contract.
    code = _code_only(block)
    for forbidden in ("write_text", "write_json_atomic", "json.dump", "open(",
                      "assessment.txt", "strategy.json", "Tracker(", "os.replace"):
        assert forbidden not in code, f"run_audits must not reference {forbidden}"
    # The workers are pure functions that RETURN their result.
    for worker in ("def application_worker()", "def research_worker()"):
        body = _code_only(
            block.split(worker)[1].split("\n    def ")[0].split("\n    out")[0])
        assert "return client." in body, worker
        # The only thing a worker does is call the provider and hand back the
        # reply: no filesystem verb appears in its body.
        for verb in ("write", "dump", "replace", "unlink", "mkdir"):
            assert verb not in body, f"{worker} touches {verb}"


def test_the_workers_receive_copies_not_the_live_structures(run, c3):
    """A worker cannot mutate what the coordinator or caller still uses."""
    source = Path(run_pipeline.__file__).read_text(encoding="utf-8")
    block = source.split("def run_audits(")[1].split("\ndef ")[0]
    assert "context_copy = dict(experience_context or {})" in block
    assert "catalogue_copy = [dict(entry) for entry in catalogue]" in block
    assert "bullets_copy = {pid: list(values)" in block
    # No `strategy` parameter exists at all, so no writable strategy can be
    # handed to a worker by accident.
    signature = source.split("def run_audits(")[1].split(") -> dict:")[0]
    assert "strategy" not in signature


def test_strategy_json_is_written_exactly_once_per_run():
    source = Path(run_pipeline.__file__).read_text(encoding="utf-8")
    # One write in the generation path, one in --assess, both atomic.
    assert source.count("write_json_atomic(") == 3      # definition + 2 call sites
    assert 'strategy_path.write_text' not in source
    assert 'json.dumps(strategy' not in source
    for block_name in ("def run_one(", "def assess_run("):
        block = source.split(block_name)[1].split("\ndef ")[0]
        assert block.count("write_json_atomic(") == 1, block_name


def test_both_audit_results_survive_the_merge_in_either_order(run, c3):
    for delays in ({"assessment": 0.2}, {"company_research": 0.2}):
        transport = _AuditTransport(
            {"assessment": _assessment_payload_for(c3),
             "company_research": _research_payload(job_posted="2026-09-01",
                                                   job_posted_confidence="HIGH")},
            delays=delays)
        _client, audits = _run_audits(run, c3, transport)
        # Neither result is lost, whichever thread got there first.
        assert audits["application_audit"]["experience_selection_score"] == 8.4
        assert audits["research"]["job_posted"] == "2026-09-01"
        report = run_pipeline.assessment_report(audits, 9.0)
        assert report["Experience Selection Score"] == "8.4 / 10"
        assert report["Job Posted"] == "2026-09-01"


# ---- atomic persistence ---------------------------------------------------

def test_the_atomic_writer_leaves_valid_json(tmp_path):
    target = tmp_path / "strategy.json"
    run_pipeline.write_json_atomic(target, {"a": 1, "nested": {"b": [1, 2]}})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 1, "nested": {"b": [1, 2]}}
    # Rewriting replaces cleanly and leaves no temp files behind.
    run_pipeline.write_json_atomic(target, {"a": 2})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 2}
    assert list(tmp_path.iterdir()) == [target]


def test_an_interrupted_write_keeps_the_previous_valid_strategy(tmp_path, monkeypatch):
    target = tmp_path / "strategy.json"
    run_pipeline.write_json_atomic(target, {"version": "good"})

    class _Unserializable:
        pass

    # json.dump raises part-way through, after the temp file was created.
    with pytest.raises(TypeError):
        run_pipeline.write_json_atomic(target, {"version": "bad", "x": _Unserializable()})
    assert json.loads(target.read_text(encoding="utf-8")) == {"version": "good"}
    assert list(tmp_path.iterdir()) == [target], "the temp file must be cleaned up"

    # And an interruption during os.replace leaves the original untouched too.
    monkeypatch.setattr(run_pipeline.os, "replace",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("interrupted")))
    with pytest.raises(OSError):
        run_pipeline.write_json_atomic(target, {"version": "also bad"})
    assert json.loads(target.read_text(encoding="utf-8")) == {"version": "good"}


def test_the_writer_uses_a_process_unique_temp_name():
    source = Path(run_pipeline.__file__).read_text(encoding="utf-8")
    block = source.split("def write_json_atomic(")[1].split("\ndef ")[0]
    assert "os.getpid()" in block
    assert "os.replace(temp, path)" in block
    assert "os.fsync(" in block
    assert "path.with_name(" in block, "the temp must be in the SAME directory"


# ---- the call shape after the refactor -----------------------------------

def test_a_clean_run_makes_exactly_two_groq_calls():
    folders = sorted((ROOT / "output" / "_smoke_tests").glob("C3_AI_*"))
    if not folders:
        pytest.skip("no C3 smoke folder; run the C3 mock first")
    log = (folders[-1] / "run.log").read_text(encoding="utf-8")
    purposes = re.findall(r"purpose=([a-z_]+)", log)
    audits = [p for p in purposes if p in ("assessment", "company_research")]
    assert sorted(audits) == ["assessment", "company_research"], audits
    assert "experience_audit" not in purposes, "the third audit call must be gone"
    # Gemini's five logical calls: selection, three bullet writers, the letter.
    assert purposes.count("project_selection") == 1
    assert purposes.count("cover_letter") == 1
    assert purposes.count("project_bullets") >= 3


def test_no_groq_experience_audit_remains_anywhere():
    for module in (grounding, run_pipeline, _llm_client()):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "audit_experience_selection" not in source, module.__name__
        assert "experience_option_space" not in source, module.__name__
    # The score itself survives, sourced from the final assessment.
    assert "Experience Selection Score" in grounding.ASSESSMENT_FIELDS


# ================== AK. research source scoping, keys, daily limits, --assess

def _scoped(scope: str, evidence: str, title: str = "Source") -> dict:
    return {"title": title, "url": "https://example.com/x", "scope": scope,
            "evidence": evidence}


# ---- source scoping -------------------------------------------------------

def test_an_unrelated_posting_cannot_establish_a_company_level_no():
    """The live C3 lesson: another team's requisition is not company policy."""
    research, _ = grounding.validate_company_research(
        {"company_visa_sponsorship": "NO", "company_visa_confidence": "HIGH",
         "checked_at": "2026-09-16",
         "sources": [_scoped("unrelated_posting",
                             "This role does not offer visa sponsorship.",
                             "A different job at the same company")]},
        jd_text="We are hiring a backend engineer.")
    assert research["company_visa_sponsorship"] == "UNKNOWN"
    assert research["company_visa_confidence"] == "LOW"
    assert any("cannot establish a company-wide position" in note
               for note in research["deterministic_overrides"])


def test_an_official_company_policy_can_establish_yes_or_no():
    for verdict in ("YES", "NO"):
        research, _ = grounding.validate_company_research(
            {"company_visa_sponsorship": verdict, "company_visa_confidence": "HIGH",
             "checked_at": "2026-09-16",
             "sources": [_scoped("company_policy",
                                 "Our published policy on work authorization.")]},
            jd_text="We are hiring a backend engineer.")
        assert research["company_visa_sponsorship"] == verdict


def test_an_exact_job_restriction_overrides_a_company_level_yes():
    research, _ = grounding.validate_company_research(
        {"company_visa_sponsorship": "YES", "company_visa_confidence": "HIGH",
         "checked_at": "2026-09-16",
         "sources": [_scoped("company_policy", "We sponsor visas company-wide."),
                     _scoped("this_job",
                             "We are not able to sponsor visas for this position.")]},
        jd_text="We are hiring a backend engineer.")
    assert research["company_visa_sponsorship"] == "NO"
    assert research["company_visa_confidence"] == "HIGH"
    assert any("job-specific restriction overrides" in note
               for note in research["deterministic_overrides"])


def test_everify_inside_an_authoritative_source_still_cannot_prove_stem_opt():
    research, _ = grounding.validate_company_research(
        {"company_stem_opt_support": "YES", "company_stem_opt_confidence": "HIGH",
         "checked_at": "2026-09-16",
         "sources": [_scoped("company_policy",
                             "We are an E-Verify participating employer.")]},
        jd_text="We are hiring.")
    assert research["company_stem_opt_support"] == "UNKNOWN"


def test_scope_labels_are_normalized_and_unknown_scopes_become_context():
    for raw, expected in (("this-job", "this_job"), ("Company Wide", "company_policy"),
                          ("other_job", "unrelated_posting"), ("gossip", "context"),
                          (None, "context")):
        research, _ = grounding.validate_company_research(
            {"checked_at": "x", "sources": [_scoped(raw, "some evidence here")]},
            jd_text="We are hiring.")
        assert research["sources"][0]["scope"] == expected, raw
    assert grounding.SOURCE_SCOPES == ("this_job", "company_policy",
                                       "unrelated_posting", "context")


def test_the_research_prompt_demands_a_scope_per_source(run, c3):
    request = _captured_audit_requests(run, c3)["company_research"]
    assert "SOURCE SCOPING" in request.prompt
    for scope in grounding.SOURCE_SCOPES:
        assert scope in request.prompt, scope
    assert "unrelated posting is supporting context only" in request.prompt


# ---- key pool -------------------------------------------------------------

def test_the_key_pool_is_deterministic_and_supports_both_schemes(monkeypatch, tmp_path):
    import llm_client

    monkeypatch.setattr(engine, "PROJECT_ROOT", tmp_path)
    for name in ("GROQ_API_KEY", "GROQ_API_KEY_1", "GROQ_API_KEY_2"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / ".env").write_text(
        "GROQ_API_KEY_1=first\nGROQ_API_KEY_2=second\nGROQ_API_KEY=legacy\n",
        encoding="utf-8")
    assert llm_client.groq_key_pool() == ("first", "second", "legacy")
    # Stable across calls, and duplicates collapse.
    (tmp_path / ".env").write_text("GROQ_API_KEY_1=same\nGROQ_API_KEY=same\n",
                                   encoding="utf-8")
    assert llm_client.groq_key_pool() == ("same",)
    # Legacy alone still works.
    (tmp_path / ".env").write_text("GROQ_API_KEY=only\n", encoding="utf-8")
    assert llm_client.groq_key_pool() == ("only",)


def test_a_rate_limit_never_rotates_the_key(monkeypatch):
    """Two keys in one organization share every ceiling."""
    import llm_client

    monkeypatch.setattr(llm_client.time, "sleep", lambda _s: None)
    seen: list[int] = []

    def fail(self, request):
        seen.append(self._key_index)
        raise llm_client.ProviderError("rate_limited", "TPM exceeded")

    transport = _stub_groq(generate=fail)
    transport.keys = ("k1", "k2")
    with pytest.raises(llm_client.ProviderError):
        transport.generate(llm_client.Request("assessment", "p"))
    assert set(seen) == {0}, "a rate limit must not spend the second key"
    assert transport._key_index == 0


@pytest.mark.parametrize("category", ["auth_permission", "invalid_key"])
def test_a_key_specific_failure_rotates_once_per_key(category, monkeypatch):
    import llm_client

    monkeypatch.setattr(llm_client.time, "sleep", lambda _s: None)
    seen: list[int] = []

    def fail(self, request):
        seen.append(self._key_index)
        raise llm_client.ProviderError(category, "bad key")

    transport = _stub_groq(generate=fail)
    transport.keys = ("k1", "k2")
    with pytest.raises(llm_client.ProviderError):
        transport.generate(llm_client.Request("assessment", "p"))
    # Each key is tried, and the loop is bounded: no infinite rotation.
    assert seen == [0, 1], seen
    assert len(seen) <= llm_client.GROQ_MAX_ATTEMPTS


def test_keys_are_never_logged(run, c3):
    import logging

    import llm_client

    records: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    logger = logging.getLogger("key-safety")
    logger.addHandler(_Capture())
    logger.setLevel(logging.INFO)
    transport = llm_client.GroqTransport(_credentials(), logger, model="m",
                                         fallback=None, keys=("supersecretkey",),
                                         budget=llm_client.TokenBudget(999999))

    def boom(request):
        raise llm_client.ProviderError("server_error", "500")

    transport._generate = boom
    with pytest.raises(llm_client.ProviderError):
        transport.generate(llm_client.Request("assessment", "p"))
    joined = " ".join(records)
    assert "supersecretkey" not in joined
    assert "key=#1" in joined


# ---- daily exhaustion is not transient -----------------------------------

@pytest.mark.parametrize("body", [
    "Rate limit reached: tokens per day (TPD) limit of 100000 exceeded",
    "Rate limit reached for requests per day (RPD)",
    "You have exhausted your daily quota",
])
def test_a_daily_ceiling_is_classified_and_never_retried(body, monkeypatch):
    import llm_client

    assert llm_client.classify_http(429, body) == "daily_limit_exhausted"
    monkeypatch.setattr(llm_client.time, "sleep", lambda _s: None)
    attempts = []

    def fail(self, request):
        attempts.append(1)
        raise llm_client.ProviderError("daily_limit_exhausted", body)

    transport = _stub_groq(generate=fail)
    with pytest.raises(llm_client.ProviderError):
        transport.generate(llm_client.Request("assessment", "p"))
    assert attempts == [1], "a daily ceiling will not clear inside this run"


def test_a_per_minute_limit_still_retries(monkeypatch):
    import llm_client

    monkeypatch.setattr(llm_client.time, "sleep", lambda _s: None)
    assert llm_client.classify_http(
        429, "Rate limit reached: tokens per minute (TPM)") == "rate_limited"
    attempts = []

    def fail(self, request):
        attempts.append(1)
        raise llm_client.ProviderError("rate_limited", "TPM")

    transport = _stub_groq(generate=fail)
    with pytest.raises(llm_client.ProviderError):
        transport.generate(llm_client.Request("assessment", "p"))
    assert len(attempts) == llm_client.GROQ_MAX_ATTEMPTS


def test_the_non_retryable_set_is_explicit():
    import llm_client

    for category in ("request_too_large", "output_truncated", "daily_limit_exhausted"):
        assert category in llm_client.NON_RETRYABLE_CATEGORIES, category
    for category in ("rate_limited", "server_error", "transport"):
        assert category not in llm_client.NON_RETRYABLE_CATEGORIES, category


# ---- standalone --assess --------------------------------------------------

def test_assess_loads_strategy_once_and_persists_it_once():
    source = Path(run_pipeline.__file__).read_text(encoding="utf-8")
    block = _code_only(source.split("def assess_run(")[1].split("\ndef ")[0])
    assert block.count("strategy_path.read_text") == 1, "load exactly once"
    assert block.count("write_json_atomic(") == 1, "persist exactly once"
    assert "json.loads(strategy_path.read_text" in block
    # And the audit workers still never write: assess_run delegates to the
    # same coordinator the generation path uses.
    assert "run_audits(" in block


def test_assess_regenerates_assessment_txt_from_the_merged_results():
    run_dir = _c3_smoke_folder()
    result = run_pipeline.assess_run(run_dir, mock=True, console=False)
    assert result.status == "success", result.issues
    rendered = (run_dir / "assessment.txt").read_text(encoding="utf-8")
    strategy = json.loads((run_dir / "strategy.json").read_text(encoding="utf-8"))
    audit = strategy["audit"]
    # Both halves of the merge are present in the persisted strategy.
    assert audit["application"], "the assessment result is missing"
    assert audit["company_research"], "the research result is missing"
    # And assessment.txt is rendered from that same merged report.
    assert rendered == grounding.render_assessment_txt(audit["report"])
    assert audit["report"]["Experience Selection Score"] == grounding._fmt(
        audit["application"].get("experience_selection_score"))


# ============ AL. end-to-end: a recovered semantic signal changes Experience
#
# The whole point of the recall layer, proved through the real pipeline: a
# posting whose code-quality emphasis is purely semantic ships the EXISTING
# approved PR-review bullet, chosen by Python, with no model-authored wording
# anywhere in Professional Experience.

def test_end_to_end_a_recovered_signal_ships_the_approved_pr_review_bullet(
        tmp_path, monkeypatch):
    import llm_client

    jd_path = tmp_path / "meridian.txt"
    jd_path.write_text(SEMANTIC_JD, encoding="utf-8")
    master, policy = engine.load_master(), engine.load_policy()
    template = engine.load_template()

    # Baseline: deterministic classification alone misses the signal.
    signals = engine.classify_jd(SEMANTIC_JD, master.section_order)
    assert signals.code_quality is False
    baseline = engine.select_experience(template, policy, signals)

    # The mock reads the posting semantically for this run, exactly as the
    # live model is asked to: quote the phrase, claim the signal.
    real_signals = llm_client.MockTransport._semantic_signals

    def recovering(self, jd_text):
        found = real_signals(self, jd_text)
        if "participate in design and code reviews" in jd_text:
            found["code_quality_collaboration_heavy"] = {
                "present": True,
                "evidence": ["participate in design and code reviews"]}
        return found

    monkeypatch.setattr(llm_client.MockTransport, "_semantic_signals", recovering)
    result = run_pipeline.run_one(jd_path, mock=True, smoke=True, console=False)
    assert result.status == "success", result.issues

    strategy = result.strategy
    block = strategy["jd_signals"]
    # 1. Gemini's claim was validated, not merely trusted.
    assert block["gemini_semantic_signals_validated"][
        "code_quality_collaboration_heavy"] is True
    assert block["deterministic_signals"]["code_quality_collaboration_heavy"] is False
    assert block["merged_signals"]["code_quality_collaboration_heavy"] is True
    assert block["recovered_by_gemini"] == ["code_quality_collaboration_heavy"]

    # 2. PYTHON then applied its own existing rule.
    assert strategy["experience"]["rule_fired"] == (
        "Non-healthcare + code-quality/collaboration-heavy (jd_signal)")
    assert strategy["experience"]["resolved_from"] == (
        "merged deterministic + validated Gemini signals")
    assert strategy["experience"]["shipped_ids"] != baseline.shipped_ids

    # 3. The shipped bullet is the EXISTING approved PR-review variant, by id.
    swapped = {row["target_id"] for row in strategy["experience"]["swaps"]}
    assert any("PR-REVIEW" in target for target in swapped), swapped

    # 4. No model-authored Experience wording exists. Every shipped line is
    #    approved wording byte for byte, and it is what the PDF renders.
    approved = {e.plain for e in policy.experience_library}
    approved |= {e.plain for e in policy.alternate_library}
    for row in strategy["experience"]["shipped"]:
        assert row["text"] in approved, row["bullet_id"]
    resume_text = (result.run_dir / "resume.txt").read_text(encoding="utf-8")

    def flatten(text: str) -> str:
        # pdftotext renders "-" as U+2212 and wraps lines; compare on letters.
        return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()

    flat = flatten(resume_text)
    shipped_lines = [row["text"] for row in strategy["experience"]["shipped"]
                     if not row["bullet_id"].startswith("ROLE-")]
    assert shipped_lines
    for text in shipped_lines:
        assert flatten(text)[:60] in flat, text[:60]


def test_end_to_end_an_unrecoverable_claim_changes_nothing(tmp_path, monkeypatch):
    """Ungrounded evidence leaves the deterministic decision exactly as it was."""
    import llm_client

    jd_path = tmp_path / "meridian2.txt"
    # A distinct company, so this run cannot overwrite the other test's folder.
    jd_path.write_text(SEMANTIC_JD.replace("Meridian Systems", "Halyard Systems"),
                       encoding="utf-8")
    real_signals = llm_client.MockTransport._semantic_signals

    def hallucinating(self, jd_text):
        found = real_signals(self, jd_text)
        found["healthcare"] = {"present": True,
                               "evidence": ["we build clinical software for hospitals"]}
        return found

    monkeypatch.setattr(llm_client.MockTransport, "_semantic_signals", hallucinating)
    result = run_pipeline.run_one(jd_path, mock=True, smoke=True, console=False)
    assert result.status == "success", result.issues

    block = result.strategy["jd_signals"]
    assert block["gemini_semantic_signals_validated"]["healthcare"] is False
    assert block["merged_signals"]["healthcare"] is False
    assert block["recovered_by_gemini"] == []
    assert any("none of its evidence appears" in note for note in block["rejected"])
    assert result.strategy["experience"]["resolved_from"] == "deterministic signals only"
    assert not result.strategy["experience"]["rule_fired"].startswith("Healthcare")


# ===================== AM. OTPM: output tokens per minute is its own ceiling
#
# The first live run: Qwen accepted the request, then refused with
# "OTPM Limit 1000, Requested 1178". That is the OUTPUT allowance, separate
# from the combined 8K TPM budget, and it needs its own reservation.

QWEN = "qwen/qwen3.8-27b"
RESEARCHER = "openai/gpt-oss-20b"


def _otpm_env(monkeypatch, tmp_path, **settings):
    lines = [f"{k}={v}" for k, v in settings.items()]
    (tmp_path / ".env").write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setattr(engine, "PROJECT_ROOT", tmp_path)
    for name in ("GROQ_OTPM_LIMIT", "GROQ_OTPM_LIMIT__QWEN_QWEN3_8_27B",
                 "GROQ_OTPM_LIMIT__OPENAI_GPT_OSS_20B", "GROQ_TPM_LIMIT"):
        monkeypatch.delenv(name, raising=False)
    import llm_client

    llm_client._MODEL_OUTPUT_BUDGETS.clear()


# ---- 1/2. the cap itself --------------------------------------------------

def test_the_assessment_reserves_at_most_900_completion_tokens(run, c3):
    import llm_client

    assert llm_client.AUDIT_OUTPUT_TOKENS["assessment"] <= 900
    request = _captured_audit_requests(run, c3)["assessment"]
    assert request.max_tokens == 900
    # Research is deliberately untouched.
    assert llm_client.AUDIT_OUTPUT_TOKENS["company_research"] == 1000


def test_the_sdk_is_sent_max_completion_tokens_not_max_tokens():
    source = Path(_llm_client().__file__).read_text(encoding="utf-8")
    generate = source.split("def _generate(self, request: Request) -> str:")[1]
    generate = generate.split("\n\n\n")[0]
    assert "max_completion_tokens=request.max_tokens" in generate
    assert "\n                max_tokens=" not in generate


def test_the_assessment_sends_only_reasoning_effort_none(run, c3):
    import llm_client

    request = _captured_audit_requests(run, c3)["assessment"]
    assert request.reasoning_effort == "none"
    assert request.reasoning_format is None
    assert request.include_reasoning is None

    # Prove what the SDK would actually receive: reasoning_effort and nothing
    # else from the reasoning family.
    settings = {}
    if request.json and not request.response_schema:
        settings["response_format"] = {"type": "json_object"}
    if request.response_schema:
        settings["response_format"] = request.response_schema
    if request.reasoning_effort:
        settings["reasoning_effort"] = request.reasoning_effort
    if request.reasoning_format:
        settings["reasoning_format"] = request.reasoning_format
    elif request.include_reasoning is not None:
        settings["include_reasoning"] = request.include_reasoning
    assert settings["reasoning_effort"] == "none"
    assert "reasoning_format" not in settings
    assert "include_reasoning" not in settings

    # The two spellings remain mutually exclusive in the transport.
    source = Path(llm_client.__file__).read_text(encoding="utf-8")
    generate = source.split("def _generate(self, request: Request) -> str:")[1]
    generate = generate.split("\n\n\n")[0]
    assert 'settings["reasoning_format"] = request.reasoning_format' in generate
    assert "elif request.include_reasoning is not None:" in generate


def test_the_assessment_schema_fits_the_compact_contract():
    """Ids-only buckets: the prose Python discards is no longer requested."""
    ids = {f"REQ-{i:03d}" for i in range(1, 26)}
    payload = {
        "fit_score": 7.4, "recommendation": "apply",
        "summary": "Most scored requirements are evidenced on the shipped resume.",
        "eligibility": {"status": "uncertain", "details": []},
        "strong_matches": [{"requirement_id": f"REQ-{i:03d}"} for i in range(1, 16)],
        "partial_matches": [{"requirement_id": f"REQ-{i:03d}"} for i in range(16, 21)],
        "gaps": [{"requirement_id": f"REQ-{i:03d}"} for i in range(21, 24)],
        "manual_review": [{"requirement_id": f"REQ-{i:03d}"} for i in range(24, 26)],
        "complementary_strengths": ["AWS", "PostgreSQL", "WebSockets"],
        "tailoring_quality": {"score": 9.0, "notes": ["strong use of evidence"]},
        "experience_selection": {"score": 8.5, "notes": ["the rule fits the posting"]},
        "project_selection": {"score": 8.5, "components": {
            "jd_relevance": 8.0, "best_available_chosen": 9.0,
            "complementary_coverage": 8.0, "ranking_and_allocation": 9.0}, "notes": []},
        "project_bullets": {"score": 8.6, "components": {
            "jd_relevance": 9.0, "technical_specificity": 9.0, "evidence_fidelity": 10.0,
            "impact_ownership": 8.0, "non_redundancy": 8.0}, "notes": []},
        "callback_likelihood": "MEDIUM", "risk_flags": ["thin cloud evidence"],
    }
    # It validates completely with no per-entry prose at all.
    assert grounding.errors(grounding.validate_assessment(
        payload, "A posting.", ids)) == []
    # Every field assessment.txt needs is present and readable.
    audit, problems = grounding.validate_application_audit(payload)
    assert problems == []
    for key in ("experience_selection_score", "resume_tailoring_score",
                "project_selection_score", "project_bullet_score",
                "callback_likelihood"):
        assert audit[key] is not None, key
    # And it fits the budget with room to spare.
    assert len(json.dumps(payload)) // 4 < 900

    # A model that still volunteers an enum is held to it.
    payload["strong_matches"][0]["source"] = "not-a-source"
    assert grounding.errors(grounding.validate_assessment(payload, "A posting.", ids))
    # An invented id is still rejected: that check is the point of the buckets.
    payload["strong_matches"][0]["source"] = "experience"
    payload["gaps"].append({"requirement_id": "REQ-999"})
    problems = grounding.errors(grounding.validate_assessment(payload, "A posting.", ids))
    assert any("unknown requirement_id" in p.message for p in problems)


# ---- 3. the independent output budget ------------------------------------

def test_qwen_has_an_independent_1000_token_output_budget(monkeypatch, tmp_path):
    import llm_client

    _otpm_env(monkeypatch, tmp_path, GROQ_OTPM_LIMIT__QWEN_QWEN3_8_27B="1000")
    assert llm_client.model_otpm_limit(QWEN) == 1000
    budget = llm_client.output_budget_for(QWEN)
    assert budget is not None and budget.limit == 1000
    # Separate object from the COMBINED budget for the same model.
    assert budget is not llm_client.budget_for(QWEN)
    # A model with no configured ceiling is not paced on output at all.
    assert llm_client.output_budget_for(RESEARCHER) is None


def test_a_global_otpm_fallback_is_honoured(monkeypatch, tmp_path):
    import llm_client

    _otpm_env(monkeypatch, tmp_path, GROQ_OTPM_LIMIT="1500")
    assert llm_client.model_otpm_limit(QWEN) == 1500
    assert llm_client.model_otpm_limit(RESEARCHER) == 1500
    # A model-specific value still wins.
    monkeypatch.setenv("GROQ_OTPM_LIMIT__QWEN_QWEN3_8_27B", "1000")
    assert llm_client.model_otpm_limit(QWEN) == 1000
    assert llm_client.model_otpm_limit(RESEARCHER) == 1500


def test_qwens_output_budget_cannot_block_gpt_oss(monkeypatch, tmp_path):
    import llm_client

    _otpm_env(monkeypatch, tmp_path, GROQ_OTPM_LIMIT__QWEN_QWEN3_8_27B="1000")
    qwen = llm_client.output_budget_for(QWEN)
    qwen.record(1000)                     # the whole minute is spent
    assert llm_client.output_budget_for(RESEARCHER) is None, "research is unpaced"

    # Even when research DOES have a ceiling, it is a different bucket.
    monkeypatch.setenv("GROQ_OTPM_LIMIT__OPENAI_GPT_OSS_20B", "6000")
    llm_client._MODEL_OUTPUT_BUDGETS.pop(RESEARCHER, None)
    research = llm_client.output_budget_for(RESEARCHER)
    assert research is not qwen
    slept: list[float] = []
    research._sleep = lambda s: slept.append(s)
    assert research.wait_for(1000) == 0.0
    assert slept == [], "research paced on Qwen's output window"


def test_two_rapid_assessments_are_paced_by_the_output_window(monkeypatch, tmp_path):
    """900 + 900 exceeds 1000/min, so the second call waits for the window."""
    import llm_client

    _otpm_env(monkeypatch, tmp_path, GROQ_OTPM_LIMIT__QWEN_QWEN3_8_27B="1000")
    state = {"now": 0.0, "slept": []}
    budget = llm_client.output_budget_for(QWEN)
    budget._sleep = lambda s: (state["slept"].append(s), state.__setitem__("now",
                                                                          state["now"] + s))
    budget._clock = lambda: state["now"]

    calls = []

    def answer(self, request):
        calls.append(request.purpose)
        self.last_output_usage = 880      # nearly the whole reservation
        return "{}"

    transport = _stub_groq(generate=answer, model=QWEN,
                           budget=llm_client.TokenBudget(1_000_000))
    transport.output_budget = budget
    request = lambda: llm_client.Request(            # noqa: E731 - test brevity
        "assessment", "p" * 3000, max_tokens=900, **llm_client.AUDIT_REASONING)

    transport.generate(request())
    assert state["slept"] == [], "the first call has the whole window"
    transport.generate(request())
    assert state["slept"], "the second call must wait for the output window"
    assert sum(state["slept"]) == pytest.approx(60.0)
    assert calls == ["assessment", "assessment"]


def test_actual_output_usage_replaces_the_reservation(monkeypatch, tmp_path):
    import llm_client

    _otpm_env(monkeypatch, tmp_path, GROQ_OTPM_LIMIT__QWEN_QWEN3_8_27B="1000")
    budget = llm_client.output_budget_for(QWEN)

    def answer(self, request):
        self.last_output_usage = 240      # far below the 900 cap
        self.last_usage = 4000
        return "{}"

    transport = _stub_groq(generate=answer, model=QWEN,
                           budget=llm_client.TokenBudget(1_000_000))
    transport.output_budget = budget
    transport.generate(llm_client.Request("assessment", "p" * 3000, max_tokens=900))
    assert [t for _, t in budget.used] == [240], budget.used
    # So a second call is NOT paced: 240 + 900 still fits 1000... just.
    slept: list[float] = []
    budget._sleep = lambda s: slept.append(s)
    assert budget.wait_for(760) == 0.0
    assert slept == []


def test_a_refused_request_releases_its_output_reservation(monkeypatch, tmp_path):
    import llm_client

    _otpm_env(monkeypatch, tmp_path, GROQ_OTPM_LIMIT__QWEN_QWEN3_8_27B="1000")
    budget = llm_client.output_budget_for(QWEN)

    def refuse(self, request):
        raise llm_client.ProviderError("output_rate_limited",
                                       "Limit 1000, Requested 1178 OTPM")

    transport = _stub_groq(generate=refuse, model=QWEN,
                           budget=llm_client.TokenBudget(1_000_000))
    transport.output_budget = budget
    monkeypatch.setattr(llm_client.time, "sleep", lambda _s: None)
    with pytest.raises(llm_client.ProviderError):
        transport.generate(llm_client.Request("assessment", "p" * 3000, max_tokens=900))
    # Nothing was generated, so nothing may sit in the output window.
    assert budget.used == [], budget.used


# ---- 4. classification ----------------------------------------------------

@pytest.mark.parametrize("body", [
    "Rate limit reached: Limit 1000, Requested 1178 on output tokens per minute (OTPM)",
    "OTPM limit exceeded for this model",
])
def test_otpm_is_classified_separately_from_tpm(body):
    import llm_client

    assert llm_client.classify_http(429, body) == "output_rate_limited"
    assert llm_client.classify_exception(Exception(body)) == "output_rate_limited"
    # The combined ceiling and an oversize request keep their own categories.
    assert llm_client.classify_http(
        429, "tokens per minute (TPM): Limit 8000") == "rate_limited"
    assert llm_client.classify_http(
        413, "Request too large for model") == "request_too_large"
    assert llm_client.classify_http(
        429, "tokens per day (TPD) limit") == "daily_limit_exhausted"


def test_a_cap_above_the_otpm_ceiling_is_a_configuration_error(monkeypatch, tmp_path):
    """Retrying identical bytes cannot help; say so instead."""
    import llm_client

    _otpm_env(monkeypatch, tmp_path, GROQ_OTPM_LIMIT__QWEN_QWEN3_8_27B="1000")
    attempts = []

    def answer(self, request):
        attempts.append(1)
        return "{}"

    transport = _stub_groq(generate=answer, model=QWEN,
                           budget=llm_client.TokenBudget(1_000_000))
    transport.output_budget = llm_client.output_budget_for(QWEN)
    with pytest.raises(llm_client.ProviderError) as raised:
        transport.generate(llm_client.Request("assessment", "p", max_tokens=2600))
    assert raised.value.category == "bad_request"
    assert "exceeds the configured OTPM ceiling" in str(raised.value)
    assert attempts == [], "the request must never be sent"


def test_a_rolling_otpm_refusal_is_retried_not_abandoned(monkeypatch, tmp_path):
    import llm_client

    _otpm_env(monkeypatch, tmp_path, GROQ_OTPM_LIMIT__QWEN_QWEN3_8_27B="1000")
    waits: list[float] = []
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: waits.append(s))
    attempts = []

    def flaky(self, request):
        attempts.append(len(attempts))
        if len(attempts) == 1:
            raise llm_client.ProviderError(
                "output_rate_limited",
                "Rate limit reached. Please try again in 2.5s (OTPM)")
        self.last_output_usage = 300
        return "{}"

    transport = _stub_groq(generate=flaky, model=QWEN,
                           budget=llm_client.TokenBudget(1_000_000))
    transport.output_budget = llm_client.output_budget_for(QWEN)
    reply = transport.generate(llm_client.Request("assessment", "p" * 100, max_tokens=900))
    assert reply.text == "{}"
    assert len(attempts) == 2
    # The provider's own retry-after was honoured.
    assert waits == [pytest.approx(2.5)]


def test_an_otpm_limit_never_rotates_the_api_key(monkeypatch):
    """Two keys in one organization share the output ceiling too."""
    import llm_client

    monkeypatch.setattr(llm_client.time, "sleep", lambda _s: None)
    seen: list[int] = []

    def refuse(self, request):
        seen.append(self._key_index)
        raise llm_client.ProviderError("output_rate_limited", "OTPM limit reached")

    transport = _stub_groq(generate=refuse, model=QWEN)
    transport.keys = ("k1", "k2")
    with pytest.raises(llm_client.ProviderError):
        transport.generate(llm_client.Request("assessment", "p", max_tokens=900))
    assert set(seen) == {0}, "an output rate limit must not spend the second key"
    assert transport._key_index == 0
    assert "output_rate_limited" not in llm_client.KEY_SPECIFIC_CATEGORIES


# ---- 5. nothing else moved ------------------------------------------------

def test_the_research_path_is_byte_for_byte_unchanged(run, c3):
    import llm_client

    request = _captured_audit_requests(run, c3)["company_research"]
    assert request.max_tokens == 1000
    assert request.web_search is True
    assert request.json is False
    assert request.reasoning_effort == "low"
    source = Path(llm_client.__file__).read_text(encoding="utf-8")
    generate = source.split("def _generate(self, request: Request) -> str:")[1]
    generate = generate.split("\n\n\n")[0]
    assert 'settings["tool_choice"] = "required"' in generate
    assert 'settings["tools"] = [{"type": "browser_search"}]' in generate


def test_parallelism_and_the_single_writer_are_untouched():
    source = Path(run_pipeline.__file__).read_text(encoding="utf-8")
    block = source.split("def run_audits(")[1].split("\ndef ")[0]
    assert "ThreadPoolExecutor(max_workers=2)" in block
    assert block.count("pool.submit(") == 2
    code = _code_only(block)
    for forbidden in ("write_text", "write_json_atomic", "json.dump"):
        assert forbidden not in code
    assert source.count("write_json_atomic(") == 3


# ================ AN. ITPM: input tokens per minute is a third ceiling
#
# The second live run: Qwen counted 7256 input tokens against a 7000 ITPM
# limit while the local estimate said 6169. Two defects at once - the
# assessment prompt was sending the generation corpus twice, and the generic
# estimator was 18% optimistic for Qwen's tokenizer.

def _itpm_env(monkeypatch, tmp_path, **settings):
    lines = [f"{k}={v}" for k, v in settings.items()]
    (tmp_path / ".env").write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setattr(engine, "PROJECT_ROOT", tmp_path)
    for name in ("GROQ_ITPM_LIMIT", "GROQ_ITPM_LIMIT__QWEN_QWEN3_8_27B",
                 "GROQ_ITPM_LIMIT__OPENAI_GPT_OSS_20B"):
        monkeypatch.delenv(name, raising=False)
    import llm_client

    llm_client._MODEL_INPUT_BUDGETS.clear()


def _assessment_prompt(run, c3, jd_path=None):
    """The real assessment prompt, captured without a network call."""
    return _captured_audit_requests(run, c3)["assessment"].prompt


# ---- 1. the prompt actually shrank ---------------------------------------

def test_the_assessment_prompt_is_materially_smaller(run, c3):
    """Measured against the size that was refused live: 31,502 characters."""
    import llm_client

    prompt = _assessment_prompt(run, c3)
    assert len(prompt) < 31502 * 0.75, len(prompt)
    # And the refused request's own input count is no longer reachable.
    request = _captured_audit_requests(run, c3)["assessment"]
    honest = llm_client.estimate_input_tokens(request, QWEN, safety=False)
    assert honest < 7256, honest


def test_the_c3_assessment_estimate_is_below_the_safe_ceiling(run, c3, monkeypatch,
                                                              tmp_path):
    import llm_client

    _itpm_env(monkeypatch, tmp_path, GROQ_ITPM_LIMIT__QWEN_QWEN3_8_27B="7000")
    request = _captured_audit_requests(run, c3)["assessment"]
    adjusted = llm_client.estimate_input_tokens(request, QWEN)
    assert adjusted <= llm_client.safe_input_ceiling(QWEN), adjusted
    assert llm_client.safe_input_ceiling(QWEN) == 6300


def test_a_large_posting_still_fits_the_itpm_ceiling(run, monkeypatch, tmp_path):
    """Superhuman has the longest JD; the JD itself is never truncated."""
    import llm_client

    _itpm_env(monkeypatch, tmp_path, GROQ_ITPM_LIMIT__QWEN_QWEN3_8_27B="7000")
    master = engine.load_master()
    jd = engine.read_jd(REDWOOD_JD)
    big = {"master": master, "jd": jd,
           "requirements": engine.extract_jd_requirements(jd.text, master)}
    request = _captured_audit_requests(run, big)["assessment"]
    honest = llm_client.estimate_input_tokens(request, QWEN, safety=False)
    assert honest < llm_client.model_itpm_limit(QWEN), honest
    # The posting is present in full: no silent truncation of the JD.
    assert jd.text.strip() in request.prompt


# ---- 2. no dimension lost, alternatives still visible --------------------

def test_no_assessment_dimension_was_removed(run, c3):
    prompt = _assessment_prompt(run, c3)
    for dimension in ("tailoring_quality", "experience_selection", "project_selection",
                      "project_bullets", "callback_likelihood", "fit_score",
                      "risk_flags"):
        assert dimension in prompt, dimension
    # And the compact report still renders all ten fields from them.
    audit, problems = grounding.validate_application_audit({
        "tailoring_quality": {"score": 9.0},
        "experience_selection": {"score": 8.5},
        "project_selection": {"score": 8.0, "components": {}},
        "project_bullets": {"score": 8.0, "components": {}},
        "callback_likelihood": "HIGH"})
    assert problems == []
    assert len(grounding.ASSESSMENT_FIELDS) == 10


def test_unselected_projects_keep_enough_to_judge_the_selection(run, c3):
    prompt = _assessment_prompt(run, c3)
    master = run["master"]
    selected = {"lms", "pintos", "temp"}
    unselected = [p for p in master.projects if p.project_id not in selected]
    assert unselected, "fixture assumption"
    for project in unselected:
        assert project.project_id in prompt, project.project_id
        # Title and at least one technology, so a better choice is spottable.
        assert project.name.split("(")[0].strip()[:20] in prompt, project.project_id
        assert project.tech[0] in prompt, project.project_id
    assert "NOT SELECTED" in prompt


def test_the_full_evidence_corpus_is_no_longer_duplicated(run, c3):
    """Each selected project's evidence appears once, as a short digest."""
    prompt = _assessment_prompt(run, c3)
    master = run["master"]
    for pid in ("lms", "pintos", "temp"):
        evidence = master.project(pid).evidence
        joined = " ".join(evidence)
        # The complete corpus is NOT sent; a digest of it is.
        assert joined not in prompt, f"{pid} full evidence was duplicated"
    assert "evidence digest" in prompt
    # The bullets - the artifact under review - are still there in full.
    assert "FINAL BULLETS" in prompt
    # The requirement table appears once, not as a table plus a verdict block.
    assert prompt.count("AUTHORITATIVE REQUIREMENTS") == 1
    assert "MANUAL-REVIEW REQUIREMENTS" not in prompt


def test_deterministic_verdicts_remain_available_one_line_each(run, c3):
    import llm_client

    prompt = _assessment_prompt(run, c3)
    master = engine.load_master()
    verdicts = engine.deterministic_assessment(
        c3["requirements"], master,
        engine.ResumeEvidence(experience_text="x", project_bullets=(("lms", "y"),),
                              skills=("Python",)), c3["jd"].text)
    for bucket, label in llm_client._VERDICT_LABELS.items():
        for entry in verdicts.get(bucket) or []:
            identifier = entry["requirement_id"]
            assert f"{identifier} |" in prompt, identifier
            line = next(l for l in prompt.splitlines() if l.strip().startswith(identifier))
            assert f"| {label} |" in line, line
    # Python's own evidence paragraphs are NOT resent.
    for entry in verdicts.get("strong_matches") or []:
        if entry.get("evidence"):
            assert entry["evidence"] not in prompt


# ---- 4/5. configuration and estimation ----------------------------------

def test_qwen_itpm_resolves_to_7000_from_the_environment(monkeypatch, tmp_path):
    import llm_client

    _itpm_env(monkeypatch, tmp_path, GROQ_ITPM_LIMIT__QWEN_QWEN3_8_27B="7000")
    assert llm_client.model_itpm_limit(QWEN) == 7000
    assert llm_client.safe_input_ceiling(QWEN) == 6300
    # The global fallback applies to a model with no specific setting.
    monkeypatch.setenv("GROQ_ITPM_LIMIT", "5000")
    assert llm_client.model_itpm_limit(RESEARCHER) == 5000
    assert llm_client.model_itpm_limit(QWEN) == 7000, "specific setting must win"
    # Unset means no input pacing at all.
    monkeypatch.delenv("GROQ_ITPM_LIMIT", raising=False)
    monkeypatch.delenv("GROQ_ITPM_LIMIT__QWEN_QWEN3_8_27B", raising=False)
    (tmp_path / ".env").write_text("", encoding="utf-8")
    assert llm_client.model_itpm_limit(RESEARCHER) == 0
    assert llm_client.input_budget_for(RESEARCHER) is None


def test_qwen_uses_a_conservative_model_specific_input_estimate():
    import llm_client

    # Calibrated on the live refusal: 31,502 chars were counted as 7,256.
    request = llm_client.Request("assessment", "x" * 31502, max_tokens=900)
    honest = llm_client.estimate_input_tokens(request, QWEN, safety=False)
    assert 7000 <= honest <= 7500, honest
    adjusted = llm_client.estimate_input_tokens(request, QWEN)
    assert adjusted > honest, "the safety factor must inflate the reservation"
    assert adjusted == pytest.approx(honest * 1.25, rel=0.02)
    assert llm_client.input_safety_factor(QWEN) == 1.25
    # A model with no calibration keeps the generic assumption.
    generic = llm_client.estimate_input_tokens(request, "other/model")
    assert generic < honest
    assert llm_client.input_safety_factor("other/model") == 1.0


def test_input_and_combined_estimates_are_not_conflated():
    import llm_client

    request = llm_client.Request("assessment", "x" * 12000, max_tokens=900)
    combined = llm_client.estimate_tokens(request)
    inputs = llm_client.estimate_input_tokens(request, QWEN, safety=False)
    # The combined figure includes the reserved output; the input one does not.
    assert combined == 12000 // 6 + 900
    assert inputs == int(12000 / 4.3) + 1
    assert "max_tokens" not in llm_client.estimate_input_tokens.__doc__


def test_a_request_over_the_itpm_ceiling_fails_before_the_provider(monkeypatch,
                                                                   tmp_path):
    import llm_client

    _itpm_env(monkeypatch, tmp_path, GROQ_ITPM_LIMIT__QWEN_QWEN3_8_27B="7000")
    sent = []

    def answer(self, request):
        sent.append(1)
        return "{}"

    transport = _stub_groq(generate=answer, model=QWEN,
                           budget=llm_client.TokenBudget(1_000_000))
    transport.input_budget = llm_client.input_budget_for(QWEN)
    # Comfortably beyond any ceiling: 60k characters of prompt.
    with pytest.raises(llm_client.ProviderError) as raised:
        transport.generate(llm_client.Request("assessment", "x" * 60000, max_tokens=900))
    assert raised.value.category == "request_too_large"
    assert "ITPM ceiling" in str(raised.value)
    assert sent == [], "the request must never reach the provider"


def test_the_safety_factor_never_rejects_a_request_that_would_fit(monkeypatch,
                                                                  tmp_path):
    """An inflated reservation warns; only the honest figure can refuse."""
    import llm_client

    _itpm_env(monkeypatch, tmp_path, GROQ_ITPM_LIMIT__QWEN_QWEN3_8_27B="7000")
    sent = []

    def answer(self, request):
        sent.append(1)
        self.last_input_usage = 5600
        return "{}"

    transport = _stub_groq(generate=answer, model=QWEN,
                           budget=llm_client.TokenBudget(1_000_000))
    transport.input_budget = llm_client.input_budget_for(QWEN)
    # ~27k chars: honest ~6300, adjusted ~7900 (over ITPM), real usage 5600.
    prompt = "x" * 27000
    assert llm_client.estimate_input_tokens(
        llm_client.Request("assessment", prompt), QWEN) > 7000
    transport.generate(llm_client.Request("assessment", prompt, max_tokens=900))
    assert sent == [1], "a request that honestly fits must be sent"


# ---- 7. rolling input pacing --------------------------------------------

def test_two_rapid_assessments_are_paced_by_the_input_window(monkeypatch, tmp_path):
    import llm_client

    _itpm_env(monkeypatch, tmp_path, GROQ_ITPM_LIMIT__QWEN_QWEN3_8_27B="7000")
    state = {"now": 0.0, "slept": []}
    budget = llm_client.input_budget_for(QWEN)
    budget._sleep = lambda s: (state["slept"].append(s),
                               state.__setitem__("now", state["now"] + s))
    budget._clock = lambda: state["now"]

    calls = []

    def answer(self, request):
        calls.append(request.purpose)
        self.last_input_usage = 4800      # two of these exceed 7000/min
        return "{}"

    transport = _stub_groq(generate=answer, model=QWEN,
                           budget=llm_client.TokenBudget(1_000_000))
    transport.input_budget = budget
    request = lambda: llm_client.Request(            # noqa: E731 - test brevity
        "assessment", "x" * 20000, max_tokens=900)

    transport.generate(request())
    assert state["slept"] == [], "the first call has the whole window"
    transport.generate(request())
    assert state["slept"], "the second call must wait for the input window"
    assert sum(state["slept"]) == pytest.approx(60.0)
    assert calls == ["assessment", "assessment"]


def test_actual_prompt_usage_replaces_the_input_reservation(monkeypatch, tmp_path):
    import llm_client

    _itpm_env(monkeypatch, tmp_path, GROQ_ITPM_LIMIT__QWEN_QWEN3_8_27B="7000")
    budget = llm_client.input_budget_for(QWEN)

    def answer(self, request):
        self.last_input_usage = 4200
        self.last_output_usage = 300
        self.last_usage = 4500
        return "{}"

    transport = _stub_groq(generate=answer, model=QWEN,
                           budget=llm_client.TokenBudget(1_000_000))
    transport.input_budget = budget
    transport.generate(llm_client.Request("assessment", "x" * 18000, max_tokens=900))
    assert [t for _, t in budget.used] == [4200], budget.used


def test_qwens_input_budget_cannot_block_gpt_oss(monkeypatch, tmp_path):
    import llm_client

    _itpm_env(monkeypatch, tmp_path, GROQ_ITPM_LIMIT__QWEN_QWEN3_8_27B="7000",
              GROQ_ITPM_LIMIT__OPENAI_GPT_OSS_20B="30000")
    qwen = llm_client.input_budget_for(QWEN)
    research = llm_client.input_budget_for(RESEARCHER)
    assert qwen is not research
    assert research.limit == 30000

    qwen.record(7000)                     # Qwen's whole input minute is spent
    slept: list[float] = []
    research._sleep = lambda s: slept.append(s)
    assert research.wait_for(2000) == 0.0
    assert slept == [], "research paced on Qwen's input window"
    # All three budgets for one model are distinct objects.
    assert len({id(llm_client.budget_for(QWEN)), id(llm_client.output_budget_for(QWEN)),
                id(qwen)}) == 3


def test_an_itpm_refusal_never_rotates_the_api_key(monkeypatch):
    import llm_client

    monkeypatch.setattr(llm_client.time, "sleep", lambda _s: None)
    seen: list[int] = []

    def refuse(self, request):
        seen.append(self._key_index)
        raise llm_client.ProviderError("input_rate_limited",
                                       "Limit 7000, Requested 7256 ITPM")

    transport = _stub_groq(generate=refuse, model=QWEN)
    transport.keys = ("k1", "k2")
    with pytest.raises(llm_client.ProviderError):
        transport.generate(llm_client.Request("assessment", "p", max_tokens=900))
    assert set(seen) == {0}, "an input rate limit must not spend the second key"
    assert "input_rate_limited" not in llm_client.KEY_SPECIFIC_CATEGORIES


def test_itpm_otpm_and_tpm_are_three_separate_classifications():
    import llm_client

    assert llm_client.classify_http(
        413, "Limit 7000, Requested 7256 on input tokens per minute (ITPM)"
    ) == "input_rate_limited"
    assert llm_client.classify_http(
        429, "output tokens per minute (OTPM)") == "output_rate_limited"
    assert llm_client.classify_http(
        429, "tokens per minute (TPM): Limit 8000") == "rate_limited"
    assert llm_client.classify_http(
        429, "tokens per day (TPD)") == "daily_limit_exhausted"
    assert llm_client.classify_http(413, "Request too large") == "request_too_large"
    assert llm_client.classify_exception(
        Exception("ITPM limit reached")) == "input_rate_limited"


def test_a_rolling_itpm_refusal_is_retried(monkeypatch, tmp_path):
    import llm_client

    _itpm_env(monkeypatch, tmp_path, GROQ_ITPM_LIMIT__QWEN_QWEN3_8_27B="7000")
    waits: list[float] = []
    monkeypatch.setattr(llm_client.time, "sleep", lambda s: waits.append(s))
    attempts = []

    def flaky(self, request):
        attempts.append(1)
        if len(attempts) == 1:
            raise llm_client.ProviderError(
                "input_rate_limited", "Rate limit reached. Try again in 4s (ITPM)")
        self.last_input_usage = 4000
        return "{}"

    transport = _stub_groq(generate=flaky, model=QWEN,
                           budget=llm_client.TokenBudget(1_000_000))
    transport.input_budget = llm_client.input_budget_for(QWEN)
    reply = transport.generate(llm_client.Request("assessment", "x" * 18000,
                                                  max_tokens=900))
    assert reply.text == "{}"
    assert len(attempts) == 2
    assert waits == [pytest.approx(4.0)]


def test_the_input_preflight_logs_all_four_numbers(monkeypatch, tmp_path, caplog):
    import logging

    import llm_client

    _itpm_env(monkeypatch, tmp_path, GROQ_ITPM_LIMIT__QWEN_QWEN3_8_27B="7000")
    transport = _stub_groq(generate=lambda self, r: "{}", model=QWEN,
                           budget=llm_client.TokenBudget(1_000_000))
    transport.input_budget = llm_client.input_budget_for(QWEN)
    transport.log = logging.getLogger("itpm-preflight")
    with caplog.at_level(logging.INFO, logger="itpm-preflight"):
        transport.generate(llm_client.Request("assessment", "x" * 18000, max_tokens=900))
    logged = " ".join(r.getMessage() for r in caplog.records)
    for field in ("prompt_estimate=", "schema_estimate=",
                  "adjusted_input_estimate=", "configured_itpm=7000",
                  "safe_ceiling=6300"):
        assert field in logged, field


# ---- 8. the OTPM fix and the architecture are untouched -----------------

def test_the_otpm_and_completion_settings_are_unchanged(run, c3):
    import llm_client

    request = _captured_audit_requests(run, c3)["assessment"]
    assert request.max_tokens == 900
    # Reasoning is off entirely, so the whole 900 belongs to the document.
    assert request.reasoning_effort == "none"
    assert request.reasoning_format is None
    assert request.include_reasoning is None
    assert llm_client.AUDIT_OUTPUT_TOKENS["assessment"] == 900
    assert llm_client.model_otpm_limit(QWEN) == 1000
    # Ids-only buckets survive, now enforced by the schema itself.
    assert "requirement_ids and nothing else" in request.prompt
    buckets = llm_client.APPLICATION_ASSESSMENT_SCHEMA["properties"]
    for bucket in ("strong_matches", "partial_matches", "gaps", "manual_review"):
        assert list(buckets[bucket]["items"]["properties"]) == ["requirement_id"]


def test_the_research_path_is_still_untouched_by_the_input_work(run, c3):
    import llm_client

    request = _captured_audit_requests(run, c3)["company_research"]
    assert request.max_tokens == 1000
    assert request.web_search is True and request.json is False
    assert request.reasoning_effort == "low"
    # Research has no input ceiling configured, so it is never input-paced.
    assert llm_client.input_budget_for(RESEARCHER) is None or \
        llm_client.input_budget_for(RESEARCHER) is not llm_client.input_budget_for(QWEN)


# ================= AO. strict Structured Outputs for the Qwen assessment
#
# The third live run passed ITPM and OTPM and then failed with HTTP 400
# json_validate_failed: best-effort JSON-object mode could not certify its own
# output. Groq supports strict JSON Schema for this model, so the shape is now
# the provider's contract rather than a hope.

def _schema_objects(node, path="root"):
    """Every object node in the schema, with its path."""
    found = []
    if isinstance(node, dict):
        if node.get("type") == "object":
            found.append((path, node))
        for name, child in (node.get("properties") or {}).items():
            found += _schema_objects(child, f"{path}.{name}")
        if node.get("type") == "array":
            found += _schema_objects(node.get("items") or {}, f"{path}[]")
    return found


# ---- 1. the request uses strict structured output -----------------------

def test_the_assessment_request_uses_a_strict_json_schema(run, c3):
    import llm_client

    request = _captured_audit_requests(run, c3)["assessment"]
    assert request.response_schema is llm_client.ASSESSMENT_RESPONSE_FORMAT
    assert request.response_schema["type"] == "json_schema"
    block = request.response_schema["json_schema"]
    assert block["strict"] is True
    assert block["name"] == "application_assessment"
    assert block["schema"] is llm_client.APPLICATION_ASSESSMENT_SCHEMA


def test_the_sdk_sends_the_schema_instead_of_json_object_mode():
    source = Path(_llm_client().__file__).read_text(encoding="utf-8")
    generate = source.split("def _generate(self, request: Request) -> str:")[1]
    generate = generate.split("\n\n\n")[0]
    assert 'settings["response_format"] = request.response_schema' in generate
    # json_object mode is the ELSE branch, so the two never go together.
    assert 'elif request.json:' in generate
    assert generate.index("request.response_schema") < generate.index("json_object")


def test_only_the_assessment_uses_a_response_schema(run, c3):
    requests = _captured_audit_requests(run, c3)
    assert requests["assessment"].response_schema is not None
    assert requests["company_research"].response_schema is None
    # No generation request may carry one either.
    source = Path(_llm_client().__file__).read_text(encoding="utf-8")
    for purpose in ("project_selection", "project_bullets", "bullet_repair",
                    "cover_letter", "company_research"):
        head = source.split(f'"{purpose}", ')[1].split("))")[0]
        assert "response_schema" not in head, purpose


# ---- 2. the schema is strict-mode legal and matches the contract --------

def test_every_schema_object_forbids_additional_properties():
    import llm_client

    objects = _schema_objects(llm_client.APPLICATION_ASSESSMENT_SCHEMA)
    assert len(objects) >= 8, "the schema should have several nested objects"
    for path, node in objects:
        assert node.get("additionalProperties") is False, path


def test_every_schema_property_is_required():
    import llm_client

    for path, node in _schema_objects(llm_client.APPLICATION_ASSESSMENT_SCHEMA):
        properties = set(node.get("properties") or {})
        assert set(node.get("required") or []) == properties, path
        assert properties, path


def test_the_schema_carries_every_current_assessment_dimension():
    import llm_client

    properties = llm_client.APPLICATION_ASSESSMENT_SCHEMA["properties"]
    for field in ("fit_score", "recommendation", "summary", "eligibility",
                  "strong_matches", "partial_matches", "gaps", "manual_review",
                  "complementary_strengths", "tailoring_quality",
                  "experience_selection", "project_selection", "project_bullets",
                  "callback_likelihood", "risk_flags"):
        assert field in properties, field
    assert len(properties) == 15, sorted(properties)


def test_the_schema_keeps_every_score_component():
    import llm_client

    properties = llm_client.APPLICATION_ASSESSMENT_SCHEMA["properties"]
    selection = properties["project_selection"]["properties"]["components"]["properties"]
    assert set(selection) == set(grounding.PROJECT_SELECTION_COMPONENTS)
    bullets = properties["project_bullets"]["properties"]["components"]["properties"]
    assert set(bullets) == set(grounding.PROJECT_BULLET_COMPONENTS)
    for block in ("tailoring_quality", "experience_selection", "project_selection",
                  "project_bullets"):
        assert properties[block]["properties"]["score"]["type"] == "number", block


def test_the_schema_keeps_the_verdict_buckets_ids_only():
    import llm_client

    properties = llm_client.APPLICATION_ASSESSMENT_SCHEMA["properties"]
    for bucket in ("strong_matches", "partial_matches", "gaps", "manual_review"):
        node = properties[bucket]
        assert node["type"] == "array", bucket
        item = node["items"]
        assert list(item["properties"]) == ["requirement_id"], bucket
        assert item["properties"]["requirement_id"]["type"] == "string", bucket
        # No evidence prose may creep back in.
        for prose in ("evidence", "limitation", "detail", "reason", "source",
                      "importance", "status"):
            assert prose not in item["properties"], f"{bucket}.{prose}"


def test_the_schema_uses_the_existing_enum_contracts():
    import llm_client

    properties = llm_client.APPLICATION_ASSESSMENT_SCHEMA["properties"]
    assert properties["recommendation"]["enum"] == list(grounding.RECOMMENDATIONS)
    assert properties["callback_likelihood"]["enum"] == list(grounding.CALLBACK_LIKELIHOOD)
    assert properties["eligibility"]["properties"]["status"]["enum"] == \
        list(grounding.ELIGIBILITY_STATUS)


def test_a_schema_shaped_reply_passes_python_validation_unchanged():
    """Whatever the schema certifies must still satisfy our own validators."""
    import llm_client

    ids = {f"REQ-{i:03d}" for i in range(1, 6)}
    reply = {
        "fit_score": 7.4, "recommendation": "apply",
        "summary": "Most scored requirements are evidenced.",
        "eligibility": {"status": "uncertain", "details": []},
        "strong_matches": [{"requirement_id": "REQ-001"}],
        "partial_matches": [{"requirement_id": "REQ-002"}],
        "gaps": [{"requirement_id": "REQ-003"}],
        "manual_review": [{"requirement_id": "REQ-004"}],
        "complementary_strengths": [], "risk_flags": [],
        "tailoring_quality": {"score": 9.0, "notes": []},
        "experience_selection": {"score": 8.5, "notes": []},
        "project_selection": {"score": 8.5, "notes": [], "components": {
            name: 8.0 for name in grounding.PROJECT_SELECTION_COMPONENTS}},
        "project_bullets": {"score": 8.6, "notes": [], "components": {
            name: 9.0 for name in grounding.PROJECT_BULLET_COMPONENTS}},
        "callback_likelihood": "MEDIUM",
    }
    # Shape matches the schema's own required/properties sets exactly.
    assert set(reply) == set(llm_client.APPLICATION_ASSESSMENT_SCHEMA["required"])
    assert grounding.errors(grounding.validate_assessment(reply, "A posting.", ids)) == []
    audit, problems = grounding.validate_application_audit(reply)
    assert problems == []
    for key in ("experience_selection_score", "resume_tailoring_score",
                "project_selection_score", "project_bullet_score"):
        assert audit[key] is not None, key
    assert audit["callback_likelihood"] == "MEDIUM"


# ---- 6. the schema is counted as input ----------------------------------

def test_the_input_estimate_includes_the_schema(run, c3):
    import llm_client

    request = _captured_audit_requests(run, c3)["assessment"]
    with_schema = llm_client.estimate_input_tokens(request, QWEN, safety=False)
    bare = llm_client.Request("assessment", request.prompt, max_tokens=900)
    without = llm_client.estimate_input_tokens(bare, QWEN, safety=False)
    assert with_schema > without, "the schema must not be counted as free"
    # The difference is the serialized schema, at the calibrated rate.
    serialized = len(json.dumps(request.response_schema, separators=(",", ":")))
    assert with_schema - without == pytest.approx(serialized / 4.3, abs=2)
    assert "response_format" in llm_client.request_input_text.__doc__


def test_c3_stays_under_the_hard_itpm_ceiling_with_the_schema(run, c3, monkeypatch,
                                                              tmp_path):
    import llm_client

    _itpm_env(monkeypatch, tmp_path, GROQ_ITPM_LIMIT__QWEN_QWEN3_8_27B="7000")
    request = _captured_audit_requests(run, c3)["assessment"]
    honest = llm_client.estimate_input_tokens(request, QWEN, safety=False)
    adjusted = llm_client.estimate_input_tokens(request, QWEN)
    assert honest < 7000, honest
    assert adjusted < 7000, adjusted
    # The live refusal was at 7256 provider-counted input tokens.
    assert honest < 7256


def test_the_calibration_and_limits_are_unchanged():
    import llm_client

    assert llm_client.input_chars_per_token(QWEN) == 4.3
    assert llm_client.input_safety_factor(QWEN) == 1.25
    assert llm_client.model_itpm_limit(QWEN) == 7000
    assert llm_client.safe_input_ceiling(QWEN) == 6300
    assert llm_client.model_otpm_limit(QWEN) == 1000
    assert llm_client.AUDIT_OUTPUT_TOKENS["assessment"] == 900


# ---- 8. nothing else moved ----------------------------------------------

def test_the_research_response_path_is_unchanged(run, c3):
    import llm_client

    request = _captured_audit_requests(run, c3)["company_research"]
    assert request.response_schema is None, "research keeps textual JSON"
    assert request.json is False
    assert request.web_search is True
    assert request.max_tokens == 1000
    assert request.reasoning_effort == "low"
    assert request.include_reasoning is False
    assert "Return ONLY valid JSON" in request.prompt, "its parser needs the ask"
    # And it still goes through parse_json plus the deterministic validator.
    source = Path(llm_client.__file__).read_text(encoding="utf-8")
    research = source.split("def research_company(")[1].split("\n    def ")[0]
    assert 'parse_json(reply.text, "company_research")' in research
    assert "grounding.validate_company_research(" in research


def test_the_architecture_is_untouched_by_the_schema_work():
    source = Path(run_pipeline.__file__).read_text(encoding="utf-8")
    block = source.split("def run_audits(")[1].split("\ndef ")[0]
    assert "ThreadPoolExecutor(max_workers=2)" in block
    assert block.count("pool.submit(") == 2
    code = _code_only(block)
    for forbidden in ("write_text", "write_json_atomic", "json.dump"):
        assert forbidden not in code
    assert source.count("write_json_atomic(") == 3
    # Neither audit may reach Gemini.
    client_source = Path(_llm_client().__file__).read_text(encoding="utf-8")
    builder = client_source.split("def build_client(")[1]
    assert builder.count("fallback=None") == 2


# ============ AP. truncation is not a schema rejection
#
# The fourth live run: the strict schema was ACCEPTED, reasoning_format was
# ACCEPTED, both preflights passed, and generation then ran out of completion
# budget mid-document. Groq reports that as a validation failure, which the
# previous classifier read as schema_rejected - sending the reader after the
# wrong bug entirely.

@pytest.mark.parametrize("body", [
    "max completion tokens reached before generating a valid document",
    "the output was truncated to fit max_completion_tokens",
    "json_validate_failed: missing required content, increase max_completion_tokens",
    "Generation stopped: finish_reason=length",
])
def test_a_truncated_document_classifies_as_output_truncated(body):
    import llm_client

    assert llm_client.classify_http(400, body) == "output_truncated", body
    assert llm_client.classify_exception(Exception(body)) == "output_truncated", body


@pytest.mark.parametrize("body", [
    "invalid json_schema: unsupported keyword 'patternProperties'",
    "json_schema configuration error: strict mode requires additionalProperties false",
    "Failed to validate JSON. Please adjust your prompt.",
    "response_format.json_schema is invalid",
])
def test_a_real_schema_problem_still_classifies_as_schema_rejected(body):
    import llm_client

    assert llm_client.classify_http(400, body) == "schema_rejected", body


def test_truncation_outranks_the_schema_check():
    """A message carrying BOTH signals is a truncation, not a schema bug."""
    import llm_client

    both = ("json_validate_failed: max completion tokens reached before "
            "generating a valid document")
    assert llm_client.classify_http(400, both) == "output_truncated"
    # Order matters in the classifier, so assert the precedence explicitly.
    source = Path(llm_client.__file__).read_text(encoding="utf-8")
    block = _code_only(
        source.split("if status == 400:")[1].split("if status >= 500:")[0])
    assert block.index("_TRUNCATED_OUTPUT") < block.index("json_validate_failed")


@pytest.mark.parametrize("category", ["output_truncated", "schema_rejected"])
def test_both_categories_are_non_retryable_and_never_rotate_keys(category, monkeypatch):
    import llm_client

    monkeypatch.setattr(llm_client.time, "sleep", lambda _s: None)
    assert category in llm_client.NON_RETRYABLE_CATEGORIES
    assert category not in llm_client.KEY_SPECIFIC_CATEGORIES

    seen: list[int] = []

    def fail(self, request):
        seen.append(self._key_index)
        raise llm_client.ProviderError(category, f"simulated {category}")

    transport = _stub_groq(generate=fail, model=QWEN)
    transport.keys = ("k1", "k2")
    with pytest.raises(llm_client.ProviderError) as raised:
        transport.generate(llm_client.Request("assessment", "p", max_tokens=900))
    assert seen == [0], "identical bytes must not be resent, on any key"
    assert raised.value.category == category
    assert transport._key_index == 0


def test_a_truncated_completion_keeps_its_output_reservation(monkeypatch, tmp_path):
    """Truncated tokens were really generated, so the window must hold them."""
    import llm_client

    _otpm_env(monkeypatch, tmp_path, GROQ_OTPM_LIMIT__QWEN_QWEN3_8_27B="1000")
    budget = llm_client.output_budget_for(QWEN)

    def truncate(self, request):
        self.last_output_usage = 900      # the whole cap was spent
        raise llm_client.ProviderError("output_truncated",
                                       "max completion tokens reached")

    transport = _stub_groq(generate=truncate, model=QWEN,
                           budget=llm_client.TokenBudget(1_000_000))
    transport.output_budget = budget
    with pytest.raises(llm_client.ProviderError):
        transport.generate(llm_client.Request("assessment", "p" * 3000, max_tokens=900))
    assert [t for _, t in budget.used] == [900], budget.used


# ---- the rest of the contract is untouched ------------------------------

def test_the_completion_cap_and_otpm_are_unchanged():
    import llm_client

    assert llm_client.AUDIT_OUTPUT_TOKENS["assessment"] == 900
    assert llm_client.AUDIT_OUTPUT_TOKENS["company_research"] == 1000
    assert llm_client.model_otpm_limit(QWEN) == 1000
    assert llm_client.model_itpm_limit(QWEN) == 7000
    assert llm_client.safe_input_ceiling(QWEN) == 6300


def test_strict_structured_output_is_still_enabled(run, c3):
    import llm_client

    request = _captured_audit_requests(run, c3)["assessment"]
    assert request.response_schema["type"] == "json_schema"
    assert request.response_schema["json_schema"]["strict"] is True
    assert request.response_schema["json_schema"]["schema"] is \
        llm_client.APPLICATION_ASSESSMENT_SCHEMA
    assert len(llm_client.APPLICATION_ASSESSMENT_SCHEMA["properties"]) == 15
    # No retreat to json_object mode and no loosening.
    source = Path(llm_client.__file__).read_text(encoding="utf-8")
    assert '"strict": True' in source
    assert "strict=False" not in source and '"strict": False' not in source


def test_the_research_reasoning_configuration_is_unchanged(run, c3):
    request = _captured_audit_requests(run, c3)["company_research"]
    assert request.reasoning_effort == "low"
    assert request.include_reasoning is False
    assert request.reasoning_format is None
    assert request.response_schema is None
    assert request.json is False
    assert request.web_search is True
    assert request.max_tokens == 1000
