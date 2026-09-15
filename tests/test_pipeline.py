"""The final suite: a few meaningful checks against the real artifact.

Every test here runs the production CLI once (mocked providers, zero API calls)
and then inspects the COMPILED PDF, because the PDF is what the contracts are
written about. There is no test-only renderer and no simplified pipeline.
"""
from __future__ import annotations

import dataclasses
import json
import re
import subprocess
import sys
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
    pdf_path = run_dir / "resume.pdf"
    assert pdf_path.exists(), f"no PDF was produced\n{completed.stdout[-3000:]}"

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
    return json.loads((run["dir"] / "assessment.json").read_text())


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
    # the retry prompt carried the exact validator message
    assert "failed deterministic validation" in transport.requests[1].prompt
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

def test_assessment_retries_malformed_json_then_accepts(run, bae):
    """Unparseable output is a rejected structural attempt, not a hard failure."""
    transport, client = _sequence_client(
        run, ["I'm sorry, I cannot produce JSON for this request.",
              _assessment_payload(bae, broken=False)])
    data = _assess_call(client, run, bae)

    calls = [c for c in client.calls if c.purpose == "assessment"]
    assert len(calls) == 2
    assert calls[0].transport_ok is True
    assert calls[0].accepted is False
    assert "malformed JSON" in calls[0].detail
    assert calls[1].accepted is True
    assert "not JSON" in transport.requests[1].prompt

    # Python's verdicts remain authoritative after a parse-level retry
    assert data["verdict_source"] == "python_deterministic"
    actual = {e["requirement_id"]: b for b in
              ("strong_matches", "partial_matches", "gaps", "manual_review")
              for e in data[b]}
    assert actual == engine.verdict_index(_assess_verdicts(bae))


def test_three_malformed_json_replies_raise_provider_error(run, bae):
    import llm_client

    junk = "not json at all"
    transport, client = _sequence_client(run, [junk, junk, junk])
    with pytest.raises(llm_client.ProviderError):
        _assess_call(client, run, bae)
    calls = [c for c in client.calls if c.purpose == "assessment"]
    assert len(calls) == 3
    assert all(c.transport_ok is True for c in calls)
    assert all(c.accepted is False for c in calls)


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
