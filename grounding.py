"""Factual grounding: metric semantics, project-bullet grounding, letter checks.

Every generated sentence has to be traceable to evidence that already exists in
`Master_Resume_Context.md`. This module is deliberately literal about numbers:
a metric's identity is its value, its unit, its threshold structure and whether
it is approximate. There is no numeric tolerance anywhere - `3+` is not `3`,
`approximately 90%` is not `90%`, and `sub-500ms` is not `500ms`.
"""
from __future__ import annotations

import copy
import dataclasses
import re
import unicodedata
from dataclasses import dataclass, field

from resume_engine import (MasterFacts, Project, _mentions, _skill_needles,
                           expected_graduation, fold_term, role_bullet_prefixes)

# ============================================================== typography

APPROX_MARK = "\x01"

_GLYPHS = {
    "\u223c": APPROX_MARK,   # ∼  (LaTeX $\sim$ renders to this)
    "\u2248": APPROX_MARK,   # ≈
    "~": APPROX_MARK,
    "\u2212": "-",           # Unicode minus
    "\u2011": "-",           # non-breaking hyphen
    "\u00a0": " ", "\u202f": " ", "\u2009": " ", "\u200a": " ",
    "\u2265": ">=", "\u2264": "<=",
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
}

EM_DASH = "\u2014"


def normalize_typography(text: str) -> str:
    """Canonicalize glyphs BEFORE any numeric parsing.

    Without this, `$\\sim$90\\%` extracts as a bare exact `90%` and
    `sub-500 ms` (non-breaking hyphen) extracts as an exact `500 ms`, so an
    invented exact claim would pass validation.
    """
    out = text
    for glyph, plain in _GLYPHS.items():
        out = out.replace(glyph, plain)
    # An en/em dash between digits is a numeric range separator.
    out = re.sub(r"(?<=\d)\s*[\u2013\u2014]\s*(?=\d)", "-", out)
    out = out.replace("\u2013", " - ").replace(EM_DASH, " - ")
    out = re.sub(r"\\(?:sim|approx)\b", APPROX_MARK, out)
    out = re.sub(r"\\(?:geq|ge)\b", ">=", out)
    out = re.sub(r"\\(?:leq|le)\b", "<=", out)
    return re.sub(r"[ \t]+", " ", out)


# ================================================================= metrics

_DIMENSIONAL = {"%", "s", "ms", "min", "h", "x", "gb", "mb", "kb"}
_UNIT_CANON = {
    "%": "%", "percent": "%", "pct": "%", "percentage": "%",
    "s": "s", "sec": "s", "secs": "s", "second": "s", "seconds": "s",
    "ms": "ms", "millisecond": "ms", "milliseconds": "ms",
    "min": "min", "mins": "min", "minute": "min", "minutes": "min",
    "h": "h", "hr": "h", "hrs": "h", "hour": "h", "hours": "h",
    "x": "x", "gb": "gb", "mb": "mb", "kb": "kb",
}
_SCALES = {"k": 1_000, "m": 1_000_000}
_KNOWN_GLUED = set(_UNIT_CANON) | set(_SCALES)

_SPELLED = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "hundred": 100,
}

_BEFORE_STRUCTURE = (
    ("at_least", (r"at\s+least", r"no\s+fewer\s+than", r"no\s+less\s+than",
                  r"minimum\s+of", r"a\s+minimum\s+of", r">=")),
    ("at_most", (r"at\s+most", r"up\s+to", r"no\s+more\s+than", r"maximum\s+of",
                 r"within", r"<=")),
    ("greater_than", (r"more\s+than", r"greater\s+than", r"over", r"above",
                      r"exceed(?:s|ing|ed)?", r"north\s+of", r">")),
    ("less_than", (r"less\s+than", r"fewer\s+than", r"under", r"below",
                   r"sub[-\s]?", r"faster\s+than", r"<")),
)
_AFTER_STRUCTURE = (
    ("at_least", (r"or\s+more", r"or\s+greater", r"or\s+higher", r"or\s+above",
                  r"and\s+above", r"and\s+up", r"plus")),
    ("at_most", (r"or\s+less", r"or\s+fewer", r"or\s+lower", r"or\s+below")),
)
_APPROX_WORDS = (r"approximately", r"approx\.?", r"about", r"around", r"roughly",
                 r"nearly", r"almost", r"circa", r"close\s+to", r"just\s+(?:under|over)",
                 re.escape(APPROX_MARK))

_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_METRIC_RE = re.compile(
    rf"(?P<num>{_NUM})\s*(?P<scale>[kKmM](?![A-Za-z]))?\s*(?P<plus>\+)?\s*"
    rf"(?P<unit>%|[A-Za-z][A-Za-z/\-]{{0,14}})?")
_RANGE_RE = re.compile(rf"(?P<low>{_NUM})\s*-\s*(?P<high>{_NUM})\s*(?P<unit>%|[A-Za-z]+)?")

# Identifier-ish tokens whose digits are names, not measurements.
_IDENT_ALNUM = re.compile(r"\b[A-Za-z]{1,6}\d+[A-Za-z0-9]*(?:-\d+)?\b")  # p95, x86-64, card1, IPv4
_VERSIONED = re.compile(r"\b[A-Za-z][A-Za-z0-9]*(?:[-:/.][A-Za-z0-9]+)+\b")  # Qwen-35B, gpt-oss-120b
_MIXED_SEGMENT = re.compile(r"\d+[A-Za-z]|[A-Za-z]+\d")
_GLUED_SUFFIX = re.compile(r"\b(\d+(?:\.\d+)?)([A-Za-z]{1,4})\b")


@dataclass(frozen=True)
class Metric:
    raw: str
    value: float
    unit: str
    structure: str                 # exact|at_least|greater_than|at_most|less_than|range
    approximate: bool
    high: float | None = None
    spelled: bool = False
    # Offset inside the masked text, used only to decide who a claim belongs to.
    start: int = field(default=-1, compare=False)

    @property
    def identity(self) -> tuple:
        return (round(self.value, 6), None if self.high is None else round(self.high, 6),
                self.unit, self.structure, self.approximate)

    def describe(self) -> str:
        shape = {"exact": "exactly", "at_least": "at least", "greater_than": "more than",
                 "at_most": "at most", "less_than": "less than", "range": "range"}[self.structure]
        value = f"{self.value:g}" + (f"-{self.high:g}" if self.high is not None else "")
        if self.approximate:
            shape = "approximately" if self.structure == "exact" else f"approximately {shape}"
        return f"{shape} {value}{(' ' + self.unit) if self.unit else ''}"


def _mask_identifiers(text: str, extra: tuple[str, ...] = ()) -> str:
    masked = text
    for term in extra:
        if term:
            masked = re.sub(re.escape(term), " ", masked, flags=re.IGNORECASE)
    masked = _IDENT_ALNUM.sub(" ", masked)
    masked = _VERSIONED.sub(_mask_versioned, masked)
    # `35B`, `8b`, `120b` are model sizes; `10k`, `500ms`, `2s` are measurements.
    masked = _GLUED_SUFFIX.sub(
        lambda m: m.group(0) if m.group(2).lower() in _KNOWN_GLUED else " ", masked)
    return masked


def _mask_versioned(match: re.Match) -> str:
    """Mask `Qwen-35B` / `gpt-oss-120b`, but keep `sub-500ms` and `under-200ms`."""
    segment = re.split(r"[-:/.]", match.group(0), 1)[-1]
    if not _MIXED_SEGMENT.search(segment):
        return match.group(0)
    measurement = re.fullmatch(r"(\d+(?:\.\d+)?)([A-Za-z]{1,4})", segment)
    if measurement and measurement.group(2).lower() in _KNOWN_GLUED:
        return match.group(0)
    return " "


def _structure_before(context: str) -> tuple[str, bool]:
    tail = context[-34:].lower()
    approximate = any(re.search(w + r"[\s\-]*$", tail) for w in _APPROX_WORDS)
    for structure, patterns in _BEFORE_STRUCTURE:
        for pattern in patterns:
            if re.search(pattern + r"[\s\-]*(?:" + "|".join(_APPROX_WORDS) + r")?[\s\-]*$", tail):
                return structure, approximate
    return "exact", approximate


def _structure_after(context: str) -> str | None:
    head = context[:18].lower()
    for structure, patterns in _AFTER_STRUCTURE:
        if any(re.match(r"\s*" + pattern + r"\b", head) for pattern in patterns):
            return structure
    return None


def extract_metrics(text: str, masked_terms: tuple[str, ...] = ()) -> list[Metric]:
    """Pull every measurable claim out of a piece of text."""
    source = _mask_identifiers(normalize_typography(text), masked_terms)
    metrics: list[Metric] = []
    consumed: set[int] = set()

    for match in _RANGE_RE.finditer(source):
        unit = _canon_unit(match.group("unit"))
        structure, approximate = _structure_before(source[: match.start()])
        metrics.append(Metric(match.group(0).strip(), _number(match.group("low")), unit,
                              "range", approximate, high=_number(match.group("high")),
                              start=match.start()))
        consumed.update(range(match.start(), match.end()))

    for match in _METRIC_RE.finditer(source):
        if match.start("num") in consumed:
            continue
        before = source[: match.start()]
        structure, approximate = _structure_before(before)
        if (structure == "exact" and before.rstrip().endswith(("-", "/"))
                and re.search(r"[A-Za-z]\s*[-/]\s*$", before)):
            continue        # part of a hyphenated name, not `sub-500 ms`
        value = _number(match.group("num")) * _SCALES.get((match.group("scale") or "").lower(), 1)
        unit_token = match.group("unit")
        unit = _canon_unit(unit_token)
        if unit_token and not unit:
            unit = _count_noun(unit_token)
        if match.group("plus"):
            structure = "at_least"
        after = _structure_after(source[match.end():])
        if after:
            structure = after
        metrics.append(Metric(match.group(0).strip(), value, unit, structure, approximate,
                              start=match.start()))

    for match in re.finditer(r"(?<![A-Za-z])(" + "|".join(_SPELLED) + r")(?:\s+([A-Za-z\-]+))?",
                             source, re.IGNORECASE):
        word, noun = match.group(1).lower(), (match.group(2) or "")
        if noun.lower() in _PRONOUN_CONTEXT or (not noun and word in ("one", "two")):
            continue
        structure, approximate = _structure_before(source[: match.start()])
        after = _structure_after(source[match.end(1):])
        metrics.append(Metric(match.group(0).strip(), float(_SPELLED[word]),
                              _canon_unit(noun) or _count_noun(noun), after or structure,
                              approximate, spelled=True, start=match.start()))
    return metrics


def _number(raw: str) -> float:
    return float(raw.replace(",", ""))


def _canon_unit(token: str | None) -> str:
    if not token:
        return ""
    return _UNIT_CANON.get(token.strip().lower().rstrip("."), "")


def _count_noun(token: str) -> str:
    noun = token.strip().lower().rstrip(".")
    if noun in _STOP_UNITS:
        return ""
    if noun.endswith("s") and not noun.endswith("ss") and len(noun) > 3:
        noun = noun[:-1]
    return noun


_PRONOUN_CONTEXT = {
    "of", "the", "a", "an", "is", "was", "were", "are", "that", "which", "who", "in",
    "on", "to", "for", "with", "and", "or", "but", "such", "thing", "another",
}

_STOP_UNITS = {
    "and", "or", "of", "to", "in", "on", "for", "with", "the", "a", "an", "by", "from",
    "at", "as", "that", "which", "while", "over", "under", "across", "into", "per",
    "iterations", "iteration", "times", "time", "more", "less", "than", "up", "out",
}


def _dimensional(unit: str) -> bool:
    return unit in _DIMENSIONAL


def metric_supported(claim: Metric, evidence: list[Metric]) -> Metric | None:
    """A claim is supported only by an evidence metric of identical meaning."""
    for candidate in evidence:
        if (claim.structure != candidate.structure
                or claim.approximate != candidate.approximate
                or round(claim.value, 6) != round(candidate.value, 6)
                or (claim.high is None) != (candidate.high is None)
                or (claim.high is not None
                    and round(claim.high, 6) != round(candidate.high, 6))):
            continue
        if _dimensional(claim.unit) or _dimensional(candidate.unit):
            if claim.unit == candidate.unit:
                return candidate
            continue                    # a bare number never satisfies "2 s"
        return candidate                # count nouns differ harmlessly
    return None


def nearest_evidence(claim: Metric, evidence: list[Metric]) -> Metric | None:
    same_value = [e for e in evidence if round(e.value, 6) == round(claim.value, 6)]
    if same_value:
        return same_value[0]
    return min(evidence, key=lambda e: abs(e.value - claim.value), default=None)


# ====================================================== metric attribution

# A cover letter mixes two kinds of fact with two different authorities:
# candidate claims answer to the master, employer claims answer to the posting.
# The posting's own numbers are therefore usable, but ONLY while they stay
# attributed to the employer: a JD number must never migrate into candidate
# ownership. Attribution is decided by whichever marker sits NEAREST the metric
# inside its own sentence, and an unattributed number fails.

_EMPLOYER_MARKERS = (
    "the company", "the posting", "the role", "the position", "the team",
    "the product", "the platform", "the organization", "the employer", "the job",
    "this role", "this position", "this posting", "this team", "this company",
    "your team", "your company", "your product", "your platform", "your mission",
    "your organization", "your customers", "your users", "their mission",
    "its mission", "the mission",
)

_CANDIDATE_MARKERS = (
    "i built", "i build", "i developed", "i designed", "i engineered", "i delivered",
    "i scaled", "i implemented", "i integrated", "i architected", "i led", "i created",
    "i shipped", "i automated", "i processed", "i reduced", "i improved", "i migrated",
    "i trained", "i deployed", "i wrote", "i maintained", "i handled", "i optimized",
    "i parallelized", "i benchmarked", "i added", "i tested", "i supported", "i served",
    "i serve", "i cut", "i raised",
    # Bare possessive: "my React expertise and systematic performance profiling".
    # Nearest-marker-wins still lets an employer reference closer to the claim
    # take precedence, so "my interest in Superhuman's platform" is unaffected.
    "my",
    "my pipeline", "my system", "my systems", "my module", "my modules", "my service",
    "my services", "my platform", "my implementation", "my code", "my project",
    "my projects", "my work", "my experience",
    "we built", "we developed", "we delivered", "we designed", "we scaled", "we shipped",
    "we implemented", "we served",
)

_SENTENCE_BREAK = re.compile(r"[.!?](?:\s|$)")


def _sentence_prefix(text: str, start: int) -> str:
    """Everything from the start of the metric's own sentence up to the metric."""
    if start < 0:
        return ""
    begin = 0
    for match in _SENTENCE_BREAK.finditer(text, 0, start):
        begin = match.end()
    return text[begin:start]


def _last_marker(prefix: str, markers: tuple[str, ...]) -> int:
    """Offset of the marker closest to the end of `prefix`, or -1."""
    last = -1
    for marker in markers:
        for match in re.finditer(r"(?<![a-z])" + re.escape(marker) + r"(?![a-z])", prefix):
            last = max(last, match.start())
    return last


def _attribution(prefix: str, company: str | None) -> str:
    """Who owns the claim sitting at the end of `prefix`? Nearest marker wins."""
    lowered = prefix.lower()
    employer_markers = _EMPLOYER_MARKERS
    if company:
        head = company.split("(")[0].strip().lower()
        if head:
            employer_markers = employer_markers + (head,)
    employer = _last_marker(lowered, employer_markers)
    candidate = _last_marker(lowered, _CANDIDATE_MARKERS)
    if employer < 0 and candidate < 0:
        return "ambiguous"
    return "employer" if employer > candidate else "candidate"


@dataclass(frozen=True)
class MetricSource:
    """One letter metric resolved against its authoritative source."""

    metric: Metric
    source: str                  # "master" | "jd" | "none"
    attribution: str             # "candidate" | "employer" | "ambiguous"
    support: Metric | None = None

    @property
    def ok(self) -> bool:
        return self.source in ("master", "jd")

    @property
    def display(self) -> str:
        """The claim as a human reads it (the approximation mark restored)."""
        return self.metric.raw.replace(APPROX_MARK, "~")


def classify_letter_metrics(text: str, master: MasterFacts, *, jd_text: str | None = None,
                            company: str | None = None,
                            masked_terms: tuple[str, ...] = ()) -> list[MetricSource]:
    """Resolve every metric in a cover letter to the source that authorizes it.

    Candidate metrics must trace to the master. A metric the master does not
    carry is allowed only when the posting states it AND the sentence keeps it
    attributed to the employer.
    """
    masked = _mask_identifiers(normalize_typography(text), masked_terms)
    candidate_evidence = extract_metrics(master.raw, masked_terms)
    employer_evidence = extract_metrics(jd_text, masked_terms) if jd_text else []

    resolved: list[MetricSource] = []
    for claim in extract_metrics(text, masked_terms):
        support = metric_supported(claim, candidate_evidence)
        if support is not None:
            resolved.append(MetricSource(claim, "master", "candidate", support))
            continue
        attribution = _attribution(_sentence_prefix(masked, claim.start), company)
        employer_support = metric_supported(claim, employer_evidence)
        if attribution == "employer" and employer_support is not None:
            resolved.append(MetricSource(claim, "jd", "employer", employer_support))
        else:
            resolved.append(MetricSource(claim, "none", attribution, None))
    return resolved


# ================================================================ problems


@dataclass
class Problem:
    kind: str
    severity: str            # "error" blocks; "warning" is logged only
    message: str

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.kind}: {self.message}"

    @property
    def signature(self) -> tuple[str, str]:
        """Semantic identity of the defect, independent of which check found it.

        Two checks legitimately look for the same em dash, and a word can sit on
        both the built-in discouraged list and the spreadsheet's banned list.
        That is one defect, so it collapses to one signature.
        """
        message = " ".join(self.message.lower().split())
        if "em dash" in message:
            return ("style", "em_dash")
        word = re.search(r"discouraged wording '([^']+)'", message)
        if word:
            return ("style", f"wording:{word.group(1)}")
        banned = re.search(r"banned phrase '([^']+)'", message)
        if banned:
            return ("style", f"wording:{banned.group(1)}")
        return (self.kind, message)


def errors(problems: list[Problem]) -> list[Problem]:
    return [p for p in problems if p.severity == "error"]


def dedupe_problems(problems: list[Problem]) -> list[Problem]:
    """One entry per semantic defect, in first-seen order.

    Genuinely distinct problems are never merged: only an identical signature
    collapses, and an error outranks a warning that reports the same thing.
    """
    order: list[Problem] = []
    index: dict[tuple[str, str], int] = {}
    for problem in problems:
        key = problem.signature
        if key not in index:
            index[key] = len(order)
            order.append(problem)
        elif problem.severity == "error" and order[index[key]].severity != "error":
            order[index[key]] = problem
    return order


# =================================================== style / wording checks

PREFERRED_VERBS = ("Built", "Developed", "Designed", "Engineered", "Implemented",
                   "Integrated", "Architected", "Parallelized", "Delivered", "Optimized",
                   "Automated", "Scaled", "Reduced", "Migrated", "Trained", "Deployed")

DISCOURAGED_WORDS = ("leveraged", "leveraging", "spearheaded", "cutting-edge", "seamless",
                     "seamlessly", "innovative", "transformative", "world-class",
                     "results-driven", "synergy", "best-in-class", "state-of-the-art",
                     "utilize", "utilized", "utilizing")


def check_style(text: str, banned: tuple[str, ...] = ()) -> list[Problem]:
    problems: list[Problem] = []
    if EM_DASH in text:
        problems.append(Problem("style", "error", "contains an em dash"))
    lowered = text.lower()
    for word in DISCOURAGED_WORDS + tuple(b.lower() for b in banned if len(b) > 3):
        if re.search(r"(?<![a-z])" + re.escape(word) + r"(?![a-z])", lowered):
            problems.append(Problem("style", "error", f"uses discouraged wording {word!r}"))
    for hack in (r"\newline", r"\linebreak", r"\\[", "\u00a0", r"\hspace*", r"\phantom"):
        if hack in text:
            problems.append(Problem("layout_hack", "error",
                                    f"contains a layout hack {hack!r}"))
    return problems


# ============================================== project bullet grounding


def _vocabulary(master: MasterFacts) -> dict[str, str]:
    """Distinctive technology / proper-noun vocabulary, folded -> display form."""
    vocab: dict[str, str] = {}
    for project in master.projects:
        for tech in project.tech:
            vocab[fold_term(tech)] = tech
        vocab[fold_term(project.name)] = project.name
    for skill in master.skills:
        vocab[fold_term(skill.name)] = skill.name
        bare = re.sub(r"\s*\([^)]*\)", "", skill.name).strip()
        if bare:
            vocab.setdefault(fold_term(bare), bare)
    for noun in master.proper_nouns:
        vocab.setdefault(fold_term(noun), noun)
    # "WebSocket" must resolve to the "WebSockets" skill, and vice versa.
    for key, display in list(vocab.items()):
        if key.endswith("s") and not key.endswith("ss") and len(key) > 4:
            vocab.setdefault(key[:-1], display)
        elif len(key) > 3:
            vocab.setdefault(key + "s", display)
    return {k: v for k, v in vocab.items() if len(k) >= 3}


def _phrases(text: str, max_words: int = 4) -> set[str]:
    words = re.findall(r"[A-Za-z0-9+#./]+", text)
    out: set[str] = set()
    for size in range(1, max_words + 1):
        for index in range(len(words) - size + 1):
            out.add(fold_term(" ".join(words[index:index + size])))
    return {p for p in out if p}


def check_bullet_grounding(bullet: str, project: Project, master: MasterFacts,
                           banned: tuple[str, ...] = ()) -> list[Problem]:
    """A project bullet may only use that project's own evidence."""
    problems = check_style(bullet, banned)
    evidence_text = project.blob
    evidence_metrics = extract_metrics(evidence_text)

    for claim in extract_metrics(bullet):
        if metric_supported(claim, evidence_metrics):
            continue
        nearest = nearest_evidence(claim, evidence_metrics)
        detail = f"; nearest project evidence is {nearest.describe()}" if nearest else \
                 "; this project's evidence states no such number"
        problems.append(Problem("metric", "error",
                                f"{project.project_id} bullet claims {claim.describe()} "
                                f"({claim.raw!r}){detail}"))

    vocab = _vocabulary(master)
    own = _phrases(evidence_text)
    used = _phrases(bullet)
    for phrase in sorted(used & set(vocab)):
        if phrase in own:
            continue
        display = vocab[phrase]
        owners = [p.project_id for p in master.projects
                  if fold_term(display) in _phrases(p.blob) and p.project_id != project.project_id]
        if owners:
            problems.append(Problem("borrowed_technology", "error",
                                    f"{project.project_id} bullet claims {display!r}, which is "
                                    f"evidence from {', '.join(owners)}"))
        elif master.supported_anywhere(display):
            problems.append(Problem("out_of_scope", "error",
                                    f"{project.project_id} bullet claims {display!r}, supported "
                                    f"elsewhere in the master but not by this project"))
        else:
            problems.append(Problem("unsupported_technology", "error",
                                    f"{project.project_id} bullet claims {display!r}, which the "
                                    f"master does not support at all"))

    for head in master.unsupported_heads():
        if re.search(r"(?<![a-z])" + re.escape(head.lower()) + r"(?![a-z])", bullet.lower()):
            problems.append(Problem("explicitly_unsupported", "error",
                                    f"{project.project_id} bullet uses explicitly unsupported "
                                    f"claim {head!r}"))
    return problems


# ==================================================== cover letter grounding

OWNERSHIP_VERBS = ("led", "leading", "lead", "owned", "owning", "managed", "managing",
                   "directed", "drove", "driving", "oversaw", "overseeing", "headed",
                   "spearheaded", "supervised", "coordinated")
_OWNERSHIP_RE = re.compile(r"(?<![a-z])(" + "|".join(OWNERSHIP_VERBS) + r")(?![a-z])",
                           re.IGNORECASE)
_CLAUSE_END = re.compile(r"[.;:]|,\s+and\b|\bwhile\b|\bwhich\b|\bbut\b"
                         r"|\bbecause\b", re.IGNORECASE)


def check_ownership_fusion(text: str, master: MasterFacts) -> list[Problem]:
    """Reject claims that fuse one capsule's ownership onto another's artifact.

    The master's evidence is organized as capsules (one bullet, project or
    evidence-bank entry each). A sentence may draw on one capsule. Saying "led
    the WebSocket pipeline" takes the ownership language from the delivery
    capsule and the artifact from a different capsule that never claimed
    leadership, so it is rejected.
    """
    problems: list[Problem] = []
    vocab = _vocabulary(master)
    capsule_phrases = {c.capsule_id: _phrases(c.text) for c in master.capsules}
    capsule_owns = {c.capsule_id: bool(_OWNERSHIP_RE.search(c.text)) for c in master.capsules}

    for match in _OWNERSHIP_RE.finditer(text):
        rest = text[match.end():]
        stop = _CLAUSE_END.search(rest)
        clause = rest[: stop.start()] if stop else rest[:160]
        distinctive = {p for p in _phrases(clause) if p in vocab}
        if not distinctive:
            continue
        scored = sorted(
            ((len(distinctive & phrases), cid) for cid, phrases in capsule_phrases.items()
             if distinctive & phrases), reverse=True)
        if not scored:
            problems.append(Problem("ownership_transfer", "error",
                                    f"{match.group(1)!r} is claimed over "
                                    f"{sorted(vocab[p] for p in distinctive)}, which no single "
                                    f"evidence capsule supports"))
            continue
        best_score, best = scored[0]
        supporting = [cid for score, cid in scored if score == best_score and capsule_owns[cid]]
        if not supporting:
            problems.append(Problem("ownership_transfer", "error",
                                    f"{match.group(1)!r} is attached to "
                                    f"{sorted(vocab[p] for p in distinctive)} (evidence {best}), "
                                    f"which does not claim ownership of that work"))
    return problems


def validate_cover_letter(text: str, master: MasterFacts, *,
                          masked_terms: tuple[str, ...] = (),
                          banned: tuple[str, ...] = (),
                          jd_text: str | None = None,
                          company: str | None = None,
                          unsupported_concepts: tuple[str, ...] = (),
                          capsules: dict[str, str] | None = None) -> list[Problem]:
    """Style, metric-source, ownership and unsupported-claim gate.

    `jd_text` and `company` enable employer-attributed facts. Without them the
    master stays the only authority, which is the stricter behaviour.

    `unsupported_concepts` carries Python's deterministic verdict about which
    JD-named capabilities the final resume does NOT support. The letter may
    discuss them as the employer's work; it may not claim them as experience.
    """
    problems = check_style(text, banned)
    problems += unsupported_ownership(text, unsupported_concepts, company=company)
    problems += validate_source_scope(text, capsules or {}, master)
    problems += validate_candidate_dates(text, master)
    evidence_metrics = extract_metrics(master.raw, masked_terms)

    for entry in classify_letter_metrics(text, master, jd_text=jd_text, company=company,
                                         masked_terms=masked_terms):
        claim = entry.metric
        if entry.source == "jd":
            continue                    # stated by the posting and kept with the employer
        if entry.source == "master":
            if claim.spelled and entry.support is not None and not entry.support.spelled:
                problems.append(Problem("representation", "warning",
                                        f"letter spells out {claim.raw!r}; the source states it "
                                        f"numerically ({entry.support.raw!r}) so prefer the "
                                        f"numeric form"))
            continue
        nearest = nearest_evidence(claim, evidence_metrics)
        detail = f"; nearest supported value is {nearest.describe()}" if nearest else ""
        if entry.attribution == "employer":
            problems.append(Problem("metric", "error",
                                    f"letter attributes {claim.describe()} ({claim.raw!r}) to "
                                    f"the employer, but the job description does not state it "
                                    f"and the master does not support it{detail}"))
        elif entry.attribution == "candidate":
            problems.append(Problem("metric", "error",
                                    f"letter claims {claim.describe()} ({claim.raw!r}) as the "
                                    f"candidate's own work, which the master does not "
                                    f"support{detail}"))
        else:
            problems.append(Problem("metric", "error",
                                    f"letter claims {claim.describe()} ({claim.raw!r}) which the "
                                    f"master does not support and does not attribute to the "
                                    f"candidate or to the employer{detail}"))

    problems.extend(check_ownership_fusion(text, master))

    for head in master.unsupported_heads():
        if re.search(r"(?<![a-z])" + re.escape(head.lower()) + r"(?![a-z])", text.lower()):
            problems.append(Problem("explicitly_unsupported", "error",
                                    f"letter uses explicitly unsupported claim {head!r}"))
    return problems


# ================================================ cover-letter quality checks

GENERIC_PHRASES = (
    "i am writing to express my strong interest",
    "i would welcome the chance to discuss",
    "i would welcome the opportunity to discuss",
    "perfect fit", "passionate about technology", "dynamic team",
    "hit the ground running", "wealth of experience", "proven track record",
    "think outside the box", "team player", "fast-paced environment",
    "the kind of production ownership this role describes",
    "i am excited about the opportunity to", "my diverse skill set",
)

MARKDOWN_MARKERS = ("**", "##", "```", "- [", "](", "__")


def word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'\-/+.]*", text))


# Word separators that a writer may use interchangeably inside a role name.
# NFKC alone is not enough: it folds U+2011 to U+2010, not to an ASCII hyphen,
# so the whole family is mapped to a space explicitly.
_PHRASE_SEPARATORS = "-\u2010\u2011\u2012\u2013\u2014\u2015\u2212"
_PHRASE_PUNCTUATION = ",.;:!?()[]{}\"'\u2018\u2019\u201c\u201d"
_PHRASE_SPACES = "\u00a0\u2007\u2009\u200a\u202f\u200b"
_PHRASE_TABLE = str.maketrans(
    {ch: " " for ch in _PHRASE_SEPARATORS + _PHRASE_PUNCTUATION + _PHRASE_SPACES})


def normalize_title_phrase(text: str) -> str:
    """Typography-insensitive form used ONLY for role-title matching.

    Strict about the words, tolerant about how they are typed: hyphens, dashes,
    non-breaking spaces, casing and harmless punctuation all collapse to plain
    spaces so "entry-level Software Engineer" and "Entry Level Software
    Engineer" compare equal. This never rewrites the letter itself, so style
    validation still sees (and can still reject) the original characters.
    """
    folded = unicodedata.normalize("NFKC", text or "").translate(_PHRASE_TABLE)
    return re.sub(r"\s+", " ", folded).strip().casefold()


def _normalized_title_forms(job_title: str) -> list[str]:
    """Acceptable renderings of a job title in prose."""
    title = re.sub(r"\s+", " ", job_title or "").strip()
    if not title:
        return []
    forms = {title}
    trimmed = re.sub(r"\s*,?\s*(new grad(uate)?|intern|co-op)\s*\d*\s*$", "", title,
                     flags=re.IGNORECASE).strip()
    forms.add(trimmed)
    forms.add(re.sub(r"\b(Senior|Staff|Lead|Junior)\b\s*", "", trimmed,
                     flags=re.IGNORECASE).strip())
    forms.add(trimmed.split(",")[0].strip())
    # A bare word like "Engineer" would pass on almost any letter, so every
    # accepted form has to carry at least two words of the real title.
    return [f for f in forms if len(f) >= 6 and len(f.split()) >= 2]


def validate_letter_quality(letter: str, *, company: str | None, job_title: str | None,
                            min_words: int = 140, max_words: int = 300) -> list[Problem]:
    """Deterministic quality gate. Different wording from a resume bullet is fine."""
    problems: list[Problem] = []
    text = letter.strip()
    if not text:
        return [Problem("quality", "error", "cover letter is empty")]

    words = word_count(text)
    if words < min_words or words > max_words:
        problems.append(Problem("length", "error",
                                f"cover letter is {words} words; the allowed range is "
                                f"{min_words}-{max_words}"))

    if company:
        head = company.split("(")[0].strip()
        if head and head.lower() not in text.lower():
            problems.append(Problem("tailoring", "error",
                                    f"cover letter never names the company ({head!r})"))
    if job_title:
        forms = _normalized_title_forms(job_title)
        if forms:
            # Compare normalized phrases: the complete title must still appear
            # contiguously, but its typography may differ from the posting's.
            haystack = normalize_title_phrase(text)
            if not any(normalize_title_phrase(f) in haystack for f in forms):
                problems.append(Problem("tailoring", "error",
                                        f"cover letter never names the role; expected one "
                                        f"of {forms[:3]}"))

    for marker in MARKDOWN_MARKERS:
        if marker in text:
            problems.append(Problem("format", "error",
                                    f"cover letter contains markdown ({marker!r})"))
            break
    if EM_DASH in text:
        problems.append(Problem("format", "error", "cover letter contains an em dash"))

    lowered = text.lower()
    for phrase in GENERIC_PHRASES:
        if phrase in lowered:
            problems.append(Problem("generic", "error",
                                    f"cover letter uses placeholder language: {phrase!r}"))

    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paragraphs) < 3:
        problems.append(Problem("structure", "warning",
                                f"cover letter has {len(paragraphs)} paragraph(s); the target "
                                f"shape is four short ones"))
    return problems


def letter_named_terms(letter: str, terms: list[str]) -> list[str]:
    """Which specific named technologies the letter actually mentions.

    Reported alongside the looser theme metric so partial lexical overlap is
    never presented as full coverage of every named term.
    """
    lowered = letter.lower()
    found = []
    for term in terms:
        needle = term.lower()
        bare = re.sub(r"\s*\([^)]*\)", "", needle).strip()
        variants = {needle, bare, bare.rstrip("s"), bare + "s"}
        if any(v and re.search(r"(?<![a-z0-9+#])" + re.escape(v) + r"(?![a-z0-9+#])", lowered)
               for v in variants):
            found.append(term)
    return found


def letter_themes_covered(letter: str, themes: list[str]) -> list[str]:
    """Which JD themes the letter actually touches, by their own terms."""
    lowered = letter.lower()
    covered = []
    for theme in themes:
        tokens = [t for t in re.findall(r"[A-Za-z][A-Za-z0-9+#/.]{2,}", theme.lower())
                  if t not in _THEME_STOPWORDS]
        if tokens and sum(1 for t in tokens if t in lowered) >= max(1, len(tokens) // 3):
            covered.append(theme)
    return covered


_THEME_STOPWORDS = {
    "and", "the", "for", "with", "using", "from", "into", "our", "your", "are", "will",
    "have", "has", "who", "not", "but", "all", "experience", "work", "working", "strong",
    "build", "building", "team", "teams", "engineer", "engineers", "engineering", "software",
    "role", "job", "company", "requirements", "responsibilities", "preferred", "plus",
    "skills", "ability", "familiarity", "knowledge", "such", "other", "well", "write",
    "design", "develop", "including", "across", "within", "that", "this", "their",
}


# =================================================== assessment schema checks

RECOMMENDATIONS = ("strong_apply", "apply", "borderline", "skip")
ELIGIBILITY_STATUS = ("meets", "uncertain", "does_not_meet", "not_applicable")
GAP_IMPORTANCE = ("required", "preferred", "unclear")
GAP_STATUS = ("unsupported", "weak_evidence")
EVIDENCE_SOURCES = ("experience", "project", "education", "skills")
REQUIREMENT_ID_RE = re.compile(r"^REQ-\d{3,}$")

_VAGUE_REQUIREMENT = (
    r"parts of the stack", r"some tools", r"missing parts", r"various technologies",
    r"certain technologies", r"other requirements", r"the rest of", r"general\s+\w+\s+skills",
    r"^\s*(?:some|various|other|misc)\b", r"not evidenced", r"anything else",
)


def _is_vague(requirement: str, jd_text: str | None) -> str | None:
    text = (requirement or "").strip()
    if len(text.split()) < 3:
        return "names no concrete requirement"
    for pattern in _VAGUE_REQUIREMENT:
        if re.search(pattern, text, re.IGNORECASE):
            return f"matches vague pattern {pattern!r}"
    if jd_text:
        tokens = [t for t in re.findall(r"[A-Za-z][A-Za-z0-9+#/.]{2,}", text.lower())
                  if t not in _THEME_STOPWORDS]
        if tokens and not any(t in jd_text.lower() for t in tokens):
            return "names nothing that appears in the job description"
    return None


# Exact synonym tables. A model writing "academic projects" for a source is
# semantically unambiguous and must not invalidate a whole assessment, but
# nothing is guessed: an unlisted value is left alone so validation rejects it.
_SOURCE_ALIASES = {
    "experience": "experience", "professional experience": "experience",
    "professional_experience": "experience", "work experience": "experience",
    "experience section": "experience",
    "project": "project", "projects": "project", "academic project": "project",
    "academic projects": "project", "academic_projects": "project",
    "project work": "project", "projects section": "project",
    "education": "education", "education section": "education",
    "skill": "skills", "skills": "skills", "technical skills": "skills",
    "technical_skills": "skills", "technical skills section": "skills",
    "skills section": "skills",
}

_RECOMMENDATION_ALIASES = {
    "strong_apply": "strong_apply", "strong apply": "strong_apply",
    "strong-apply": "strong_apply", "strongapply": "strong_apply",
    "apply": "apply", "borderline": "borderline", "skip": "skip",
}

_ELIGIBILITY_ALIASES = {
    "meets": "meets", "meet": "meets", "uncertain": "uncertain",
    "does_not_meet": "does_not_meet", "does not meet": "does_not_meet",
    "does-not-meet": "does_not_meet", "doesnotmeet": "does_not_meet",
    "not_applicable": "not_applicable", "not applicable": "not_applicable",
    "not-applicable": "not_applicable", "notapplicable": "not_applicable",
    "n/a": "not_applicable", "na": "not_applicable", "none": "not_applicable",
}

_IMPORTANCE_ALIASES = {
    "required": "required", "require": "required", "must have": "required",
    "preferred": "preferred", "prefer": "preferred", "nice to have": "preferred",
    "unclear": "unclear", "unknown": "unclear",
}

_GAP_STATUS_ALIASES = {
    "unsupported": "unsupported", "not supported": "unsupported",
    "weak_evidence": "weak_evidence", "weak evidence": "weak_evidence",
    "weak-evidence": "weak_evidence",
}


def canonicalize_assessment(data: dict) -> tuple[dict, list[str]]:
    """Map known enum synonyms onto canonical values before validation.

    Case and surrounding whitespace are normalized, then an EXACT table lookup
    decides. There is no fuzzy matching: an unrecognized value passes through
    unchanged and still fails schema validation. Returns the normalized copy
    and the list of substitutions, so the raw value survives in the log.
    """
    if not isinstance(data, dict):
        return data, []
    out = copy.deepcopy(data)
    events: list[str] = []

    def fix(container: dict, key: str, table: dict[str, str], label: str) -> None:
        raw = container.get(key)
        if not isinstance(raw, str):
            return
        canonical = table.get(re.sub(r"\s+", " ", raw).strip().lower())
        if canonical and canonical != raw:
            container[key] = canonical
            events.append(f"{label}: {raw!r} -> {canonical!r}")

    fix(out, "recommendation", _RECOMMENDATION_ALIASES, "recommendation")
    if isinstance(out.get("eligibility"), dict):
        fix(out["eligibility"], "status", _ELIGIBILITY_ALIASES, "eligibility.status")
    for index, match in enumerate(out.get("strong_matches") or [], start=1):
        if isinstance(match, dict):
            fix(match, "source", _SOURCE_ALIASES, f"strong_matches[{index}].source")
    for index, match in enumerate(out.get("partial_matches") or [], start=1):
        if isinstance(match, dict):
            fix(match, "source", _SOURCE_ALIASES, f"partial_matches[{index}].source")
    for index, gap in enumerate(out.get("gaps") or [], start=1):
        if isinstance(gap, dict):
            fix(gap, "importance", _IMPORTANCE_ALIASES, f"gaps[{index}].importance")
            fix(gap, "status", _GAP_STATUS_ALIASES, f"gaps[{index}].status")
    return out, events


def validate_assessment(data: dict, jd_text: str | None = None,
                        requirement_ids: set[str] | None = None) -> list[Problem]:
    """Schema, id-integrity and specificity gate for assessment.json.

    When `requirement_ids` is supplied, every classified requirement must name
    one of Python's authoritative ids exactly once. A model cannot invent a
    requirement identity, and an unknown id is rejected rather than matched.
    """
    problems: list[Problem] = []
    seen_ids: dict[str, str] = {}

    def claim(bucket: str, index: int, raw_id) -> str | None:
        """Validate one requirement_id reference and record who claimed it."""
        if not isinstance(raw_id, str) or not REQUIREMENT_ID_RE.match(raw_id.strip()):
            problems.append(Problem("requirement_id", "error",
                                    f"{bucket}[{index}] must cite a requirement_id like "
                                    f"REQ-001, got {raw_id!r}"))
            return None
        identifier = raw_id.strip()
        if requirement_ids is not None and identifier not in requirement_ids:
            problems.append(Problem("unknown_requirement", "error",
                                    f"{bucket}[{index}] cites unknown requirement_id "
                                    f"{identifier!r}; the model may not invent requirements"))
            return None
        if identifier in seen_ids:
            problems.append(Problem("duplicate_requirement", "error",
                                    f"{identifier} is classified twice: already in "
                                    f"{seen_ids[identifier]}, repeated in {bucket}"))
            return None
        seen_ids[identifier] = bucket
        return identifier

    def require(condition: bool, message: str, kind: str = "schema") -> None:
        if not condition:
            problems.append(Problem(kind, "error", message))

    score = data.get("fit_score")
    require(isinstance(score, (int, float)) and 0 <= float(score) <= 10,
            f"fit_score must be a number in 0-10, got {score!r}")
    require(data.get("recommendation") in RECOMMENDATIONS,
            f"recommendation must be one of {RECOMMENDATIONS}, got "
            f"{data.get('recommendation')!r}")
    summary = (data.get("summary") or "").strip()
    require(bool(summary), "summary is missing")
    require(len(re.findall(r"[.!?]", summary)) <= 4,
            "summary should be at most three concise sentences")

    eligibility = data.get("eligibility")
    if not isinstance(eligibility, dict):
        require(False, "eligibility block is missing")
    else:
        require(eligibility.get("status") in ELIGIBILITY_STATUS,
                f"eligibility.status must be one of {ELIGIBILITY_STATUS}, got "
                f"{eligibility.get('status')!r}")
        require(isinstance(eligibility.get("details"), list),
                "eligibility.details must be a list")

    for index, match in enumerate(data.get("strong_matches") or [], start=1):
        if not isinstance(match, dict):
            require(False, f"strong_matches[{index}] is not an object")
            continue
        claim("strong_matches", index, match.get("requirement_id"))
        # Per-entry prose is OPTIONAL: enforce_verdicts replaces every bucket
        # with Python's own entries, so nothing the model writes here survives
        # and requiring it only spends output tokens. What still matters is
        # that the id is real and claimed once - that is the hallucination
        # check. Anything the model does volunteer is still held to the enum.
        if match.get("source") is not None:
            require(match.get("source") in EVIDENCE_SOURCES,
                    f"strong_matches[{index}].source must be one of {EVIDENCE_SOURCES}, "
                    f"got {match.get('source')!r}")

    for index, match in enumerate(data.get("partial_matches") or [], start=1):
        if not isinstance(match, dict):
            require(False, f"partial_matches[{index}] is not an object")
            continue
        claim("partial_matches", index, match.get("requirement_id"))

    for index, gap in enumerate(data.get("gaps") or [], start=1):
        if not isinstance(gap, dict):
            require(False, f"gaps[{index}] is not an object")
            continue
        claim("gaps", index, gap.get("requirement_id"))
        if gap.get("importance") is not None:
            require(gap.get("importance") in GAP_IMPORTANCE,
                    f"gaps[{index}].importance must be one of {GAP_IMPORTANCE}, got "
                    f"{gap.get('importance')!r}")
        if gap.get("status") is not None:
            require(gap.get("status") in GAP_STATUS,
                    f"gaps[{index}].status must be one of {GAP_STATUS}, got "
                    f"{gap.get('status')!r}")
        # Materialized text is Python's; if present it must still be specific.
        materialized = (gap.get("requirement") or "").strip()
        if materialized:
            vague = _is_vague(materialized, jd_text)
            if vague:
                problems.append(Problem("vague_gap", "error",
                                        f"gaps[{index}] is not actionable ({vague}): "
                                        f"{materialized[:80]!r}"))

    for index, entry in enumerate(data.get("manual_review") or [], start=1):
        if not isinstance(entry, dict):
            require(False, f"manual_review[{index}] is not an object")
            continue
        claim("manual_review", index, entry.get("requirement_id"))

    require(isinstance(data.get("complementary_strengths", []), list),
            "complementary_strengths must be a list")

    # Eligibility must not contradict how its own requirements are classified.
    eligibility_status = (data.get("eligibility") or {}).get("status") \
        if isinstance(data.get("eligibility"), dict) else None
    if eligibility_status == "uncertain":
        for index, gap in enumerate(data.get("gaps") or [], start=1):
            if isinstance(gap, dict) and gap.get("status") == "unsupported" \
                    and (gap.get("kind") == "eligibility"):
                problems.append(Problem(
                    "eligibility_inconsistent", "error",
                    f"eligibility.status is 'uncertain' but gaps[{index}] marks the same "
                    f"eligibility requirement categorically 'unsupported'"))

    tailoring = data.get("tailoring_quality")
    if not isinstance(tailoring, dict):
        require(False, "tailoring_quality block is missing")
    else:
        value = tailoring.get("score")
        require(isinstance(value, (int, float)) and 0 <= float(value) <= 10,
                f"tailoring_quality.score must be a number in 0-10, got {value!r}")
        require(isinstance(tailoring.get("notes"), list),
                "tailoring_quality.notes must be a list")
    require(isinstance(data.get("risk_flags", []), list), "risk_flags must be a list")
    return problems


# ======================================== evidence breadth and guard rails


def assessment_breadth(assessment: dict, *, scored: int, signals: int,
                       manual: int) -> dict:
    """Deterministic metadata about how much evidence the score rests on.

    The fit score itself is untouched. This only says how broad its basis is, so
    a high number off one scored requirement is visibly low-confidence.
    """
    strong = len(assessment.get("strong_matches") or [])
    partial = len(assessment.get("partial_matches") or [])
    if scored >= 4:
        coverage = (strong + 0.5 * partial) / scored
        confidence = "high" if coverage >= 0.6 else "normal"
    elif scored >= 2:
        confidence = "medium"
    else:
        confidence = "low"
    return {"scored_requirement_count": scored, "role_signal_count": signals,
            "manual_review_count": manual, "fit_confidence": confidence}


def cap_recommendation(assessment: dict, requirements) -> tuple[str, str] | None:
    """An unresolved hard eligibility requirement forbids `strong_apply`.

    This draws no immigration conclusion: the requirement stays unresolved and
    routed to a human. It only refuses to call such a posting a strong apply.
    Returns (capped_recommendation, reason) or None when nothing changes.
    """
    blocking = [r for r in requirements
                if getattr(r, "kind", "") == "eligibility"
                and getattr(r, "authority", "") == "hard"
                and getattr(r, "importance", "") == "required"]
    if not blocking:
        return None
    status = (assessment.get("eligibility") or {}).get("status")
    if status in ("meets", "not_applicable"):
        return None
    if assessment.get("recommendation") != "strong_apply":
        return None
    ids = ", ".join(r.requirement_id for r in blocking)
    return ("apply", f"required eligibility unresolved ({ids})")


# A capped verdict makes the provider's own closing phrase stale.
_STALE_VERDICT = ("strong apply", "strong_apply", "strongly apply")

_SUMMARY_SENTENCES = re.compile(r"(?<=[.!?])\s+")

# The connective that introduced the verdict clause (", so this posting is ...").
_VERDICT_CLAUSE = re.compile(r"\s*[,;]\s*(?:so|therefore|thus|hence|making|which makes)\b"
                             r"|\s+(?:therefore|thus|hence)\b", re.IGNORECASE)


def _drop_stale_verdict(sentence: str) -> str | None:
    """Keep a sentence's factual part, cutting any claim of the old verdict."""
    lowered = sentence.lower()
    hits = [lowered.find(phrase) for phrase in _STALE_VERDICT if phrase in lowered]
    if not hits:
        return sentence
    head = sentence[:min(hits)]
    clause = None
    for match in _VERDICT_CLAUSE.finditer(head):
        clause = match
    if clause:
        head = head[:clause.start()]
    head = head.rstrip(" ,;:-").rstrip()
    return f"{head}." if len(head) >= 25 else None


def reconcile_summary(assessment: dict, reason: str) -> bool:
    """Make the user-facing summary agree with a capped recommendation.

    No model is consulted. The provider's original prose is preserved verbatim
    as `model_summary`; the stale verdict clause is removed from `summary` and
    replaced with the deterministic outcome. Returns True when it rewrote.
    """
    summary = (assessment.get("summary") or "").strip()
    target = assessment.get("recommendation")
    if not summary or not target:
        return False
    kept = [out for out in
            (_drop_stale_verdict(s.strip()) for s in _SUMMARY_SENTENCES.split(summary))
            if out]
    scored = assessment.get("scored_requirement_count")
    confidence = assessment.get("fit_confidence")
    basis = ""
    if scored is not None and confidence:
        basis = (f"The fit score rests on {scored} scored requirement(s) "
                 f"({confidence} evidence breadth), and t")
    kept.append(f"{basis or 'T'}he final recommendation is {target} after a deterministic "
                f"cap: {reason}.")
    rewritten = " ".join(kept)
    if rewritten == summary:
        return False
    assessment["model_summary"] = summary
    assessment["summary"] = rewritten
    return True


# ===================================== immutable verdicts: prose guard rails

# Phrases that deny evidence. Harmless in a genuine gap, false in any bucket
# where Python found the evidence present.
_DENIAL_PHRASES = (
    "does not list", "does not have", "not listed", "no bachelor", "no master",
    "missing", "lacks", "absent", "no evidence", "not present", "no degree",
    "without a degree", "not documented", "no documented",
)


def denies_evidence(text: str) -> str | None:
    """The denial phrase in `text`, if it asserts evidence is absent."""
    lowered = " ".join((text or "").lower().split())
    for phrase in _DENIAL_PHRASES:
        if phrase in lowered:
            return phrase
    return None


def enforce_verdicts(data: dict, verdicts: dict, table: dict) -> list[str]:
    """Overwrite every verdict field with Python's, keeping usable model prose.

    The model may describe a verdict in the summary; it may never change one and
    no model prose survives on a requirement entry. Buckets, evidence,
    limitations and eligibility all come from `verdicts`. Returns a list of
    human-readable override notes.
    """
    notes: list[str] = []
    authoritative = {b: [dict(e) for e in (verdicts.get(b) or [])]
                     for b in ("strong_matches", "partial_matches", "gaps",
                               "manual_review")}
    model_index: dict[str, str] = {}
    for bucket in authoritative:
        for entry in data.get(bucket) or []:
            if isinstance(entry, dict) and entry.get("requirement_id"):
                model_index[entry["requirement_id"].strip()] = bucket

    for bucket, entries in authoritative.items():
        for entry in entries:
            identifier = entry["requirement_id"]
            requirement = table.get(identifier)
            if requirement is not None:
                entry["requirement"] = requirement.original_text
                entry["kind"] = requirement.kind
            was = model_index.get(identifier)
            if was and was != bucket:
                notes.append(f"{identifier}: model said {was}, Python says {bucket}")
            # No model prose survives on a requirement entry. A sentence with no
            # denial phrase can still smuggle in unsupported technology, so
            # Python's evidence is the complete authority.
            entry.pop("explanation", None)
        data[bucket] = entries

    for identifier, bucket in model_index.items():
        if identifier not in {e["requirement_id"] for b in authoritative
                              for e in authoritative[b]}:
            notes.append(f"{identifier}: model classified a requirement Python does not "
                         f"score; dropped")

    data["eligibility"] = {
        "status": verdicts["eligibility"]["status"],
        "details": list(verdicts["eligibility"]["details"]),
    }
    data["verdict_source"] = "python_deterministic"
    return notes


def summary_contradicts_verdicts(summary: str, data: dict) -> str | None:
    """A denial in the summary about something Python classified as present."""
    denial = denies_evidence(summary)
    if not denial:
        return None
    present = (data.get("strong_matches") or []) + (data.get("partial_matches") or [])
    lowered = " ".join((summary or "").lower().split())
    for entry in present:
        text = (entry.get("requirement") or "").lower()
        if "degree" in text and ("degree" in lowered or "bachelor" in lowered):
            return denial
        if "course work" in text and "coursework" in lowered.replace(" ", ""):
            return denial
    return None


def deterministic_summary(data: dict) -> str:
    """User-facing summary built from the FINAL Python-authoritative state.

    Exactly three sentences at most, to stay inside the concise-summary policy:
    requirement coverage, then eligibility/manual-review/breadth joined into one
    sentence, then the final (possibly capped) recommendation.
    """
    strong = len(data.get("strong_matches") or [])
    partial = len(data.get("partial_matches") or [])
    gaps = len(data.get("gaps") or [])
    manual = len(data.get("manual_review") or [])
    scored = data.get("scored_requirement_count")
    total = scored if isinstance(scored, int) and scored else strong + partial + gaps

    # Sentence 1: coverage.
    if total:
        head = (f"{strong} of {total} scored requirement(s) "
                f"{'is' if strong == 1 else 'are'} directly evidenced")
    else:
        head = "no scored requirements were extracted from this posting"
    tail = "none are unsupported" if not gaps else (
        f"{gaps} {'is' if gaps == 1 else 'are'} unsupported")
    if partial:
        coverage = (f"{head}, {partial} {'is' if partial == 1 else 'are'} partly "
                    f"evidenced, and {tail}.")
    else:
        coverage = f"{head} and {tail}."
    sentences = [coverage]

    # Sentence 2: eligibility, manual review and breadth, semicolon-joined.
    status = (data.get("eligibility") or {}).get("status")
    context = {
        "not_applicable": "No separate work-authorization or graduation requirement was "
                          "identified",
        "uncertain": "Required work-authorization eligibility remains unresolved",
        "does_not_meet": "A stated eligibility requirement is not satisfied by the "
                         "recorded facts",
        "meets": "The posting's stated eligibility requirement is satisfied",
    }.get(status)
    clauses: list[str] = []
    if context:
        if manual:
            clauses.append(f"{context} and {manual} requirement(s) need manual review")
        else:
            clauses.append(context)
    elif manual:
        clauses.append(f"{manual} requirement(s) need manual review")
    confidence = data.get("fit_confidence")
    if confidence:
        clauses.append(f"evidence breadth is {confidence}")
    if clauses:
        sentences.append("; ".join(clauses) + ".")

    # Sentence 3: the final recommendation, after any cap.
    recommendation = data.get("recommendation")
    if recommendation:
        sentences.append(f"Final recommendation: {recommendation}.")
    return " ".join(sentences)


def finalize_assessment(data: dict, *, verdicts: dict, table: dict, requirements,
                        scored: int, signals: int, manual: int) -> list[str]:
    """Apply the authoritative tail of the assessment, in a fixed order.

    3 enforce Python verdicts/evidence -> 4 deterministic eligibility ->
    5 recommendation cap -> 6 breadth/confidence -> 7 preserve provider prose
    as model_summary -> 8 generate the deterministic summary LAST, so it always
    agrees with the final recommendation.
    """
    notes = enforce_verdicts(data, verdicts, table)          # steps 3 and 4

    capped = cap_recommendation(data, requirements)          # step 5
    if capped:
        target, reason = capped
        notes.append(f"recommendation capped {data.get('recommendation')} -> {target} "
                     f"reason={reason}")
        data["recommendation_capped"] = {"from": data.get("recommendation"), "to": target,
                                         "reason": reason}
        data["recommendation"] = target

    data.update(assessment_breadth(data, scored=scored, signals=signals,  # step 6
                                   manual=manual))

    # A risk flag denying evidence Python found is false; drop it.
    present = {entry["requirement_id"] for entry in
               (data.get("strong_matches") or []) + (data.get("partial_matches") or [])}
    kept = []
    for flag in data.get("risk_flags") or []:
        if denies_evidence(flag) and any(identifier in flag for identifier in present):
            notes.append(f"dropped risk flag denying evidence: {flag}")
            continue
        kept.append(flag)
    data["risk_flags"] = kept

    data["model_summary"] = (data.get("summary") or "").strip()   # step 7
    data["summary"] = deterministic_summary(data)                 # step 8
    return notes


# ============================================ final deterministic validation

# The only fields Python owns on a requirement entry. Anything else on a final
# entry means model-authored content survived.
FINAL_ENTRY_FIELDS = {"requirement_id", "requirement", "kind", "evidence", "source",
                      "limitation", "detail", "reason", "importance", "status"}

_FINAL_BUCKETS = ("strong_matches", "partial_matches", "gaps", "manual_review")
_SCORED_BUCKETS = ("strong_matches", "partial_matches", "gaps")

_STALE_RECOMMENDATIONS = {
    "apply": ("strong apply", "strong_apply", "borderline", "skip"),
    "strong_apply": ("borderline", "skip"),
    "borderline": ("strong apply", "strong_apply", "skip"),
    "skip": ("strong apply", "strong_apply", "borderline"),
}


def _bucket_index(source: dict) -> dict[str, str]:
    index: dict[str, str] = {}
    for bucket in _FINAL_BUCKETS:
        for entry in source.get(bucket) or []:
            if isinstance(entry, dict) and entry.get("requirement_id"):
                index[entry["requirement_id"]] = bucket
    return index


def validate_final_assessment(data: dict, *, verdicts: dict, table: dict,
                              requirements) -> list[Problem]:
    """Integrity gate for the FINAL Python-authoritative assessment.

    This is the second of two boundaries. The provider contract is enforced
    before finalization by validate_assessment; this one enforces what must be
    true of Python's own output, and never applies provider prose rules.
    """
    problems: list[Problem] = []

    def fail(kind: str, message: str) -> None:
        problems.append(Problem(kind, "error", message))

    scored_ids = [r.requirement_id for r in requirements if getattr(r, "scored", False)]
    manual_ids = [r.requirement_id for r in requirements
                  if getattr(r, "kind", "") in ("eligibility", "logistics", "condition")
                  or (getattr(r, "subjective", False) and getattr(r, "authority", "") == "hard")]
    signal_ids = {r.requirement_id for r in requirements
                  if getattr(r, "authority", "") == "signal"}
    known = set(table)

    # ---- A. requirement authority ---------------------------------------
    seen: dict[str, list[str]] = {}
    for bucket in _FINAL_BUCKETS:
        entries = data.get(bucket)
        if not isinstance(entries, list):
            fail("final_schema", f"{bucket} must be a list")
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                fail("final_schema", f"{bucket} holds a non-object entry")
                continue
            identifier = (entry.get("requirement_id") or "").strip()
            if not REQUIREMENT_ID_RE.match(identifier):
                fail("final_requirement_id",
                     f"{bucket} entry has no valid requirement_id ({identifier!r})")
                continue
            if identifier not in known:
                fail("final_unknown_requirement",
                     f"{bucket} cites unknown requirement_id {identifier}")
            seen.setdefault(identifier, []).append(bucket)

    for identifier, buckets in seen.items():
        if len(buckets) > 1:
            fail("final_duplicate_requirement",
                 f"{identifier} appears in {len(buckets)} buckets: {', '.join(buckets)}")
    for identifier in scored_ids:
        where = seen.get(identifier, [])
        if not where:
            fail("final_missing_requirement",
                 f"scored requirement {identifier} is absent from the final assessment")
        elif where[0] not in _SCORED_BUCKETS:
            fail("final_wrong_bucket",
                 f"scored requirement {identifier} is in {where[0]}")
    for identifier in manual_ids:
        if identifier in scored_ids:
            continue
        if seen.get(identifier, []) and seen[identifier][0] != "manual_review":
            fail("final_wrong_bucket",
                 f"manual-review requirement {identifier} is in {seen[identifier][0]}")
    for identifier in signal_ids:
        if seen.get(identifier, [None])[0] in _SCORED_BUCKETS:
            fail("final_signal_scored",
                 f"role signal {identifier} was scored in {seen[identifier][0]}")

    # ---- B. verdict consistency with Python's own classification ---------
    expected = _bucket_index(verdicts)
    actual = _bucket_index(data)
    if actual != expected:
        moved = {k: (expected.get(k), actual.get(k)) for k in set(expected) | set(actual)
                 if expected.get(k) != actual.get(k)}
        fail("final_verdict_mismatch",
             "final buckets differ from Python's verdicts: "
             + "; ".join(f"{k}: expected {want}, got {got}" for k, (want, got)
                         in sorted(moved.items())))

    # ---- C. only Python-owned fields on an entry -------------------------
    for bucket in _FINAL_BUCKETS:
        for entry in data.get(bucket) or []:
            if not isinstance(entry, dict):
                continue
            extra = sorted(set(entry) - FINAL_ENTRY_FIELDS)
            if extra:
                fail("final_entry_fields",
                     f"{bucket} entry {entry.get('requirement_id')} carries "
                     f"non-Python field(s): {', '.join(extra)}")

    # ---- D. eligibility equals the deterministic verdict -----------------
    want_eligibility = verdicts.get("eligibility") or {}
    got_eligibility = data.get("eligibility")
    if not isinstance(got_eligibility, dict):
        fail("final_eligibility", "eligibility block is missing")
    else:
        if got_eligibility.get("status") != want_eligibility.get("status"):
            fail("final_eligibility",
                 f"eligibility.status is {got_eligibility.get('status')!r}, but the "
                 f"deterministic verdict is {want_eligibility.get('status')!r}")
        if got_eligibility.get("status") not in ELIGIBILITY_STATUS:
            fail("final_eligibility",
                 f"eligibility.status must be one of {ELIGIBILITY_STATUS}")

    # ---- E. recommendation cap ------------------------------------------
    outstanding = cap_recommendation(data, requirements)
    if outstanding:
        fail("final_recommendation_cap",
             f"recommendation is {data.get('recommendation')!r} but an eligibility cap "
             f"still applies ({outstanding[1]})")
    if (got_eligibility or {}).get("status") == "not_applicable":
        capped = data.get("recommendation_capped") or {}
        if "eligibility" in (capped.get("reason") or ""):
            fail("final_recommendation_cap",
                 "an eligibility cap fired even though eligibility is not_applicable")

    # ---- F. summary ------------------------------------------------------
    summary = (data.get("summary") or "").strip()
    recommendation = data.get("recommendation")
    if not summary:
        fail("final_summary", "summary is missing")
    else:
        terminators = len(re.findall(r"[.!?]", summary))
        if terminators > 3:
            fail("final_summary",
                 f"summary has {terminators} sentence terminators; at most 3 are allowed")
        if recommendation and recommendation not in summary:
            fail("final_summary",
                 f"summary does not state the final recommendation ({recommendation})")
        lowered = summary.lower()
        for stale in _STALE_RECOMMENDATIONS.get(recommendation, ()):
            if stale in lowered:
                fail("final_summary",
                     f"summary names a stale recommendation {stale!r} while the final "
                     f"recommendation is {recommendation!r}")
        denial = denies_evidence(summary)
        if denial and ((data.get("strong_matches") or []) or
                       (data.get("partial_matches") or [])):
            fail("final_summary",
                 f"summary uses denial language {denial!r} while evidence is present")

    # ---- G. basic final schema ------------------------------------------
    score = data.get("fit_score")
    if not (isinstance(score, (int, float)) and 0 <= float(score) <= 10):
        fail("final_schema", f"fit_score must be a number in 0-10, got {score!r}")
    if recommendation not in RECOMMENDATIONS:
        fail("final_schema", f"recommendation must be one of {RECOMMENDATIONS}")
    tailoring = data.get("tailoring_quality")
    if not isinstance(tailoring, dict) or not isinstance(
            tailoring.get("score"), (int, float)):
        fail("final_schema", "tailoring_quality.score must be a number")
    elif not 0 <= float(tailoring["score"]) <= 10:
        fail("final_schema", "tailoring_quality.score must be in 0-10")
    for field_name in ("complementary_strengths", "risk_flags"):
        if not isinstance(data.get(field_name, []), list):
            fail("final_schema", f"{field_name} must be a list")
    if data.get("verdict_source") != "python_deterministic":
        fail("final_verdict_source",
             f"verdict_source must be 'python_deterministic', got "
             f"{data.get('verdict_source')!r}")
    return problems


def unsupported_ownership(text: str, unsupported_concepts: tuple[str, ...], *,
                          company: str | None = None) -> list[Problem]:
    """Reject candidate-owned phrasing for a deterministically unsupported concept.

    The concept itself is never banned: the employer builds distributed systems
    and the letter may say so. What fails is claiming it as the candidate's own
    experience when Python's requirement verdict says the resume does not
    support it. Ownership is decided by the nearest marker, the same rule the
    metric validator uses.
    """
    if not unsupported_concepts:
        return []
    normalized = normalize_typography(text)
    lowered = normalized.lower()
    problems: list[Problem] = []
    for concept in unsupported_concepts:
        needle = concept.lower()
        pattern = r"(?<![a-z0-9+#])" + re.escape(needle) + r"(?![a-z0-9+#])"
        for match in re.finditer(pattern, lowered):
            owner = _attribution(_sentence_prefix(normalized, match.start()), company)
            if owner == "candidate":
                problems.append(Problem(
                    "unsupported_claim", "error",
                    f"letter presents {concept!r} as the candidate's own experience, but "
                    f"the final resume does not support it; describe it as the employer's "
                    f"work or use supported wording instead"))
                break
    return problems


# ================================================ source-scoped attribution


def _source_anchors(capsules: dict[str, str], master: MasterFacts) -> dict[str, str]:
    """Anchors that belong to exactly ONE factual source.

    A technology or metric naming a single role or project identifies that
    source. Anything shared (Python, RDS) is useless for attribution and is
    dropped, so only unambiguous evidence drives a violation.
    """
    sources = {label: text for label, text in capsules.items() if label != "general"}
    if not sources:
        return {}
    terms: set[str] = {skill.name for skill in master.skills}
    for project in master.projects:
        terms.update(project.tech)
    for text in sources.values():
        for metric in extract_metrics(text):
            raw = metric.raw.strip()
            if len(raw) >= 4 and APPROX_MARK not in raw:
                terms.add(raw)

    anchors: dict[str, str] = {}
    for term in terms:
        if len(term) < 4:
            continue
        needles = _skill_needles(term)
        owners = [label for label, text in sources.items() if _mentions(text, needles)]
        if len(owners) == 1:
            anchors[term] = owners[0]
    return anchors


def validate_source_scope(letter: str, capsules: dict[str, str],
                          master: MasterFacts) -> list[Problem]:
    """One experience/project paragraph must stay inside one source.

    Existing grounding proves a fact exists SOMEWHERE in the master. This adds
    the missing constraint: facts may not migrate between sources, so AWS EC2
    cannot appear in the HeinOnline paragraph and a capability with no capsule
    evidence cannot be attributed to a source at all.
    """
    if not capsules:
        return []
    anchors = _source_anchors(capsules, master)
    if not anchors:
        return []
    problems: list[Problem] = []
    paragraphs = [p for p in re.split(r"\n\s*\n", letter) if p.strip()]
    for position, paragraph in enumerate(paragraphs, start=1):
        hits: dict[str, list[str]] = {}
        for term, source in anchors.items():
            if _mentions(paragraph, _skill_needles(term)):
                hits.setdefault(source, []).append(term)
        if len(hits) > 1:
            detail = "; ".join(f"{source} via {', '.join(sorted(terms)[:3])}"
                               for source, terms in sorted(hits.items()))
            problems.append(Problem(
                "source_scope", "error",
                f"paragraph {position} mixes evidence from {len(hits)} different sources "
                f"({detail}); one paragraph must stay within a single role or project"))
            continue
        if not hits:
            continue
        dominant = next(iter(hits))
        for skill in master.skills:
            if len(skill.name) < 4:
                continue
            needles = _skill_needles(skill.name)
            if not _mentions(paragraph, needles):
                continue
            if any(_mentions(text, needles) for text in capsules.values()):
                continue
            problems.append(Problem(
                "source_scope", "error",
                f"paragraph {position} claims {skill.name!r} as part of {dominant}, but no "
                f"source evidence supports it there"))
    return problems


# ============================== candidate education / authorization dates

_MONTH_NUMBERS = {"january": 1, "february": 2, "march": 3, "april": 4, "may": 5,
                  "june": 6, "july": 7, "august": 8, "september": 9, "october": 10,
                  "november": 11, "december": 12}

_CLAIM_DATE_RE = re.compile(
    r"\b(?P<month>" + "|".join(_MONTH_NUMBERS) + r")\w*\s+"
    r"(?:(?P<day>\d{1,2})\s*,?\s*)?(?P<year>(?:19|20)\d{2})\b", re.IGNORECASE)

# First person: only a candidate-owned sentence is constrained.
_FIRST_PERSON = re.compile(r"(?<![a-z])(?:I|my|me)(?![a-z])", re.IGNORECASE)

_GRADUATION_CLAIM = re.compile(
    r"\b(?:graduat\w*|conferral|finish\w*|complet\w*)\b", re.IGNORECASE)

_AUTHORIZATION_CLAIM = re.compile(
    r"\b(?:OPT|EAD|CPT|work[-\s]authoriz\w*|sponsorship|visa|start(?:ing)?\b|begin\w*|"
    r"availab\w*|join\w*)\b", re.IGNORECASE)

_OPT_QUALIFIER = re.compile(r"subject to\b[^.]{0,40}\bauthoriz", re.IGNORECASE)


def _opt_fact(master: MasterFacts) -> tuple[tuple[int, int], str] | None:
    """The canonical post-completion OPT eligibility date from standing facts."""
    for fact in master.standing_facts:
        if not re.search(r"\bOPT\b", fact, re.IGNORECASE):
            continue
        match = _CLAIM_DATE_RE.search(fact)
        if match:
            month = _MONTH_NUMBERS[match.group("month").lower()]
            return (int(match.group("year")), month), fact
    return None


def validate_candidate_dates(letter: str, master: MasterFacts) -> list[Problem]:
    """Candidate education / work-authorization dates must be the master's.

    A posting's eligibility window is the EMPLOYER's date and may be quoted
    freely, but it must never become the candidate's graduation or start date,
    and an OPT start date is never inferred. Only first-person sentences are
    constrained, so quoting the JD stays legal.
    """
    graduation = expected_graduation(master)
    opt = _opt_fact(master)
    problems: list[Problem] = []
    for sentence in re.split(r"(?<=[.!?])\s+", normalize_typography(letter)):
        if not _FIRST_PERSON.search(sentence):
            continue                     # employer/JD statement: not constrained
        dates = [( int(m.group("year")), _MONTH_NUMBERS[m.group("month").lower()])
                 for m in _CLAIM_DATE_RE.finditer(sentence)]
        graduation_claim = bool(_GRADUATION_CLAIM.search(sentence))
        authorization_claim = bool(_AUTHORIZATION_CLAIM.search(sentence))
        if not (graduation_claim or authorization_claim):
            continue

        if graduation_claim and graduation:
            for found in dates:
                if found != graduation:
                    problems.append(Problem(
                        "candidate_date", "error",
                        f"letter claims the candidate graduates/finishes in "
                        f"{_MONTH_LABEL(found)}, but the master states expected graduation "
                        f"{_MONTH_LABEL(graduation)}; a posting's eligibility window is not "
                        f"the candidate's date"))
        if authorization_claim and opt:
            opt_date, fact = opt
            for found in dates:
                if graduation_claim and found == graduation:
                    continue             # already judged as a graduation claim
                if found != opt_date:
                    problems.append(Problem(
                        "candidate_date", "error",
                        f"letter claims the candidate can start/work from "
                        f"{_MONTH_LABEL(found)}; the only supported fact is {fact!r}. An OPT "
                        f"start date is never inferred"))
            if re.search(r"\bOPT\b", sentence) and not _OPT_QUALIFIER.search(sentence):
                problems.append(Problem(
                    "candidate_date", "error",
                    "letter presents OPT work authorization without the supported "
                    "qualifier ('subject to OPT/EAD authorization'); omit OPT unless the "
                    "posting makes work authorization material"))
    return problems


def _MONTH_LABEL(value: tuple[int, int]) -> str:
    names = {number: name.capitalize() for name, number in _MONTH_NUMBERS.items()}
    return f"{names.get(value[1], '?')} {value[0]}"


# ================================================ compact assessment report

# The user-facing report. Component scores, explanations, confidences, source
# URLs and provider metadata belong in strategy.json, never here.
ASSESSMENT_FIELDS = ("Resume Tailoring Score", "Experience Selection Score",
                     "Project Selection Score", "Project Bullet Quality Score",
                     "Cover Letter Score", "Fit Match Score", "Callback Likelihood",
                     "Company Visa Sponsorship", "Company STEM OPT Support",
                     "Job Posted")

# Retired field names. Kept only so a stale artifact can be recognised.
RETIRED_ASSESSMENT_FIELDS = ("Resume Score", "Can Expect Callback", "Callback Confidence",
                             "Visa Sponsorship", "STEM OPT Extension")

# Explicit sponsorship language, in both directions.
_SPONSOR_YES = (r"\bwe (?:do|will) sponsor\b", r"\bsponsorship (?:is )?available\b",
                r"\bwill provide (?:visa )?sponsorship\b",
                r"\bvisa sponsorship (?:is )?(?:available|offered|provided)\b",
                r"\bwe offer (?:visa )?sponsorship\b", r"\bh-?1b sponsorship available\b")
_SPONSOR_NO = (r"\b(?:do|does|will) not (?:offer|provide|sponsor)\b[^.]{0,40}\bsponsor",
               r"\bno (?:visa )?sponsorship\b", r"\bunable to sponsor\b",
               r"\bcannot sponsor\b", r"\bwithout (?:visa )?sponsorship\b",
               r"\bsponsorship (?:is )?not (?:available|offered|provided)\b")

# STEM OPT is a candidate/employer fact, never inferred from E-Verify.
_STEM_OPT_YES = (r"\bSTEM OPT\b[^.]{0,40}\b(?:eligible|extension available|supported)\b",
                 r"\bsupports? the STEM OPT extension\b")
_STEM_OPT_NO = (r"\bno STEM OPT\b", r"\bSTEM OPT\b[^.]{0,30}\bnot (?:available|supported)\b",
                r"\bdoes not support the STEM OPT\b")

_POSTED_PATTERNS = (
    r"(?:posted|published|posting date|date posted)\s*(?:on|:)?\s*"
    r"(?P<value>[A-Z][a-z]+\s+\d{1,2},?\s+(?:19|20)\d{2})",
    r"(?:posted|published)\s*(?:on|:)?\s*(?P<value>\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2})?)",
    r"(?:posted|published)\s*(?:on|:)?\s*(?P<value>\d{1,2}/\d{1,2}/(?:19|20)\d{2})",
)


def _explicit(text: str, yes: tuple[str, ...], no: tuple[str, ...]) -> str:
    """YES / NO only on explicit language, UNKNOWN otherwise."""
    lowered = text or ""
    if any(re.search(pattern, lowered, re.IGNORECASE) for pattern in no):
        return "NO"
    if any(re.search(pattern, lowered, re.IGNORECASE) for pattern in yes):
        return "YES"
    return "UNKNOWN"


def job_posted_date(jd_text: str, metadata: dict | None = None) -> str:
    """A posting date only when the input actually states one.

    Never the file's creation time, the pipeline's run time or a download
    time: absent evidence is UNKNOWN.
    """
    for key in ("posted_at", "job_posted", "posted"):
        value = (metadata or {}).get(key)
        if value:
            return str(value).strip()
    for pattern in _POSTED_PATTERNS:
        match = re.search(pattern, jd_text or "", re.IGNORECASE)
        if match:
            return re.sub(r"\s+", " ", match.group("value")).strip()
    return "UNKNOWN"


_GROUNDING_KINDS = frozenset({
    "metric", "source_scope", "borrowed_technology", "ownership_transfer",
    "unsupported_technology", "unsupported_claim", "explicitly_unsupported",
    "candidate_date", "representation", "out_of_scope",
})


def cover_letter_score(*, letter: str, problems: list, words: int, priorities=None,
                       themes_covered: int = 0, evidence_sources: int = 0,
                       named_terms_covered: int = 0, company: str = "",
                       job_title: str = "") -> float:
    """The compact Cover Letter Score, on the agreed five-part rubric.

    30% factual grounding, 30% distinctive-JD evidence coverage, 20% strength
    of the evidence chosen, 10% company/role specificity, 10% writing quality.
    A blocking validation failure short-circuits all of it: an invalid letter
    is not a usable artifact and does not get graded on relevance.

    This grades the artifact. It is not a validation contract and cannot make
    a letter pass or fail - generation validation is unchanged.
    """
    blocking = [p for p in problems if getattr(p, "severity", "") == "error"]
    if blocking:
        return round(max(0.0, 5.0 - len(blocking)), 1)
    warnings = [p for p in problems if getattr(p, "severity", "") == "warning"]
    kinds = [getattr(p, "kind", "") for p in warnings]

    # -- 30%: factual grounding / source correctness ---------------------
    grounding_hits = sum(1 for kind in kinds if kind in _GROUNDING_KINDS)
    factual = max(0.0, 3.0 - 1.0 * grounding_hits)

    # -- 30%: distinctive-JD evidence coverage ---------------------------
    supported = supported_priorities(priorities)
    if supported:
        distinctive = 2.0 if priority_addressed(letter, supported[0]) else 0.0
        extra = sum(1 for p in supported[1:] if priority_addressed(letter, p))
        distinctive += min(1.0, 0.5 * extra)
    else:
        # Nothing distinctive is supported, so JD-theme breadth is the only
        # honest proxy. A generic posting is not penalised for being generic.
        distinctive = {0: 1.0, 1: 2.0}.get(min(themes_covered, 2), 3.0)

    # -- 20%: strength / relevance of the evidence chosen ----------------
    strength = 0.8 if evidence_sources >= 1 else 0.0
    if evidence_sources >= 2:
        strength += 0.6
    if supported:
        if priority_uses_own_source(letter, supported[0]):
            strength += 0.6
    elif themes_covered >= 2:
        strength += 0.6

    # -- 10%: company / role specificity ---------------------------------
    specificity = 0.0
    head = (company or "").split("(")[0].strip().lower()
    if head and head in (letter or "").lower():
        specificity += 0.4
    if job_title and normalize_title_phrase(job_title) in normalize_title_phrase(letter or ""):
        specificity += 0.3
    if named_terms_covered >= 1:
        specificity += 0.3

    # -- 10%: writing quality / concision --------------------------------
    writing = 1.0
    writing -= 0.3 * sum(1 for kind in kinds
                         if kind in ("structure", "format", "length", "style", "generic"))
    if not 140 <= words <= 300:
        writing -= 0.3
    writing = max(0.0, writing)

    # Anything not already charged above (an unclassified warning, notably the
    # relevance warning itself) still costs something small.
    other = sum(1 for kind in kinds if kind not in _GROUNDING_KINDS
                and kind not in ("structure", "format", "length", "style", "generic",
                                 "relevance"))
    total = factual + distinctive + strength + specificity + writing - 0.2 * other
    return round(max(0.0, min(10.0, total)), 1)


# The old eight-field compact report was replaced by the Groq audit layer's
# ten-field report (audit_report / render_assessment_txt further down). The
# deterministic Cover Letter Score above survives it unchanged.
# ============================== cover-letter distinctive priorities
#
# Every letter this pipeline produced was true, source-scoped and validated,
# and several still led with the wrong evidence: a forward-deployed posting got
# a backend paragraph while the supported client-delivery evidence sat unused.
# Truth is necessary but not sufficient - the letter must also SELECT the
# evidence that makes this role different from every other SWE posting.
#
# Nothing here invents, rewrites or re-ranks evidence. It reads the
# requirements and source capsules the pipeline already built, names which
# distinctive signals the posting carries, and points at the strongest capsule
# that already supports each one. A signal with no supporting capsule stays
# uncovered; it is never padded with a forced paragraph.


@dataclass(frozen=True)
class DistinctiveTheme:
    """One way a posting can differ from a generic software-engineering role.

    `jd_cues` are deliberately multi-word: "customer training" is distinctive,
    "customers" is not. Generic craft terms (Python, Git, REST, communication)
    are absent on purpose - they describe every posting, so they can never
    identify what is distinctive about one.
    """

    key: str
    label: str
    jd_cues: tuple[str, ...]
    evidence_cues: tuple[str, ...]
    letter_cues: tuple[str, ...]


DISTINCTIVE_THEMES = (
    DistinctiveTheme(
        "customer_facing", "customer-facing delivery",
        jd_cues=("forward deployed", "forward-deployed", "customer facing",
                 "customer-facing", "client facing", "client-facing",
                 "customer requirements", "client requirements", "customer training",
                 "customer workshops", "customer demonstrations", "demonstrations",
                 "proof of technology", "proofs of technology", "proofs-of-technology",
                 "proof-of-technology", "proof of concept", "customer success",
                 "solution engineer", "solutions engineer", "pre-sales", "presales",
                 "consulting", "consultant", "onsite with customers",
                 "on-site with customers", "customer deployments", "trusted advisor"),
        evidence_cues=("client requirements", "client/stakeholder meetings",
                       "requirements gathering", "technical specifications",
                       "prototypes to validate scope", "end-to-end delivery",
                       "stakeholder collaboration", "clients on UI/UX",
                       "product collaboration", "scope decomposition"),
        letter_cues=("client", "customer", "stakeholder", "requirements gathering",
                     "gathering requirements", "prototype", "prototyping",
                     "demonstration", "demo", "training", "end to end", "end-to-end"),
    ),
    DistinctiveTheme(
        "ai_native", "AI-native development",
        jd_cues=("ai-native", "ai native", "claude code", "cursor", "copilot",
                 "ai coding agent", "ai coding agents", "coding agents", "ai agents",
                 "agentic", "llm-powered", "llm powered", "large language model",
                 "generative ai", "prompt engineering", "ai-assisted",
                 "ai assisted development", "ai tooling", "ai-first"),
        evidence_cues=("github copilot", "cursor", "claude code",
                       "ai-assisted code generation", "test-driven iteration",
                       "repository-level software development", "llm output as untrusted",
                       "prompt engineering", "llm orchestration", "llm integration",
                       "hallucination prevention", "structured output validation"),
        letter_cues=("claude code", "cursor", "copilot", "ai-assisted", "ai assisted",
                     "coding agent", "llm", "large language model", "prompt"),
    ),
    DistinctiveTheme(
        "healthcare", "healthcare / clinical software",
        jd_cues=("healthcare", "health care", "health system", "health systems",
                 "clinical", "clinician", "clinicians", "patient", "patients",
                 "hospital", "hospitals", "electronic health record", "ehr",
                 "medical errors", "medical record", "medical records",
                 "medical center", "medical centers", "care delivery",
                 "quality of care", "telemedicine", "telehealth"),
        evidence_cues=("healthcare modules", "patient management", "consent",
                       "telemedicine", "webrtc", "on-site nurses", "remote doctors",
                       "color-blindness diagnostic", "ishihara",
                       "clinical sensitivity", "iot baby"),
        letter_cues=("healthcare", "patient", "clinical", "telemedicine", "nurse",
                     "doctor", "diagnostic", "consent", "care"),
    ),
    DistinctiveTheme(
        "low_level_systems", "low-level systems work",
        jd_cues=("operating system", "operating systems", "kernel", "systems programming",
                 "low-level", "low level", "memory management", "memory safety",
                 "concurrency", "concurrent", "multithreaded", "multithreading",
                 "thread synchronization", "embedded", "device driver", "drivers",
                 "compilers", "distributed systems internals", "real-time systems"),
        evidence_cues=("pintos", "kernel", "x86", "thread synchronization",
                       "process lifecycle", "system-call", "system call",
                       "concurrency", "memory-fault", "file systems",
                       "parent-child synchronization", "user programs"),
        letter_cues=("pintos", "kernel", "operating system", "concurrency",
                     "synchronization", "system call", "system-call", "memory",
                     "thread", "low-level", "low level"),
    ),
    DistinctiveTheme(
        "ml_data", "machine-learning / data-heavy work",
        jd_cues=("machine learning", "deep learning", "neural network", "neural networks",
                 "model training", "model inference", "data science", "data scientist",
                 "pytorch", "tensorflow", "scikit-learn", "feature engineering",
                 "recommendation system", "recommender", "data pipeline",
                 "data pipelines", "etl", "big data", "statistical analysis",
                 "predictive model", "predictive models", "graph neural"),
        evidence_cues=("pytorch", "pytorch geometric", "graphsage", "graph neural",
                       "focal loss", "scikit-learn", "catboost", "librosa",
                       "etl workflow", "pandas", "numpy", "class imbalance",
                       "ieee-cis"),
        letter_cues=("machine learning", "model", "pytorch", "graphsage",
                     "graph neural", "catboost", "scikit-learn", "etl", "dataset",
                     "classifier", "classification", "precision", "recall"),
    ),
    DistinctiveTheme(
        "cloud_infra", "cloud infrastructure and deployment",
        jd_cues=("aws", "amazon web services", "ec2", "rds", "terraform",
                 "infrastructure as code", "cloud infrastructure", "cloud computing",
                 "provisioning", "devops", "site reliability", "observability",
                 "containerization", "docker compose", "vpc"),
        evidence_cues=("ec2", "rds", "vpc networking", "security groups",
                       "cloud resource provisioning", "docker compose",
                       "containerized", "reproducible execution",
                       "staging", "deployment environments"),
        letter_cues=("aws", "ec2", "rds", "vpc", "docker", "provisioning",
                     "infrastructure", "deployment", "container"),
    ),
    DistinctiveTheme(
        "realtime", "real-time and streaming systems",
        jd_cues=("real-time", "real time", "streaming", "websocket", "websockets",
                 "low latency", "low-latency", "event-driven", "pub/sub",
                 "message queue", "high throughput", "sensor data", "telemetry"),
        evidence_cues=("websocket", "redis", "django channels", "daphne",
                       "broadcast latency", "iot sensor", "live monitoring",
                       "concurrent users", "locust"),
        letter_cues=("real-time", "real time", "websocket", "redis", "latency",
                     "streaming", "broadcast", "concurrent users"),
    ),
)

# Cues that are too common to make a posting distinctive. Kept explicit so a
# future cue list cannot quietly reintroduce them.
GENERIC_CUES = frozenset({
    "python", "java", "javascript", "typescript", "git", "rest", "rest api",
    "rest apis", "sql", "communication", "teamwork", "agile", "scrum",
    "problem solving", "debugging", "testing", "object oriented programming",
    "computer science", "software engineering", "full stack", "backend",
    "frontend",
})

_PRIORITY_MIN_WEIGHT = 2
_PRIORITY_LIMIT = 3


def _cue_present(haystack: str, cue: str) -> bool:
    """Whole-phrase, case-insensitive containment.

    Word boundaries matter: "cursor" must not match "precursor", and "ehr"
    must not match "there". Hyphens and spaces inside a cue are allowed to
    vary because postings punctuate "customer-facing" both ways.
    """
    pattern = r"[\s\-/]+".join(re.escape(part) for part in re.split(r"[\s\-]+", cue))
    return re.search(r"(?<![a-z0-9])" + pattern + r"(?![a-z0-9])", haystack) is not None


def _theme_matches(theme: DistinctiveTheme, jd_lower: str, title_lower: str,
                   requirement_text: dict[str, str]) -> tuple[int, list[str], str]:
    """(weight, matched cues, owning requirement headline).

    Three tiers, most authoritative first: a cue in the JOB TITLE is what the
    role is called, a cue an extracted REQUIREMENT names is what the posting
    asked for, and a cue only in the prose is context.
    """
    weight = 0
    matched: list[str] = []
    owner = ""
    for cue in theme.jd_cues:
        if cue in GENERIC_CUES:
            continue
        in_title = bool(title_lower) and _cue_present(title_lower, cue)
        in_requirement = next((headline for headline, text in requirement_text.items()
                               if _cue_present(text, cue)), "")
        in_body = _cue_present(jd_lower, cue)
        if not in_title and not in_requirement and not in_body:
            continue
        matched.append(cue)
        weight += 3 if in_title else (2 if in_requirement else 1)
        if in_requirement and not owner:
            owner = in_requirement
    return weight, matched, owner


def _source_display(label: str, master: MasterFacts | None) -> str:
    """A capsule label the model can read: 'role:SWE' means nothing to it."""
    if label == "general":
        return "general evidence (skills bank)"
    if label.startswith("project:") and master is not None:
        identifier = label.split(":", 1)[1]
        for project in master.projects:
            if project.project_id == identifier:
                return project.name.split("(")[0].strip()
    if label.startswith("role:") and master is not None:
        prefix = label.split(":", 1)[1]
        for title, mapped in role_bullet_prefixes(master).items():
            if mapped == prefix:
                return title
    return label


def _best_source(theme: DistinctiveTheme, capsules: dict[str, str],
                 on_resume: frozenset) -> tuple[str, list[str]]:
    """The capsule that already supports this theme most strongly.

    Ties break toward evidence the shipped resume actually carries, so a
    letter is never pushed to lead with a project the reader cannot see.
    """
    ranked: list[tuple[int, int, int, str, list[str]]] = []
    for label, text in capsules.items():
        lowered = text.lower()
        hits = [cue for cue in theme.evidence_cues if _cue_present(lowered, cue)]
        if not hits:
            continue
        shipped = 0 if (label in on_resume or not label.startswith("project:")) else 1
        kind = {"role": 0, "general": 1}.get(label.split(":", 1)[0], 2)
        ranked.append((shipped, -len(hits), kind, label, hits))
    if not ranked:
        return "", []
    # Evidence the reader can see outranks a stronger match they cannot: a
    # project left off this resume is a weaker story than a shipped role.
    ranked.sort(key=lambda row: (row[0], row[1], row[2], row[3]))
    return ranked[0][3], ranked[0][4]


def letter_priorities(*, jd_text: str, job_title: str = "", requirements=(),
                      capsules: dict[str, str] | None = None,
                      master: MasterFacts | None = None, on_resume=(),
                      limit: int = _PRIORITY_LIMIT) -> list[dict]:
    """The 2-3 signals that define this role, each with its strongest evidence.

    Read-only: requirement extraction, capsule construction and project
    selection all happen before this and are untouched by it.
    """
    jd_lower = (jd_text or "").lower()
    requirement_text: dict[str, str] = {}
    for requirement in requirements or []:
        headline = getattr(requirement, "headline", "") or ""
        if not headline:
            continue
        terms = " ".join(getattr(requirement, "terms", ()) or ())
        requirement_text[headline] = f"{headline} {terms}".lower()

    capsules = capsules or {}
    shipped = frozenset(f"project:{identifier}" for identifier in on_resume)
    found: list[dict] = []
    for theme in DISTINCTIVE_THEMES:
        weight, matched, owner = _theme_matches(theme, jd_lower, (job_title or "").lower(),
                                                requirement_text)
        if weight < _PRIORITY_MIN_WEIGHT:
            continue
        source, evidence_terms = _best_source(theme, capsules, shipped)
        found.append({
            "theme": theme.key,
            "label": theme.label,
            "requirement": owner or f"the posting names {matched[0]!r}",
            "weight": weight,
            "jd_cues": matched[:6],
            "source": source,
            "source_name": _source_display(source, master) if source else "",
            "evidence_terms": evidence_terms[:4],
            "supported": bool(source),
        })
    found.sort(key=lambda row: (-row["weight"], row["label"]))
    return found[:limit]


def supported_priorities(priorities) -> list[dict]:
    return [p for p in (priorities or []) if p.get("supported")]


_ECHO_STOPWORDS = frozenset({"the", "and", "with", "for", "from", "into", "that",
                             "this", "their", "using", "used"})


def _term_echoed(letter_lower: str, term: str) -> bool:
    """Did the letter reuse this evidence term, allowing for normal rewording?

    Capsule wording and letter wording legitimately differ: the capsule says
    "client/stakeholder meetings" and "requirements gathering" where a letter
    says "client meetings" and "gathering requirements". Matching on word
    stems, and requiring two of them for a multi-word term, recognises the
    same evidence without accepting a single incidental word as proof.
    """
    tokens = [t for t in re.findall(r"[a-z][a-z0-9+#]{3,}", term.lower())
              if t not in _ECHO_STOPWORDS]
    if not tokens:
        return _cue_present(letter_lower, term)
    hits = sum(1 for token in tokens
               if re.search(r"(?<![a-z0-9])" + re.escape(token[:5]) + r"[a-z0-9]*",
                            letter_lower))
    return hits >= (1 if len(tokens) == 1 else 2)


def priority_addressed(letter: str, priority: dict) -> bool:
    """Does the letter meaningfully speak to this priority?

    Deliberately generous: the theme's own vocabulary OR the chosen source's
    evidence terms both count. Missing a JD keyword is not a defect; ignoring
    the role's defining, supported signal is.
    """
    lowered = (letter or "").lower()
    theme = next((t for t in DISTINCTIVE_THEMES if t.key == priority.get("theme")), None)
    if theme and any(_cue_present(lowered, cue) for cue in theme.letter_cues if cue):
        return True
    return priority_uses_own_source(letter, priority)


def priority_uses_own_source(letter: str, priority: dict) -> bool:
    """The letter reached for the specific evidence that was pointed at."""
    lowered = (letter or "").lower()
    terms = [t for t in (priority.get("evidence_terms") or ()) if t]
    return any(_term_echoed(lowered, term) for term in terms)


def render_letter_priorities(priorities) -> str:
    """The prompt block. Empty when the posting carries nothing distinctive."""
    supported = supported_priorities(priorities)
    if not supported:
        return ""
    lines = ["ROLE-DISTINCTIVE PRIORITIES (what makes THIS role different):"]
    for index, priority in enumerate(supported, start=1):
        lines.append(f"  {index}. {priority['label']} - {priority['requirement']}")
        lines.append(f"     strongest supported evidence: {priority['source_name']} "
                     f"({', '.join(priority['evidence_terms'])})")
    lines.append("")
    lines.append("At least one substantive paragraph must directly address priority 1 using "
                 "that evidence. Do not keyword-stuff and do not try to mention every "
                 "priority: two strong evidence stories beat five shallow ones. This "
                 "changes WHICH supported evidence you choose, never what you may claim.")
    return "\n".join(lines)


def letter_relevance(letter: str, priorities) -> list[Problem]:
    """Warn when a true letter still led with the wrong evidence.

    Warning, never error: a grounded letter is a usable artifact. The bounded
    retry path treats this as a repair reason while attempts remain, and the
    compact Cover Letter Score reflects it either way.
    """
    supported = supported_priorities(priorities)
    if not supported or not (letter or "").strip():
        return []
    top = supported[0]
    if priority_addressed(letter, top):
        return []
    return [Problem("relevance", "warning",
                    f"the letter is grounded but does not use the strongest supported "
                    f"evidence for the role's primary distinctive requirement: "
                    f"{top['label']} (available evidence: {top['source_name']} - "
                    f"{', '.join(top['evidence_terms'])})")]


# ============================== post-run Groq audit layer
#
# Three independent Groq calls run AFTER the resume and cover letter are
# final. Nothing they return may re-enter generation: the pipeline reads these
# results only to write assessment.txt and the audit block of strategy.json.
# Every validator here is defensive - a missing or malformed field degrades to
# UNKNOWN rather than failing the run, because the audit is advisory.

AUDIT_VERDICTS = ("PASS", "REVIEW")
CONFIDENCE_LEVELS = ("LOW", "MEDIUM", "HIGH")
CALLBACK_LIKELIHOOD = ("LOW", "MEDIUM", "HIGH", "UNKNOWN")
TERNARY = ("YES", "NO", "UNKNOWN")

EXPERIENCE_AUDIT_COMPONENTS = ("signal_interpretation", "policy_choice",
                               "evidence_relevance")
PROJECT_SELECTION_COMPONENTS = ("jd_relevance", "best_available_chosen",
                                "complementary_coverage", "ranking_and_allocation")
PROJECT_BULLET_COMPONENTS = ("jd_relevance", "technical_specificity",
                             "evidence_fidelity", "impact_ownership", "non_redundancy")

_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_LONG_DATE = re.compile(r"^([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})$")


def iso_date(value) -> str:
    """YYYY-MM-DD, or UNKNOWN. The report contract admits nothing else.

    The JD extractor reports whatever the posting wrote ("March 4, 2026"), so
    that spelling is normalized here rather than leaking into the artifact.
    """
    text = str(value or "").strip()
    if not text or text.upper() == "UNKNOWN":
        return "UNKNOWN"
    if _ISO_DATE.match(text):
        return text
    long_form = _LONG_DATE.match(text)
    if long_form:
        month = _MONTH_NUMBERS.get(long_form.group(1).lower())
        if month:
            day, year = int(long_form.group(2)), int(long_form.group(3))
            if 1 <= day <= 31:
                return f"{year:04d}-{month:02d}-{day:02d}"
    return "UNKNOWN"


def _score(value, *, low: float = 0.0, high: float = 10.0) -> float | None:
    """A 0-10 score, or None when the provider sent something unusable."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if number != number or not low <= number <= high:     # NaN or out of range
        return None
    return round(number, 1)


def _enum(value, allowed: tuple[str, ...]) -> str:
    text = str(value or "").strip().upper().replace(" ", "_")
    return text if text in allowed else ""


def _components(raw, names: tuple[str, ...]) -> dict[str, float]:
    """Only in-range component scores survive; absent ones are simply absent."""
    source = raw if isinstance(raw, dict) else {}
    found: dict[str, float] = {}
    for name in names:
        score = _score(source.get(name))
        if score is not None:
            found[name] = score
    return found


def _observations(raw, limit: int = 6) -> list[str]:
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        text = re.sub(r"\s+", " ", str(item or "")).strip()
        if text:
            out.append(text[:240])
    return out[:limit]


def validate_experience_audit(data: dict) -> tuple[dict, list[Problem]]:
    """Groq call #1. Returns a normalized audit plus advisory problems.

    The audit may never restate the Experience decision - it only judges it -
    so no field here can carry a bullet id list that the pipeline would act on.
    """
    problems: list[Problem] = []
    source = data if isinstance(data, dict) else {}
    overall = _score(source.get("experience_selection_score"))
    components = _components(source.get("components"), EXPERIENCE_AUDIT_COMPONENTS)
    verdict = _enum(source.get("verdict"), AUDIT_VERDICTS)
    if overall is None:
        problems.append(Problem("audit_schema", "warning",
                                "experience audit returned no usable "
                                "experience_selection_score"))
    missing = [name for name in EXPERIENCE_AUDIT_COMPONENTS if name not in components]
    if missing:
        problems.append(Problem("audit_schema", "warning",
                                "experience audit omitted component score(s): "
                                + ", ".join(missing)))
    if not verdict:
        # A score without a verdict is still usable: derive it, and say so.
        verdict = "PASS" if (overall or 0.0) >= 7.0 else "REVIEW"
        problems.append(Problem("audit_schema", "warning",
                                f"experience audit sent no PASS/REVIEW verdict; derived "
                                f"{verdict} from the score"))
    better = source.get("better_permitted_choice_exists")
    audit = {
        "experience_selection_score": overall,
        "components": components,
        "verdict": verdict,
        "missed_signals": _observations(source.get("missed_signals")),
        "better_permitted_choice_exists": bool(better) if isinstance(better, bool) else None,
        "explanation": _observations(source.get("explanation"), limit=1),
    }
    return audit, problems


def validate_application_audit(data: dict) -> tuple[dict, list[Problem]]:
    """Groq call #2's added scores. The verdict architecture is untouched.

    fit_score, the immutable verdicts and their finalization are validated by
    validate_assessment/validate_final_assessment exactly as before; only the
    new audit scores are read here.
    """
    problems: list[Problem] = []
    source = data if isinstance(data, dict) else {}

    tailoring = source.get("tailoring_quality")
    tailoring_score = _score((tailoring or {}).get("score")
                             if isinstance(tailoring, dict) else None)

    selection = source.get("project_selection") if isinstance(
        source.get("project_selection"), dict) else {}
    bullets = source.get("project_bullets") if isinstance(
        source.get("project_bullets"), dict) else {}

    experience = source.get("experience_selection") if isinstance(
        source.get("experience_selection"), dict) else {}

    audit = {
        "experience_selection_score": _score(experience.get("score")),
        "experience_selection_notes": _observations(experience.get("notes"), limit=2),
        "resume_tailoring_score": tailoring_score,
        "project_selection_score": _score(selection.get("score")),
        "project_selection_components": _components(selection.get("components"),
                                                    PROJECT_SELECTION_COMPONENTS),
        "project_selection_notes": _observations(selection.get("notes")),
        "project_bullet_score": _score(bullets.get("score")),
        "project_bullet_components": _components(bullets.get("components"),
                                                 PROJECT_BULLET_COMPONENTS),
        "project_bullet_notes": _observations(bullets.get("notes")),
        "callback_likelihood": _enum(source.get("callback_likelihood"),
                                     CALLBACK_LIKELIHOOD) or "UNKNOWN",
    }
    for label, key in (("resume tailoring", "resume_tailoring_score"),
                       ("experience selection", "experience_selection_score"),
                       ("project selection", "project_selection_score"),
                       ("project bullet quality", "project_bullet_score")):
        if audit[key] is None:
            problems.append(Problem("audit_schema", "warning",
                                    f"application audit returned no usable {label} score"))
    return audit, problems


# ---- Groq call #3: company research ---------------------------------------
#
# Two rules are enforced in Python because a model will cheerfully over-read
# weak evidence: E-Verify participation is not STEM OPT support, and a
# job-specific restriction outranks anything found at company level.

_EVERIFY_ONLY = re.compile(r"e-?verify", re.IGNORECASE)
_STEM_OPT_EVIDENCE = re.compile(
    r"STEM\s*OPT|24-month extension|I-983|training plan for STEM OPT", re.IGNORECASE)
_H1B_ONLY = re.compile(r"\bH-?1B\b|\bLCA\b|\bPERM\b", re.IGNORECASE)


# What a piece of evidence is actually about. An unrelated posting is context,
# never a company-wide policy: the live C3 run found a DIFFERENT C3 job that
# mentioned sponsorship and nearly reported it as the company's position.
SOURCE_SCOPES = ("this_job", "company_policy", "unrelated_posting", "context")


def _scope(value) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "job": "this_job", "this_posting": "this_job", "audited_job": "this_job",
        "exact_job": "this_job", "job_specific": "this_job",
        "company": "company_policy", "official": "company_policy",
        "company_wide": "company_policy", "policy": "company_policy",
        "other_posting": "unrelated_posting", "another_job": "unrelated_posting",
        "different_job": "unrelated_posting", "other_job": "unrelated_posting",
    }
    text = aliases.get(text, text)
    return text if text in SOURCE_SCOPES else "context"


def _sources(raw, limit: int = 8) -> list[dict]:
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        entry = {
            "title": re.sub(r"\s+", " ", str(item.get("title") or "")).strip()[:160],
            "url": str(item.get("url") or "").strip()[:400],
            "evidence": re.sub(r"\s+", " ", str(item.get("evidence") or "")).strip()[:300],
            "scope": _scope(item.get("scope")),
        }
        if entry["title"] or entry["url"]:
            out.append(entry)
    return out[:limit]


def authoritative_sources(sources: list[dict]) -> list[dict]:
    """Only evidence that may settle a company-level verdict.

    This posting itself and an official company-wide policy qualify. An
    unrelated posting and general context do not, however clearly they seem to
    state something: one team's requisition is not the employer's policy.
    """
    return [s for s in (sources or []) if s.get("scope") in ("this_job", "company_policy")]


def everify_only(sources: list[dict]) -> bool:
    """True when the ONLY sponsorship-adjacent evidence is E-Verify."""
    text = " ".join(f"{s.get('title','')} {s.get('evidence','')}"
                     for s in authoritative_sources(sources) or sources)
    if not _EVERIFY_ONLY.search(text):
        return False
    return not _STEM_OPT_EVIDENCE.search(text)


def validate_company_research(data: dict, *, jd_text: str = "",
                              jd_posted: str = "UNKNOWN") -> tuple[dict, list[Problem]]:
    """Groq call #3, with the two over-reading rules enforced deterministically."""
    problems: list[Problem] = []
    source = data if isinstance(data, dict) else {}
    sources = _sources(source.get("sources"))

    sponsorship = _enum(source.get("company_visa_sponsorship"), TERNARY) or "UNKNOWN"
    sponsor_confidence = _enum(source.get("company_visa_confidence"),
                               CONFIDENCE_LEVELS) or "LOW"
    stem = _enum(source.get("company_stem_opt_support"), TERNARY) or "UNKNOWN"
    stem_confidence = _enum(source.get("company_stem_opt_confidence"),
                            CONFIDENCE_LEVELS) or "LOW"
    overrides: list[str] = []

    # E-Verify participation says nothing about STEM OPT support.
    if stem == "YES" and everify_only(sources):
        stem, stem_confidence = "UNKNOWN", "LOW"
        overrides.append("STEM OPT support downgraded to UNKNOWN: the only supporting "
                         "evidence is E-Verify participation, which does not establish it")

    # SOURCE SCOPING. A company-level YES/NO needs evidence that is actually
    # about the company or about this exact job. An unrelated requisition is
    # supporting context and can establish nothing on its own.
    authoritative = authoritative_sources(sources)
    if sponsorship in ("YES", "NO") and sources and not authoritative:
        overrides.append(
            f"company visa sponsorship reset to UNKNOWN: the only evidence is "
            f"{sources[0].get('scope', 'context')}-scoped, which cannot establish a "
            f"company-wide position")
        sponsorship, sponsor_confidence = "UNKNOWN", "LOW"
    if stem in ("YES", "NO") and sources and not authoritative:
        overrides.append("company STEM OPT support reset to UNKNOWN: no company-level or "
                         "job-specific evidence was cited")
        stem, stem_confidence = "UNKNOWN", "LOW"

    # Historical H-1B filings are not a current sponsorship policy.
    evidence_text = " ".join(f"{s.get('title','')} {s.get('evidence','')}"
                             for s in (authoritative or sources))
    if (sponsorship == "YES" and sponsor_confidence == "HIGH"
            and _H1B_ONLY.search(evidence_text)
            and not re.search(r"sponsor", evidence_text, re.IGNORECASE)):
        sponsor_confidence = "MEDIUM"
        overrides.append("visa sponsorship confidence lowered to MEDIUM: historical H-1B "
                         "filings alone do not prove current sponsorship policy")

    # The posting is an official company source about THIS role, so explicit
    # language in it settles what company-level research left open.
    jd_verdict = _explicit(jd_text or "", _SPONSOR_YES, _SPONSOR_NO)
    if jd_verdict == "YES" and sponsorship == "UNKNOWN":
        overrides.append("this posting explicitly offers sponsorship, which settles the "
                         "company-level UNKNOWN for this role")
        sponsorship, sponsor_confidence = "YES", "HIGH"
    jd_stem = _explicit(jd_text or "", _STEM_OPT_YES, _STEM_OPT_NO)
    if jd_stem != "UNKNOWN" and stem == "UNKNOWN":
        overrides.append(f"this posting states STEM OPT support explicitly ({jd_stem})")
        stem, stem_confidence = jd_stem, "HIGH"

    # An exact job-specific restriction outranks a company-level finding, both
    # when the researcher found it and when the posting itself states it.
    job_specific_no = any(
        s.get("scope") == "this_job"
        and re.search(r"\b(?:no|not|without|unable)\b[^.]{0,40}\bsponsor",
                      s.get("evidence", ""), re.IGNORECASE)
        for s in sources)
    if job_specific_no and sponsorship != "NO":
        overrides.append(f"an exact job-specific restriction overrides the company-level "
                         f"finding ({sponsorship}) for this application")
        sponsorship, sponsor_confidence = "NO", "HIGH"

    # A job-specific restriction always wins over a company-level finding.
    if jd_verdict == "NO" and sponsorship != "NO":
        overrides.append(f"this posting states sponsorship is unavailable, which overrides "
                         f"the company-level finding ({sponsorship})")
        sponsorship, sponsor_confidence = "NO", "HIGH"
    if jd_verdict == "NO" and stem == "YES":
        overrides.append("this posting states sponsorship is unavailable, so company-level "
                         "STEM OPT support cannot be reported as YES")
        stem, stem_confidence = "UNKNOWN", "LOW"

    posted = iso_date(source.get("job_posted"))
    posted_confidence = _enum(source.get("job_posted_confidence"),
                              CONFIDENCE_LEVELS) or "LOW"
    # A trustworthy date already in the posting is preferred over a searched one.
    from_jd = iso_date(jd_posted)
    if from_jd != "UNKNOWN":
        if posted != from_jd:
            overrides.append(f"job_posted taken from the posting itself ({from_jd}) "
                             f"rather than the researched value ({posted})")
        posted, posted_confidence = from_jd, "HIGH"

    checked_at = str(source.get("checked_at") or "").strip()[:40]
    if not checked_at:
        problems.append(Problem("audit_schema", "warning",
                                "company research returned no checked_at timestamp"))
    if posted == "UNKNOWN" and not sources:
        problems.append(Problem("audit_schema", "warning",
                                "company research returned no sources"))

    research = {
        "company_visa_sponsorship": sponsorship,
        "company_visa_confidence": sponsor_confidence,
        "company_stem_opt_support": stem,
        "company_stem_opt_confidence": stem_confidence,
        "job_posted": posted,
        "job_posted_confidence": posted_confidence,
        "checked_at": checked_at,
        "sources": sources,
        "deterministic_overrides": overrides,
    }
    return research, problems


# ---- the report -----------------------------------------------------------

def _fmt(score) -> str:
    return f"{score:.1f} / 10" if isinstance(score, (int, float)) else "UNKNOWN"


def audit_report(*, experience_audit=None, application_audit=None, research=None,
                 letter_score=None) -> dict:
    """assessment.txt's ten fields, assembled from whatever succeeded.

    Every field degrades to UNKNOWN independently, so one failed Groq call
    never blanks the other two.
    """
    experience_audit = experience_audit or {}
    application_audit = application_audit or {}
    research = research or {}
    return {
        "Resume Tailoring Score": _fmt(application_audit.get("resume_tailoring_score")),
        # Scored by the final assessment as one of its components: there is no
        # separate Groq experience-audit call any more.
        "Experience Selection Score":
            _fmt(application_audit.get("experience_selection_score")
                 if application_audit.get("experience_selection_score") is not None
                 else experience_audit.get("experience_selection_score")),
        "Project Selection Score": _fmt(application_audit.get("project_selection_score")),
        "Project Bullet Quality Score": _fmt(application_audit.get("project_bullet_score")),
        "Cover Letter Score": _fmt(letter_score),
        "Fit Match Score": _fmt(application_audit.get("fit_score")),
        "Callback Likelihood": application_audit.get("callback_likelihood") or "UNKNOWN",
        "Company Visa Sponsorship": research.get("company_visa_sponsorship") or "UNKNOWN",
        "Company STEM OPT Support": research.get("company_stem_opt_support") or "UNKNOWN",
        "Job Posted": research.get("job_posted") or "UNKNOWN",
    }


def render_assessment_txt(report: dict) -> str:
    """The assessment.txt body: only the agreed fields, in order."""
    return "\n".join(f"{field}: {report.get(field, 'UNKNOWN')}"
                     for field in ASSESSMENT_FIELDS) + "\n"


def render_assessment_terminal(report: dict, *, company: str, job_title: str) -> list[str]:
    """The compact terminal block, printed after every completed job."""
    head = " | ".join(part for part in (company or "Unknown company",
                                        job_title or "Unknown role") if part)
    return [f"[ASSESSMENT] {head}"] + [
        f"  {field}: {report.get(field, 'UNKNOWN')}" for field in ASSESSMENT_FIELDS]


# ============================== semantic JD signals (Gemini recall layer)
#
# Deterministic classification is keyword-driven and therefore has recall
# gaps: a posting that says "participate in design and code reviews" and
# "establish engineering best practices" is code-quality-heavy without using
# any term the classifier matches. Gemini reads the same JD it already reads
# for project selection and reports which of OUR OWN taxonomy signals it sees,
# with short quotes from the posting.
#
# Gemini helps UNDERSTAND the posting. It never selects an Experience bullet,
# never writes Experience wording and can never turn a deterministic TRUE into
# a FALSE. Every positive signal it claims must be grounded in the JD text or
# it is discarded.

# Our taxonomy, mapped onto what the deterministic policy already understands.
# `family` is the role-family label the Experience swap table is keyed by;
# `field` is the Signals attribute a boolean signal maps onto. Nothing here is
# dynamic: an unknown signal name is rejected, never learned.
SEMANTIC_SIGNALS: dict[str, dict] = {
    "backend": {"family": "Backend SWE"},
    "agentic_ai": {"family": "Agentic AI / AI Agent Engineer",
                   "field": "jd_emphasizes_agentic_ai"},
    "ai_ml_adjacent": {"family": "AI/ML-adjacent SWE"},
    "data_engineering": {"family": "Data Engineer"},
    "cloud_infrastructure": {"family": "Cloud / Infrastructure SWE"},
    "systems_low_level": {"family": "Systems / Low-Level SWE"},
    "code_quality_collaboration_heavy": {"field": "code_quality"},
    "healthcare": {"family": "HealthTech / Healthcare SWE", "field": "healthcare"},
}

SEMANTIC_SIGNAL_NAMES = tuple(SEMANTIC_SIGNALS)

# Evidence must be a real phrase from the posting, not a paraphrase and not a
# single word that would match almost anything.
_EVIDENCE_MIN_WORDS = 3
_EVIDENCE_MAX_PER_SIGNAL = 3


def _flatten(text: str) -> str:
    """Case- and whitespace-insensitive form for grounding comparisons."""
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def evidence_is_grounded(snippet: str, jd_flat: str) -> bool:
    """Is this snippet actually present in the posting?

    Compared on the flattened form so punctuation, capitalisation and line
    wrapping in the JD cannot cause a false rejection, while an invented
    phrase still fails.
    """
    flat = _flatten(snippet)
    if len(flat.split()) < _EVIDENCE_MIN_WORDS:
        return False
    return flat in jd_flat


def validate_semantic_signals(raw, jd_text: str) -> tuple[dict, list[Problem]]:
    """Keep only taxonomy signals whose positive claims the JD supports.

    Returns `{signal_name: bool}` for every signal the model spoke about, plus
    advisory problems. Never raises: a malformed payload yields an empty map
    and generation continues on the deterministic signals alone.
    """
    problems: list[Problem] = []
    if raw in (None, {}, []):
        return {}, problems
    if not isinstance(raw, dict):
        return {}, [Problem("semantic_signals", "warning",
                            f"semantic_jd_signals must be an object, got "
                            f"{type(raw).__name__}")]

    jd_flat = _flatten(jd_text)
    validated: dict[str, bool] = {}
    for name, payload in raw.items():
        key = str(name or "").strip().lower()
        if key not in SEMANTIC_SIGNALS:
            problems.append(Problem("semantic_signals", "warning",
                                    f"unknown semantic signal {name!r} ignored; the "
                                    f"taxonomy is fixed"))
            continue
        if not isinstance(payload, dict):
            problems.append(Problem("semantic_signals", "warning",
                                    f"{key}: expected an object with present/evidence"))
            continue
        present = payload.get("present")
        if not isinstance(present, bool):
            problems.append(Problem("semantic_signals", "warning",
                                    f"{key}: present must be true or false, got "
                                    f"{present!r}"))
            continue
        if not present:
            # A negative claim needs no evidence, and can never remove a
            # deterministic positive later in the merge.
            validated[key] = False
            continue

        snippets = payload.get("evidence")
        if isinstance(snippets, str):
            snippets = [snippets]
        if not isinstance(snippets, list) or not snippets:
            problems.append(Problem("semantic_signals", "warning",
                                    f"{key}: present=true requires 1-"
                                    f"{_EVIDENCE_MAX_PER_SIGNAL} evidence snippet(s)"))
            validated[key] = False
            continue
        if not all(isinstance(item, str) for item in snippets):
            problems.append(Problem("semantic_signals", "warning",
                                    f"{key}: evidence must be strings"))
            validated[key] = False
            continue

        grounded = [s for s in snippets[:_EVIDENCE_MAX_PER_SIGNAL]
                    if evidence_is_grounded(s, jd_flat)]
        if not grounded:
            problems.append(Problem("semantic_signals", "warning",
                                    f"{key}: rejected, none of its evidence appears in the "
                                    f"job description ({[s[:60] for s in snippets[:2]]})"))
            validated[key] = False
            continue
        if len(grounded) < len(snippets[:_EVIDENCE_MAX_PER_SIGNAL]):
            problems.append(Problem("semantic_signals", "warning",
                                    f"{key}: kept on {len(grounded)} grounded snippet(s); "
                                    f"the rest were not found in the posting"))
        validated[key] = True
    return validated, problems


def deterministic_signal_map(signals) -> dict[str, bool]:
    """What the deterministic classifier already believes, in taxonomy terms."""
    family = getattr(signals, "role_family", "")
    scores = getattr(signals, "domain_scores", {}) or {}
    out: dict[str, bool] = {}
    for name, spec in SEMANTIC_SIGNALS.items():
        field = spec.get("field")
        if field:
            out[name] = bool(getattr(signals, field, False))
            continue
        # A domain signal is "detected" when it is the chosen family or the
        # keyword scorer found it at all.
        out[name] = bool(spec.get("family") == family or scores.get(spec.get("family")))
    return out


def merge_semantic_signals(signals, validated: dict[str, bool]
                           ) -> tuple[dict[str, bool], dict]:
    """OR the two sources together. Gemini may only ADD.

    Returns the merged taxonomy map plus the field/family overrides Python
    should apply before running its own Experience policy. Deterministic
    positives are never withdrawn, so the worst a wrong Gemini answer can do
    is leave the deterministic decision exactly as it was.
    """
    deterministic = deterministic_signal_map(signals)
    merged = dict(deterministic)
    added: list[str] = []
    for name, value in (validated or {}).items():
        if name not in SEMANTIC_SIGNALS:
            continue
        if value and not merged.get(name):
            merged[name] = True
            added.append(name)

    overrides: dict = {"fields": {}, "role_family": None, "added": added}
    for name in added:
        field = SEMANTIC_SIGNALS[name].get("field")
        if field:
            overrides["fields"][field] = True
    # The role family is only PROMOTED, and only when deterministic
    # classification found nothing to go on. A detected family is never
    # replaced by a model's opinion.
    if not getattr(signals, "domain_scores", None):
        families = [SEMANTIC_SIGNALS[n].get("family") for n in added
                    if SEMANTIC_SIGNALS[n].get("family")]
        if len(families) == 1:
            overrides["role_family"] = families[0]
    return merged, overrides


def apply_semantic_overrides(signals, overrides: dict):
    """A copy of Signals carrying the merged view, for the Experience policy.

    Returns the original object untouched when nothing was added, so the
    common case is provably identical to the pre-change behaviour.
    """
    fields = dict((overrides or {}).get("fields") or {})
    family = (overrides or {}).get("role_family")
    if not fields and not family:
        return signals
    changes = dict(fields)
    if family:
        changes["role_family"] = family
    if "healthcare" in changes:
        changes.setdefault("healthcare_reason",
                           "recovered from the posting by validated semantic signal")
    if "code_quality" in changes:
        changes.setdefault("code_quality_reason",
                           "recovered from the posting by validated semantic signal")
    merged = dataclasses.replace(signals, **changes)
    # `_grad_ok` carries the section-order decision and is not an init field.
    merged._grad_ok = getattr(signals, "_grad_ok", False)
    return merged
