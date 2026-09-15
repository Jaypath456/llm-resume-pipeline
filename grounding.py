"""Factual grounding: metric semantics, project-bullet grounding, letter checks.

Every generated sentence has to be traceable to evidence that already exists in
`Master_Resume_Context.md`. This module is deliberately literal about numbers:
a metric's identity is its value, its unit, its threshold structure and whether
it is approximate. There is no numeric tolerance anywhere - `3+` is not `3`,
`approximately 90%` is not `90%`, and `sub-500ms` is not `500ms`.
"""
from __future__ import annotations

import copy
import re
import unicodedata
from dataclasses import dataclass, field

from resume_engine import MasterFacts, Project, fold_term

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
                          company: str | None = None) -> list[Problem]:
    """Style, metric-source, ownership and unsupported-claim gate.

    `jd_text` and `company` enable employer-attributed facts. Without them the
    master stays the only authority, which is the stricter behaviour.
    """
    problems = check_style(text, banned)
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
        require(bool((match.get("evidence") or "").strip()),
                f"strong_matches[{index}] has no evidence")
        require(match.get("source") in EVIDENCE_SOURCES,
                f"strong_matches[{index}].source must be one of {EVIDENCE_SOURCES}, got "
                f"{match.get('source')!r}")

    for index, match in enumerate(data.get("partial_matches") or [], start=1):
        if not isinstance(match, dict):
            require(False, f"partial_matches[{index}] is not an object")
            continue
        claim("partial_matches", index, match.get("requirement_id"))
        require(bool((match.get("evidence") or "").strip()),
                f"partial_matches[{index}] has no evidence")
        require(bool((match.get("limitation") or "").strip()),
                f"partial_matches[{index}] has no limitation")

    for index, gap in enumerate(data.get("gaps") or [], start=1):
        if not isinstance(gap, dict):
            require(False, f"gaps[{index}] is not an object")
            continue
        claim("gaps", index, gap.get("requirement_id"))
        require(gap.get("importance") in GAP_IMPORTANCE,
                f"gaps[{index}].importance must be one of {GAP_IMPORTANCE}, got "
                f"{gap.get('importance')!r}")
        require(gap.get("status") in GAP_STATUS,
                f"gaps[{index}].status must be one of {GAP_STATUS}, got {gap.get('status')!r}")
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
        require(bool((entry.get("reason") or "").strip()),
                f"manual_review[{index}] has no reason")

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
