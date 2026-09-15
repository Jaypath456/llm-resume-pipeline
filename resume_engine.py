"""Inputs, policy and every deterministic resume decision.

Three authoritative inputs, and this module is where they meet:

  * `Master_Resume_Context.md`  - FACTUAL source of truth (what may be claimed).
  * `Resume_analysis.xlsx`      - TAILORING POLICY source (how to tailor).
  * `Resume_Template.tex`       - PRESENTATION, and the home of the approved
                                  static Experience/Education content.

Division of authority:

  * Professional Experience is assembled deterministically from the template's
    approved base bullets plus the exact swaps listed in the spreadsheet. No
    model ever rewrites it.
  * Technical Skills start from the spreadsheet's mandatory block and are fitted
    to exactly 8 rendered lines using the spreadsheet's overflow priorities.
  * Project selection and project bullet wording are the LLM's job; this module
    only prepares the inputs, fixes the allocation and renders the LaTeX.

Where the spreadsheet and older Python policy disagree, the spreadsheet wins.
Where the spreadsheet is silent (section ordering, JD signal definitions), the
Master's PART B is used and that is logged.
"""
from __future__ import annotations

import os
import re
import dataclasses
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

# ============================================================== configuration

PROJECT_ROOT = Path(__file__).resolve().parent
MASTER_PATH = PROJECT_ROOT / "Master_Resume_Context.md"
POLICY_PATH = PROJECT_ROOT / "Resume_analysis.xlsx"
TEMPLATE_PATH = PROJECT_ROOT / "Resume_Template.tex"
JOBS_DIR = PROJECT_ROOT / "jobs"
SYNTHETIC_JOBS_DIR = PROJECT_ROOT / "jobs_synthetic"
OUTPUT_DIR = PROJECT_ROOT / "output"
SMOKE_OUTPUT_DIR = OUTPUT_DIR / "_smoke_tests"
PROCESSED_CSV = PROJECT_ROOT / "processed_jobs.csv"
PROCESSED_INDEX = PROJECT_ROOT / ".processed_index.json"

# Model ids are env-overridable so a retired model never needs a code change.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

GEMINI_MAX_ACCOUNTS = 8
RATE_LIMIT_MAX_ATTEMPTS = 3
SERVER_ERROR_MAX_ATTEMPTS = 2
GROQ_MAX_ATTEMPTS = 3
GROQ_MAX_OUTPUT_TOKENS = 4096
BACKOFF_BASE_SECONDS = 2.0
BACKOFF_MAX_SECONDS = 20.0

# Iteration ceilings. Each skills iteration is one pdflatex compile plus one
# real measurement, so the budget has to allow a full add/remove sequence.
SKILLS_MAX_ITERATIONS = 24
LAYOUT_MAX_ROUNDS = 6

PLACEHOLDER_EXPERIENCE = "[AUTO:EXPERIENCE]"
PLACEHOLDER_SKILLS = "[AUTO:SKILLS]"
PLACEHOLDER_PROJECTS = "[AUTO:PROJECTS]"
SECTIONS_MARKER = "%% <<SECTIONS>>"


@dataclass(frozen=True)
class Credentials:
    """Credential availability, expressed without exposing key material."""

    gemini_accounts: tuple[int, ...] = ()
    has_groq: bool = False
    has_bullsai: bool = False
    _gemini_keys: dict[int, str] = field(default_factory=dict, repr=False)
    _groq_key: str | None = field(default=None, repr=False)

    def gemini_key(self, account: int) -> str:
        return self._gemini_keys[account]

    @property
    def groq_key(self) -> str | None:
        return self._groq_key

    @property
    def secrets(self) -> list[str]:
        values = list(self._gemini_keys.values())
        if self._groq_key:
            values.append(self._groq_key)
        return values

    def describe(self) -> str:
        accounts = ", ".join(f"#{n}" for n in self.gemini_accounts) or "none"
        return (f"gemini accounts: {accounts} | groq: "
                f"{'configured' if self.has_groq else 'absent'} | bullsai: "
                f"{'key present (intentionally unwired)' if self.has_bullsai else 'absent'}")


def load_credentials(env: dict[str, str] | None = None) -> Credentials:
    if env is None:
        try:
            from dotenv import dotenv_values
            env = {**dotenv_values(PROJECT_ROOT / ".env"), **os.environ}
        except Exception:                                    # pragma: no cover
            env = dict(os.environ)
    keys = {}
    for n in range(1, GEMINI_MAX_ACCOUNTS + 1):
        value = (env.get(f"GEMINI_API_KEY_{n}") or "").strip()
        if value:
            keys[n] = value
    groq = (env.get("GROQ_API_KEY") or "").strip() or None
    return Credentials(
        gemini_accounts=tuple(sorted(keys)), has_groq=groq is not None,
        has_bullsai=bool((env.get("BULLSAI_API_KEY") or "").strip()),
        _gemini_keys=keys, _groq_key=groq,
    )


class PolicyError(RuntimeError):
    """An input is missing something the pipeline cannot safely invent."""


# ================================================================ LaTeX text

_ESCAPES = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
            "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
            "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}

FORBIDDEN_LAYOUT_HACKS = (r"\newline", r"\linebreak", r"\hspace*", r"\vspace*{-",
                          r"\phantom", r"\textcolor{white}", "\u00a0", r"\\[")


def escape_latex(text: str) -> str:
    return "".join(_ESCAPES.get(ch, ch) for ch in text)


_TYPESET_RULES = (
    (re.compile(r"\bat least\s+(?=\d)"), r"\geq"),
    (re.compile(r"\bapproximately\s+(?=\d)"), r"\sim"),
    (re.compile(r"(?<![\w])~(?=\d)"), r"\sim"),
)
_STASH = "\x00{}\x00"


def typeset(text: str) -> str:
    """Turn generated plain text into template-quality LaTeX.

    Reproduces the conventions the approved resume already uses: `$\\sim$` for
    approximations, `$\\geq$` for thresholds, `--` for numeric ranges and
    `` `` '' `` for quoted identifiers. Used for GENERATED content only -
    Experience and Education come from the template verbatim.
    """
    macros: list[str] = []

    def stash(latex: str) -> str:
        macros.append(latex)
        return _STASH.format(len(macros) - 1)

    text = re.sub(r"`([^`]+)`", lambda m: stash("``" + escape_latex(m.group(1)) + "''"), text)
    text = text.replace("\u2014", stash("--")).replace("\u2013", stash("--"))
    for pattern, macro in _TYPESET_RULES:
        text = pattern.sub(lambda _m, macro=macro: stash(f"${macro}$"), text)
    text = re.sub(r"(?<=\d)\s*-\s*(?=\d)", lambda _m: stash("--"), text)
    text = re.sub(r"\(([^()]*?)\s-\s([^()]*?)\)",
                  lambda m: "(" + m.group(1) + stash(" $-$ ") + m.group(2) + ")", text)
    out = escape_latex(text)
    for index, macro in enumerate(macros):
        out = out.replace(_STASH.format(index), macro)
    return out


def itemize(items: list[str], indent: str = "\t") -> str:
    body = [indent + r"\begin{itemize}", indent + "\t" + r"\setlength{\itemsep}{0pt}"]
    body += [indent + "\t" + r"\item " + item for item in items]
    body.append(indent + r"\end{itemize}")
    return "\n".join(body)


_MATH_SPAN = re.compile(r"\$[^$]*\$")
_CMD_ARG = re.compile(r"\\[a-zA-Z]+\*?\{([^{}]*)\}")
_BARE_CMD = re.compile(r"\\[a-zA-Z]+\*?")

# Glyph variants pdftotext may emit for the same source construct.
_GLYPH_CANON = {
    "\u223c": "~", "\u2248": "~", "\u2212": "-", "\u2011": "-", "\u2013": "-",
    "\u2014": "-", "\u2265": ">=", "\u2264": "<=", "\u201c": '"', "\u201d": '"',
    "\u2018": "'", "\u2019": "'", "\u00a0": " ", "\u202f": " ", "\u2009": " ",
}


def latex_to_plain(latex: str) -> str:
    """Reduce a LaTeX `\\item` to the plain words a PDF extractor would show."""
    text = latex.strip()
    text = re.sub(r"^\\item\b", "", text)
    text = text.replace(r"$\sim$", "~").replace(r"$\geq$", ">=").replace(r"$\leq$", "<=")
    text = text.replace(r"$-$", "-").replace(r"$|$", "|")
    text = _MATH_SPAN.sub(" ", text)
    text = text.replace(r"\\", " ")
    for _ in range(3):
        text = _CMD_ARG.sub(r" \1 ", text)
    text = _BARE_CMD.sub(" ", text)
    text = text.replace("``", '"').replace("''", '"').replace("--", "-")
    for escaped, plain in ((r"\%", "%"), (r"\&", "&"), (r"\_", "_"), (r"\#", "#"),
                           (r"\{", "{"), (r"\}", "}"), (r"\$", "$")):
        text = text.replace(escaped, plain)
    return normalize_plain(text)


def normalize_plain(text: str) -> str:
    """Whitespace- and glyph-canonical form for comparing rendered text.

    Only presentation is normalized: wrapping, spacing and the glyph a
    construct rendered as. Every word must still match exactly.
    """
    for variant, canon in _GLYPH_CANON.items():
        text = text.replace(variant, canon)
    # pdftotext separates a math glyph from its number ("~ 90%", ">= 80%").
    # That is how the glyph rendered, not a different word, so it normalizes
    # away. Nothing else about the wording is touched.
    text = re.sub(r"(?<=[~<>=])\s+(?=[\d.])", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ================================================================== template


@dataclass
class Template:
    """The one resume template, split into reorderable content blocks."""

    path: Path
    skeleton: str                       # document with the sections marker
    blocks: dict[str, str]              # block name -> LaTeX

    def render(self, order: list[str], *, experience: str, skills: str, projects: str) -> str:
        missing = [name for name in order if name not in self.blocks]
        if missing:
            raise PolicyError(f"template has no block(s) named {missing}")
        unused = [name for name in self.blocks if name not in order]
        if unused:
            raise PolicyError(f"template block(s) missing from the section order: {unused}")

        filled = {
            "Experience": self.blocks["Experience"].replace(PLACEHOLDER_EXPERIENCE, experience),
            "Education": self.blocks["Education"],
            "Technical Skills": self.blocks["Technical Skills"].replace(PLACEHOLDER_SKILLS, skills),
            "Academic Projects": self.blocks["Academic Projects"].replace(PLACEHOLDER_PROJECTS, projects),
        }
        body = "\n".join(filled[name].rstrip() + "\n" for name in order)
        out = self.skeleton.replace(SECTIONS_MARKER, body)
        for leftover in (PLACEHOLDER_EXPERIENCE, PLACEHOLDER_SKILLS, PLACEHOLDER_PROJECTS,
                         SECTIONS_MARKER):
            if leftover in out:
                raise PolicyError(f"placeholder survived rendering: {leftover}")
        return out


_BLOCK_RE = re.compile(r"^%% ==== BLOCK: (?P<name>[^=]+?) ====\s*$(?P<body>.*?)^%% ==== END BLOCK ====\s*$",
                       re.MULTILINE | re.DOTALL)


def load_template(path: Path = TEMPLATE_PATH) -> Template:
    raw = path.read_text(encoding="utf-8")
    blocks: dict[str, str] = {}
    for match in _BLOCK_RE.finditer(raw):
        name = match.group("name").strip()
        if name in blocks:
            raise PolicyError(f"template defines block {name!r} twice")
        blocks[name] = match.group("body").strip("\n")
    expected = {"Experience", "Education", "Technical Skills", "Academic Projects"}
    if set(blocks) != expected:
        raise PolicyError(f"template blocks {sorted(blocks)} != expected {sorted(expected)}")
    prose = [needle for needle in (r"\item Led", r"\item Built", r"\item Engineered",
                                   r"\item Integrated", r"\item Architected")
             if needle in blocks["Experience"]]
    if prose:
        raise PolicyError(
            f"{path.name} still contains Professional Experience prose ({prose}). The template "
            f"is presentation only; approved wording lives in {POLICY_PATH.name}.")
    if PLACEHOLDER_EXPERIENCE not in blocks["Experience"]:
        raise PolicyError(f"template Experience block must contain {PLACEHOLDER_EXPERIENCE}")
    if SECTIONS_MARKER not in raw:
        raise PolicyError(f"template is missing the {SECTIONS_MARKER} marker")
    skeleton = _BLOCK_RE.sub("", raw)
    return Template(path=path, skeleton=skeleton, blocks=blocks)


# ============================================================= master (facts)


def fold(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def fold_term(value: str) -> str:
    """Like `fold` but keeps `+`/`#`, so `C++` never collapses onto `C`."""
    return re.sub(r"[^a-z0-9+#]+", "", value.lower())


@dataclass(frozen=True)
class Project:
    project_id: str
    name: str
    context: str
    date: str
    tech: tuple[str, ...]
    evidence: tuple[str, ...]
    tags: tuple[str, ...]

    @property
    def blob(self) -> str:
        return " ".join((self.name, *self.tech, *self.evidence, *self.tags))


@dataclass(frozen=True)
class Skill:
    name: str
    category: str                       # Master tier category
    tier: str                           # "A" or "B"
    condition: str | None = None        # e.g. jd_emphasizes_agentic_ai


@dataclass(frozen=True)
class Capsule:
    """One unit of evidence: an Experience bullet, a project or an evidence-bank entry."""

    capsule_id: str
    text: str
    tags: tuple[str, ...] = ()


@dataclass
class SectionOrder:
    experience_first: tuple[str, ...]
    education_first: tuple[str, ...]
    threshold: int
    weights: dict[str, int]

    def order_for(self, mode: str) -> list[str]:
        return list(self.education_first if mode == "education_first" else self.experience_first)


@dataclass
class MasterFacts:
    raw: str
    projects: tuple[Project, ...]
    skills: tuple[Skill, ...]
    unsupported_terms: tuple[str, ...]
    standing_facts: tuple[str, ...]
    experience_text: str                # all approved Experience wording + evidence
    normalization: dict[str, str]
    evidence_blobs: dict[str, str]
    skill_evidence: dict[str, tuple[str, ...]]
    capsules: tuple[Capsule, ...]
    section_order: SectionOrder
    signal_docs: str
    proper_nouns: tuple[str, ...]
    tier_categories: tuple[str, ...]
    # `raw` minus the do-not-claim section, so a forbidden name listed
    # there can never be read back as positive evidence.
    evidence_corpus: str = ""

    def project(self, project_id: str) -> Project | None:
        return next((p for p in self.projects if p.project_id == project_id), None)

    def skill(self, name: str) -> Skill | None:
        target = fold(name)
        return next((s for s in self.skills if fold(s.name) == target), None)

    def canonical_skill(self, name: str) -> str | None:
        """Map an arbitrary spelling onto a canonical claimable skill name.

        `fold` drops `+` and `#`, which would collapse "C#" and "C++" onto the
        claimable skill "C". A language whose name differs only by those
        characters is a DIFFERENT language, so an alias lookup is refused when
        it would cross that boundary.
        """
        # This check must precede the `skill()` lookup, which compares on
        # `fold` and would otherwise match "C#" against the claimable "C".
        probe = name.strip()
        if probe and re.search(r"[+#]", probe):
            exact = next((s for s in self.skills
                          if fold_term(s.name) == fold_term(probe)), None)
            return exact.name if exact else None
        direct = self.skill(name)
        if direct:
            return direct.name
        alias = self.normalization.get(fold(name))
        if alias:
            hit = self.skill(alias)
            return hit.name if hit else alias
        return None

    def supported_anywhere(self, phrase: str) -> bool:
        """Whether a CLAIMABLE section of the master evidences this phrase.

        Searches the claimable corpus only. "Kubernetes" appears in the master
        solely inside Explicitly Unsupported, and that must never read as
        support.
        """
        needle = phrase.strip()
        if not needle:
            return False
        haystack = (self.evidence_corpus or self.raw).lower()
        pattern = r"(?<![a-z0-9+#])" + re.escape(needle.lower()) + r"(?![a-z0-9+#])"
        return bool(re.search(pattern, haystack))

    def supported_by_experience(self, phrase: str) -> bool:
        pattern = r"(?<![a-z0-9+#])" + re.escape(phrase.strip().lower()) + r"(?![a-z0-9+#])"
        return bool(re.search(pattern, self.experience_text.lower()))

    def unsupported_heads(self) -> list[str]:
        """Every do-not-claim name, including spaced alternatives.

        The separator is " / " with spaces: splitting on a bare "/" turned
        "CI/CD" into "CI" and stopped it matching a JD that asks for CI/CD.
        """
        heads: list[str] = []
        for term in self.unsupported_terms:
            body = term.split(" unless ")[0].split(" beyond ")[0].strip()
            for alternative in body.split(" / "):
                head = alternative.strip()
                if head and len(head.split()) <= 3 and head not in heads:
                    heads.append(head)
        return heads


@dataclass
class _Node:
    level: int
    title: str
    lines: list[str] = field(default_factory=list)
    children: list["_Node"] = field(default_factory=list)

    def find(self, pattern: str, level: int | None = None) -> "_Node | None":
        rx = re.compile(pattern, re.IGNORECASE)
        for child in self.children:
            if (level is None or child.level == level) and rx.search(child.title):
                return child
            found = child.find(pattern, level)
            if found:
                return found
        return None

    def body(self) -> str:
        return "\n".join(self.lines)


def _tree(text: str) -> _Node:
    root = _Node(0, "<root>")
    stack = [root]
    for line in text.splitlines():
        match = re.match(r"^(#{1,6})\s+(.*?)\s*$", line)
        if match:
            node = _Node(len(match.group(1)), match.group(2))
            while stack and stack[-1].level >= node.level:
                stack.pop()
            stack[-1].children.append(node)
            stack.append(node)
        else:
            stack[-1].lines.append(line)
    return root


def _bullets(lines: list[str]) -> list[str]:
    return [l.strip()[2:].strip() for l in lines if l.strip().startswith("- ")]


def _labelled(lines: list[str], label: str) -> str | None:
    for line in lines:
        if line.strip().lower().startswith(f"{label.lower()}:"):
            return line.split(":", 1)[1].strip().rstrip("\\").strip()
    return None


def _tags(lines: list[str]) -> tuple[str, ...]:
    collected, capturing = [], False
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("relevance tags:"):
            capturing = True
            rest = stripped.split(":", 1)[1].strip()
            if rest:
                collected.append(rest)
            continue
        if capturing:
            if not stripped or stripped.startswith(("#", "**", "-", "---")):
                capturing = False
                continue
            collected.append(stripped)
    return tuple(t.strip() for t in " ".join(collected).split(",") if t.strip())


def _sub_block(lines: list[str], label: str) -> list[str]:
    out, capturing = [], False
    for line in lines:
        stripped = line.strip()
        if re.match(rf"^{re.escape(label)}\s*:\s*$", stripped, re.IGNORECASE):
            capturing = True
            continue
        if capturing:
            if stripped.startswith("- "):
                out.append(stripped[2:].strip())
            elif stripped and not stripped.startswith("-"):
                break
    return out


_CONDITION_RE = re.compile(r"\[conditional:\s*([a-z0-9_]+)\s*\]", re.IGNORECASE)
_CATEGORY_RE = re.compile(r"\[category:\s*([^\]]+?)\s*\]", re.IGNORECASE)
_ARROW_RE = re.compile(r"`([^`]+)`(?:\s*(?:/|and)\s*`([^`]+)`)?\s*(?:->|normalize[sd]? to)\s*"
                       r"(?:canonical resume label\s*)?`([^`]+)`")


# Only these top-level sections state what the candidate has actually done.
# Everything else in the master is policy prose, and policy prose names
# forbidden technologies ("JD says Kubernetes -> Docker evidence may be
# relevant"), which must never read back as evidence.
_CLAIMABLE_HEADINGS = ("contact", "education", "professional experience",
                       "technical skills", "academic projects", "general evidence bank",
                       "certifications", "standing facts")


def _claimable_corpus(raw: str) -> str:
    """The factual sections of the master, with policy and guards excluded."""
    kept: list[str] = []
    inside = False
    for line in raw.splitlines():
        heading = re.match(r"^#\s+(?!#)(.+)$", line)
        if heading:
            title = heading.group(1).strip().lower()
            inside = (any(needle in title for needle in _CLAIMABLE_HEADINGS)
                      and "unsupported" not in title
                      and not re.match(r"^\d+\.", title))
        if inside:
            kept.append(line)
    return "\n".join(kept)


def load_master(path: Path = MASTER_PATH) -> MasterFacts:
    """Parse PART A facts plus the two policies the spreadsheet does not define."""
    raw = path.read_text(encoding="utf-8")
    root = _tree(raw)

    # --- projects -----------------------------------------------------------
    projects_node = root.find(r"^Academic Projects$", 1)
    if projects_node is None:
        raise PolicyError("Master has no '# Academic Projects' section")
    projects: list[Project] = []
    for node in projects_node.children:
        if node.level != 2:
            continue
        pid = re.sub(r"[`*]", "", _labelled(node.lines, "ID") or "").strip()
        if not pid:
            raise PolicyError(f"project {node.title!r} has no ID")
        evidence: tuple[str, ...] = ()
        tags: tuple[str, ...] = ()
        for sub in node.children:
            if "supported evidence" in sub.title.lower():
                evidence = tuple(_bullets(sub.lines))
                tags = _tags(sub.lines)
        projects.append(Project(
            project_id=pid, name=node.title.strip(),
            context=re.sub(r"[`*]", "", _labelled(node.lines, "Context") or "").strip(),
            date=re.sub(r"[`*]", "", _labelled(node.lines, "Date") or "").strip(),
            tech=tuple(t.strip() for t in (_labelled(node.lines, "Tech") or "").split(",") if t.strip()),
            evidence=evidence, tags=tags,
        ))

    # --- claimable skill pool ----------------------------------------------
    pool_node = root.find(r"Technical Skills — Claimable Pool", 1)
    if pool_node is None:
        raise PolicyError("Master has no claimable skill pool")
    skills: list[Skill] = []
    tier_categories: list[str] = []
    for tier_node in pool_node.children:
        tier_match = re.search(r"Evidence Tier ([AB])", tier_node.title, re.IGNORECASE)
        if not tier_match:
            continue
        tier = tier_match.group(1).upper()
        for sub in tier_node.children:
            if sub.level != 3 or re.search(r"evidence support|guidance|note", sub.title, re.IGNORECASE):
                continue
            default_category = re.sub(r"\s*Tier [AB] skills?$", "", sub.title, flags=re.IGNORECASE).strip()
            for line in sub.lines:
                stripped = line.strip()
                if not stripped or stripped.startswith(("-", "#", "`", ">")):
                    continue
                body = _CATEGORY_RE.sub("", _CONDITION_RE.sub("", stripped)).strip()
                if not body or body.endswith((".", ":", ";")) or len(body.split()) > 6:
                    continue
                condition = _CONDITION_RE.search(stripped)
                category = _CATEGORY_RE.search(stripped)
                skill_category = category.group(1).strip() if category else default_category
                if tier == "A" and skill_category not in tier_categories:
                    tier_categories.append(skill_category)
                skills.append(Skill(name=body, category=skill_category, tier=tier,
                                    condition=condition.group(1).lower() if condition else None))
    if not skills:
        raise PolicyError("Master claimable skill pool is empty")

    # --- normalization aliases ---------------------------------------------
    normalization: dict[str, str] = {}
    guidance = pool_node.find(r"normalization guidance", 2)
    for rule in _bullets(guidance.lines if guidance else []):
        arrow = _ARROW_RE.search(re.sub(r"[*]", "", rule))
        if arrow:
            canonical = arrow.group(3).strip()
            for alias in (arrow.group(1), arrow.group(2)):
                if alias:
                    normalization[fold(alias)] = canonical
    for skill in skills:
        normalization.setdefault(fold(skill.name), skill.name)
        bare = re.sub(r"\s*\([^)]*\)", "", skill.name).strip()
        if bare:
            normalization.setdefault(fold(bare), skill.name)
        if "/" in skill.name and "(" not in skill.name:
            for part in skill.name.split("/"):
                if part.strip():
                    normalization.setdefault(fold(part.strip()), skill.name)

    # --- experience capsules (facts, not resume wording) -------------------
    experience_node = root.find(r"^Professional Experience$", 1)
    if experience_node is None:
        raise PolicyError("Master has no '# Professional Experience' section")
    capsules: list[Capsule] = []
    experience_chunks: list[str] = []
    proper_nouns: set[str] = set()
    for role in experience_node.children:
        if role.level != 2:
            continue
        title, _, company = role.title.partition(" — ")
        proper_nouns.update({company.strip(), *[p.strip() for p in company.split(",")], title.strip()})
        for sub in role.children:
            lowered = sub.title.lower()
            if "resume bullet" in lowered:
                for bullet_id, text, evidence, tags in _parse_master_bullets(sub.lines):
                    blob = " ".join((text, *evidence, *tags))
                    capsules.append(Capsule(bullet_id, blob, tags))
                    experience_chunks.append(blob)
            elif "additional supported evidence" in lowered:
                blob = " ".join((*_bullets(sub.lines), *_tags(sub.lines)))
                capsules.append(Capsule(f"{title.strip()} (additional evidence)", blob, _tags(sub.lines)))
                experience_chunks.append(blob)

    for project in projects:
        capsules.append(Capsule(f"project:{project.project_id}", project.blob, project.tags))
        proper_nouns.add(project.name)

    bank_node = root.find(r"^General Evidence Bank$", 1)
    for entry in (bank_node.children if bank_node else []):
        if entry.level != 2:
            continue
        blob = " ".join((entry.title, *_sub_block(entry.lines, "Supported evidence"), *_tags(entry.lines)))
        capsules.append(Capsule(f"bank:{fold(entry.title)[:24]}", blob, _tags(entry.lines)))
    coursework = root.find(r"Coursework evidence supporting", 3)
    if coursework:
        capsules.append(Capsule("bank:coursework",
                                " ".join((coursework.title, *_bullets(coursework.lines),
                                          *_tags(coursework.lines))), _tags(coursework.lines)))

    # --- unsupported terms and standing facts ------------------------------
    unsupported_node = root.find(r"Explicitly Unsupported", 1)
    if unsupported_node is None:
        raise PolicyError("Master has no 'Explicitly Unsupported' section")
    unsupported, in_guards = [], False
    for line in unsupported_node.lines:
        stripped = line.strip()
        if stripped.lower().startswith("important upgrade guards"):
            in_guards = True
        elif stripped.startswith("- ") and not in_guards:
            unsupported.append(stripped[2:].strip())
    standing_node = root.find(r"Standing Facts for Cover Letters", 1)
    standing = tuple(_bullets(standing_node.lines)) if standing_node else ()

    # --- policies the spreadsheet does not define --------------------------
    policy_text = raw[raw.find("PART B"):]
    section_order = _parse_section_order(policy_text)
    signal_node = root.find(r"Signal definitions", 2)
    signal_docs = signal_node.body().strip() if signal_node else ""

    evidence_corpus = _claimable_corpus(raw)
    evidence_blobs = {c.capsule_id: c.text for c in capsules}
    skill_evidence: dict[str, tuple[str, ...]] = {}
    for skill in skills:
        needles = _skill_needles(skill.name)
        skill_evidence[skill.name] = tuple(
            cid for cid, blob in evidence_blobs.items() if _mentions(blob, needles)
        )

    return MasterFacts(
        raw=raw, projects=tuple(projects), skills=tuple(skills),
        unsupported_terms=tuple(unsupported), standing_facts=standing,
        experience_text=" ".join(experience_chunks), normalization=normalization,
        evidence_blobs=evidence_blobs, skill_evidence=skill_evidence,
        capsules=tuple(capsules), section_order=section_order, signal_docs=signal_docs,
        proper_nouns=tuple(sorted({n for n in proper_nouns if len(n) >= 3}, key=len, reverse=True)),
        tier_categories=tuple(tier_categories), evidence_corpus=evidence_corpus,
    )


_MASTER_BULLET_RE = re.compile(r"^\*\*([A-Z][A-Z0-9\-]*[A-Z0-9])(?:\s+[—-]\s+.*?)?\*\*\s*$")


def _parse_master_bullets(lines: list[str]):
    blocks, current = [], None
    for line in lines:
        match = _MASTER_BULLET_RE.match(line.strip())
        if match:
            if current:
                blocks.append(current)
            current = (match.group(1), [])
        elif current is not None:
            current[1].append(line)
    if current:
        blocks.append(current)
    for bullet_id, body in blocks:
        text_parts = []
        for line in body:
            stripped = line.strip()
            if not stripped:
                if text_parts:
                    break
                continue
            if re.match(r"^[A-Z][A-Za-z /]+:\s*$", stripped):
                break
            text_parts.append(stripped)
        yield (bullet_id, " ".join(text_parts).strip(),
               tuple(_sub_block(body, "Underlying evidence")), _tags(body))


# Spelled-out phrasings of a supported skill. Morphology/spelling only: each
# entry must denote the SAME concept as its skill, never a related technology.
_SKILL_PHRASINGS = {
    "oop": ("object oriented programming", "object-oriented programming",
            "object oriented design", "object-oriented design", "object oriented"),
}


def _skill_needles(name: str) -> tuple[str, ...]:
    forms = {name, re.sub(r"\s*\([^)]*\)", "", name).strip()}
    # Spelled-out phrasings of the SAME concept (OOP <-> object oriented
    # programming). Spelling only; never a different technology.
    forms.update(_SKILL_PHRASINGS.get(fold_term(name), ()))
    for piece in re.findall(r"\(([^)]*)\)", name):
        forms.update(p.strip() for p in re.split(r"[/,]", piece))
    if "/" in name and "(" not in name:
        forms.update(p.strip() for p in name.split("/"))
    for form in list(forms):
        if form.endswith("s") and len(form) > 2:
            forms.add(form[:-1])
        else:
            forms.add(form + "s")
    return tuple(sorted(f for f in forms if f))


def _mentions(blob: str, needles: tuple[str, ...]) -> bool:
    lowered = blob.lower()
    return any(re.search(r"(?<![a-z0-9+#])" + re.escape(n.lower()) + r"(?![a-z0-9+#])", lowered)
               for n in needles)


def _parse_section_order(policy_text: str) -> SectionOrder:
    """Section ordering comes from Master PART B; the spreadsheet defines none."""
    orders: dict[str, tuple[str, ...]] = {}
    for match in re.finditer(
            r"\*\*(?:DEFAULT|EDUCATION-FIRST)\s*\(`(?P<mode>[a-z_]+)`\):\*\*\s*\n(?P<items>(?:\s*\d+\.\s*.+\n)+)",
            policy_text):
        orders[match.group("mode")] = tuple(
            re.sub(r"^\s*\d+\.\s*", "", line).strip()
            for line in match.group("items").splitlines() if line.strip())
    for mode in ("experience_first", "education_first"):
        if mode not in orders:
            raise PolicyError(f"Master section 16 does not define the {mode!r} order")
    threshold = re.search(r"`education_first`\s*when the score is at least\s*(\d+)", policy_text)
    if not threshold:
        raise PolicyError("Master section 16 does not state the education-first threshold")
    weights: dict[str, int] = {}
    for line in policy_text.splitlines():
        row = re.match(r"^\|\s*`([a-z_]+)`[^|]*\|\s*\+?(\d+)\s*\|", line.strip())
        if row:
            weights[row.group(1)] = int(row.group(2))
    if not weights:
        raise PolicyError("Master section 16 states no signal weights")
    return SectionOrder(orders["experience_first"], orders["education_first"],
                        int(threshold.group(1)), weights)


# ======================================================== policy (spreadsheet)


@dataclass(frozen=True)
class Swap:
    """One approved Experience swap, keyed by exact base-bullet LaTeX."""

    block: str                  # "job_role" | "jd_signal"
    label: str                  # role family or JD signal as written in the sheet
    find_latex: str
    replace_latex: str | None   # None == "NO change"
    row: int

    @property
    def find_plain(self) -> str:
        return latex_to_plain(self.find_latex)

    @property
    def replace_plain(self) -> str | None:
        return latex_to_plain(self.replace_latex) if self.replace_latex else None


@dataclass(frozen=True)
class ExperienceEntry:
    """One approved Experience line, owned by its id rather than its wording."""

    bullet_id: str
    role: str
    latex: str

    @property
    def is_role_header(self) -> bool:
        return self.bullet_id.startswith("ROLE-")

    @property
    def plain(self) -> str:
        return latex_to_plain(self.latex)


@dataclass(frozen=True)
class SwapRule:
    """`source_id -> target_id` for one policy label. Wording-independent."""

    label: str
    source_id: str
    target_id: str
    applies_when: str          # "job_role" | "jd_signal"
    row: int

    @property
    def is_no_change(self) -> bool:
        return self.target_id.lower().replace(" ", "") in ("nochange", "-", "")


@dataclass
class Policy:
    """Everything the spreadsheet decides."""

    role_swaps: tuple[Swap, ...]
    signal_swaps: tuple[Swap, ...]
    mandatory: tuple[tuple[str, tuple[str, ...]], ...]      # (category, items)
    overflow_removals: tuple[tuple[str, str], ...]          # (category, item fragment)
    style_text: str
    banned_phrases: tuple[str, ...]
    rules: tuple[str, ...]
    skills_target_lines: int
    project_count: int
    bullet_allocation: tuple[int, ...]
    page_count: int
    tail_orphan_max: float
    tail_acceptable_max: float
    tail_ideal_max: float
    tail_healthy_min: float
    second_line_max_fill: float
    verb_max_uses: int
    data_fixes: tuple[str, ...] = ()
    # The spreadsheet owns approved Experience wording, keyed by stable id.
    experience_library: tuple[ExperienceEntry, ...] = ()
    alternate_library: tuple[ExperienceEntry, ...] = ()
    swap_rules: tuple[SwapRule, ...] = ()

    @property
    def total_project_bullets(self) -> int:
        return sum(self.bullet_allocation)

    @property
    def bullet_target_lines(self) -> int:
        """Template target: 14 rendered bullet lines across 7 bullets."""
        return 2

    def category_labels(self) -> list[str]:
        return [label for label, _ in self.mandatory]

    def removal_for(self, category: str) -> str | None:
        return next((frag for cat, frag in self.overflow_removals if cat == category), None)

    # ------------------------------------------------ experience libraries
    def base_entry(self, bullet_id: str) -> ExperienceEntry | None:
        return next((e for e in self.experience_library if e.bullet_id == bullet_id), None)

    def alternate_entry(self, alt_id: str) -> ExperienceEntry | None:
        return next((e for e in self.alternate_library if e.bullet_id == alt_id), None)

    def approved_text(self, bullet_id: str) -> str | None:
        entry = self.base_entry(bullet_id) or self.alternate_entry(bullet_id)
        return entry.latex if entry else None

    @property
    def experience_bullet_ids(self) -> list[str]:
        return [e.bullet_id for e in self.experience_library if not e.is_role_header]

    def rules_for(self, label: str, applies_when: str) -> list[SwapRule]:
        return [r for r in self.swap_rules
                if r.label == label and r.applies_when == applies_when and not r.is_no_change]


_WORD_NUMBERS = {"once": 1, "twice": 2, "thrice": 3, "three times": 3}


def verify_experience_ids(policy: Policy) -> list[str]:
    """Check the Experience id graph before anything expensive happens.

    Returns a list of problems; empty means the policy is internally sound.
    No silent fallback: the caller fails the run before any provider call.
    """
    problems: list[str] = []
    if not policy.experience_library:
        problems.append("EXPERIENCE LIBRARY block is missing or empty")
    if not policy.alternate_library:
        problems.append("EXPERIENCE ALTERNATES block is missing or empty")

    for name, entries in (("base", policy.experience_library),
                          ("alternate", policy.alternate_library)):
        seen: dict[str, int] = {}
        for entry in entries:
            seen[entry.bullet_id] = seen.get(entry.bullet_id, 0) + 1
            if not entry.latex.strip():
                problems.append(f"{name} bullet id={entry.bullet_id} has no approved text")
            if not entry.is_role_header and not entry.latex.lstrip().startswith(r"\item"):
                problems.append(f"{name} bullet id={entry.bullet_id} is not an \\item line")
        for bullet_id, count in seen.items():
            if count != 1:
                problems.append(f"duplicate {name} bullet id={bullet_id} appears {count} times")

    overlap = ({e.bullet_id for e in policy.experience_library}
               & {e.bullet_id for e in policy.alternate_library})
    for bullet_id in sorted(overlap):
        problems.append(f"id={bullet_id} is defined as both a base and an alternate bullet")

    for rule in policy.swap_rules:
        if rule.is_no_change:
            continue
        if policy.base_entry(rule.source_id) is None:
            problems.append(f"missing swap source bullet id={rule.source_id} "
                            f"(rule {rule.label!r}, sheet row {rule.row})")
        if policy.alternate_entry(rule.target_id) is None:
            problems.append(f"missing alternate bullet id={rule.target_id} "
                            f"(rule {rule.label!r}, sheet row {rule.row})")

    # Every role must contribute its bullets in a contiguous, ordered run.
    roles: list[str] = []
    for entry in policy.experience_library:
        if entry.is_role_header:
            if entry.role in roles:
                problems.append(f"role {entry.role!r} has more than one header row")
            roles.append(entry.role)
        elif not roles:
            problems.append(f"bullet id={entry.bullet_id} appears before any role header")
        elif entry.role != roles[-1]:
            problems.append(f"bullet id={entry.bullet_id} (role {entry.role!r}) is filed under "
                            f"role {roles[-1]!r}; bullets must follow their own role header")
    for role in roles:
        owned = [e for e in policy.experience_library
                 if e.role == role and not e.is_role_header]
        if not owned:
            problems.append(f"role {role!r} has a header but no approved bullets")
    return problems


def load_policy(path: Path = POLICY_PATH) -> Policy:
    """Read the CURRENT spreadsheet. Nothing here is defaulted silently."""
    try:
        import openpyxl
    except ImportError as error:                             # pragma: no cover
        raise PolicyError("openpyxl is required to read Resume_analysis.xlsx") from error

    sheet = openpyxl.load_workbook(path, data_only=True).worksheets[0]
    grid: dict[int, dict[str, str]] = {}
    for row in sheet.iter_rows():
        values = {cell.column_letter: cell.value for cell in row if cell.value is not None}
        if values:
            grid[row[0].row] = {k: (v if isinstance(v, str) else str(v)) for k, v in values.items()}

    def text(row: int, col: str) -> str:
        return (grid.get(row, {}).get(col) or "").strip()

    def find_row(col: str, needle: str) -> int:
        for row in sorted(grid):
            if needle.lower() in (grid[row].get(col) or "").lower():
                return row
        raise PolicyError(f"spreadsheet has no {col} cell containing {needle!r}")

    data_fixes: list[str] = []

    def clean_item(latex: str) -> str | None:
        """Normalize a spreadsheet `\\item` cell without changing its wording."""
        if not latex:
            return None
        value = re.sub(r"\s+", " ", latex.strip())
        if value.lower() == "no change":
            return None
        # An unescaped `%` starts a LaTeX comment and would silently delete the
        # rest of the bullet, changing the rendered metric. Escaping it is a
        # data fix, not a rewording, so it is applied and logged.
        fixed = re.sub(r"(?<!\\)%", r"\\%", value)
        if fixed != value:
            note = (f"escaped an unescaped '%' (LaTeX comment character) in a spreadsheet cell; "
                    f"without this the rendered bullet would silently lose everything after it: "
                    f"...{value[-44:]!r}")
            if note not in data_fixes:
                data_fixes.append(note)
        return fixed

    # --- swap tables --------------------------------------------------------
    def read_swaps(header_row: int, block: str) -> list[Swap]:
        swaps: list[Swap] = []
        row = header_row + 1
        while row in grid or (row + 1) in grid:
            label = text(row, "B")
            if not label:
                if row - header_row > 1:
                    break
                row += 1
                continue
            find_latex = clean_item(text(row, "C"))
            replace_latex = clean_item(text(row, "D"))
            if find_latex is None and text(row, "C").lower().startswith("no change"):
                swaps.append(Swap(block, label, "", None, row))
            elif find_latex:
                swaps.append(Swap(block, label, find_latex, replace_latex, row))
            row += 1
        return swaps

    def read_library(header_needle: str, label: str) -> list[ExperienceEntry]:
        try:
            header = find_row("B", header_needle)
        except PolicyError:
            return []
        entries: list[ExperienceEntry] = []
        row = header + 1
        while row in grid:
            first, second, third = text(row, "B"), text(row, "C"), text(row, "D")
            if not first:
                break
            if first in ("bullet_id", "alt_id"):
                row += 1
                continue
            if third:
                entries.append(ExperienceEntry(first, second, clean_item(third) or third))
            row += 1
        return entries

    experience_library = read_library("EXPERIENCE LIBRARY", "base")
    alternate_library = read_library("EXPERIENCE ALTERNATES", "alternate")

    swap_rules: list[SwapRule] = []
    try:
        mapping_header = find_row("B", "EXPERIENCE SWAP MAPPINGS")
    except PolicyError:
        mapping_header = None
    if mapping_header is not None:
        row = mapping_header + 1
        while row in grid:
            label, source, target = text(row, "B"), text(row, "C"), text(row, "D")
            applies = text(row, "E") or "jd_signal"
            if not label:
                break
            if label != "rule_label":
                swap_rules.append(SwapRule(label, source, target, applies, row))
            row += 1

    role_swaps = read_swaps(find_row("B", "Job role"), "job_role")
    signal_swaps = read_swaps(find_row("B", "JD signal"), "jd_signal")
    if not role_swaps or not signal_swaps:
        raise PolicyError("spreadsheet swap tables are empty")

    # --- mandatory Technical Skills ----------------------------------------
    mandatory: list[tuple[str, tuple[str, ...]]] = []
    for row in sorted(grid):
        cell = text(row, "B")
        if cell.startswith(r"\item \textbf{"):
            label = cell.split("textbf{", 1)[1].split(":}", 1)[0].replace("\\&", "&").replace("\\", "").strip()
            items = _split_skill_items(cell.split(":}", 1)[1])
            mandatory.append((label, items))
    if not mandatory:
        raise PolicyError("spreadsheet defines no mandatory Technical Skills block")

    # --- overflow removals --------------------------------------------------
    removal_header = find_row("B", "things to remove")
    removals: list[tuple[str, str]] = []
    row = removal_header + 1
    while row in grid:
        category, fragment = text(row, "C"), text(row, "D")
        if category and fragment and category.lower() != "from":
            removals.append((category, fragment))
        elif not category and not fragment:
            break
        row += 1

    # --- writing style ------------------------------------------------------
    style_start = find_row("B", "RESUME WRITING STYLE")
    rules_start = find_row("B", "Rules:")
    style_lines = [text(row, "B") for row in range(style_start, rules_start) if text(row, "B")]
    style_text = "\n".join(style_lines)
    banned = [line.lstrip("- ").strip() for line in style_lines
              if line.startswith("- ") and len(line) < 70]

    # --- rules --------------------------------------------------------------
    rules = [text(row, "B") for row in sorted(grid) if row > rules_start and text(row, "B")]
    if not rules:
        raise PolicyError("spreadsheet states no rules")
    blob = "\n".join(rules)

    def number(pattern: str, what: str) -> str:
        match = re.search(pattern, blob, re.IGNORECASE)
        if not match:
            raise PolicyError(f"spreadsheet rules do not state {what}")
        return match.group(1)

    verb_words = number(r"no action verb may appear more than (\w+)", "the action-verb limit").lower()
    allocation = tuple(int(n) for n in number(r"(\d+(?:/\d+)+)-style", "the bullet allocation").split("/"))
    policy = Policy(
        role_swaps=tuple(role_swaps), signal_swaps=tuple(signal_swaps),
        experience_library=tuple(experience_library),
        alternate_library=tuple(alternate_library), swap_rules=tuple(swap_rules),
        mandatory=tuple(mandatory), overflow_removals=tuple(removals),
        style_text=style_text, banned_phrases=tuple(banned), rules=tuple(rules),
        skills_target_lines=int(number(r"Technical Skills occupies exactly (\d+) rendered lines",
                                       "the Technical Skills line target")),
        project_count=int(number(r"exactly (\d+) projects", "the project count")),
        bullet_allocation=allocation,
        page_count=int(number(r"Resume is exactly (\d+) page", "the page count")),
        tail_orphan_max=float(number(r"<\s*(\d+)%\s*final-line fill", "the hard-orphan band")),
        tail_acceptable_max=float(number(r"\d+\s*[-\u2013]\s*(\d+)%\s*may be accepted",
                                         "the acceptable tail band")),
        tail_ideal_max=float(number(r"\d+\s*-\s*(\d+)%\s*is ideal", "the ideal tail band")),
        tail_healthy_min=float(number(r"[\u2265>]=?\s*(\d+)%\s*is perfectly healthy",
                                      "the healthy tail band")),
        second_line_max_fill=float(number(r"(\d+)% mark of the 2nd line|limit of (\d+)%",
                                          "the second-line fill limit") or 60),
        verb_max_uses=_WORD_NUMBERS.get(verb_words, 2),
        data_fixes=tuple(data_fixes),
    )
    if len(policy.bullet_allocation) != policy.project_count:
        raise PolicyError(f"bullet allocation {allocation} does not cover "
                          f"{policy.project_count} projects")
    return policy


def _split_skill_items(text: str) -> tuple[str, ...]:
    """Split a skills line on commas that are not inside parentheses."""
    items, depth, current = [], 0, ""
    for char in text:
        if char == "(":
            depth += 1
        elif char == ")":
            depth = max(0, depth - 1)
        if char == "," and depth == 0:
            items.append(current)
            current = ""
        else:
            current += char
    items.append(current)
    return tuple(i.strip() for i in items if i.strip())


# ==================================================================== the JD


@dataclass(frozen=True)
class JobPosting:
    source_file: str
    text: str
    fingerprint: str
    company_name: str | None
    job_title: str | None
    job_id: str | None


_JOB_ID_PATTERNS = (
    re.compile(r"^\s*job\s*id\s*[:#]\s*(?P<id>[A-Za-z0-9][A-Za-z0-9._\-/]{1,31})\s*$",
               re.IGNORECASE | re.MULTILINE),
    re.compile(r"^\s*(?:job|requisition|req|posting)\s*(?:id|number|no\.?|#)\s*[:#]\s*"
               r"(?P<id>[A-Za-z0-9][A-Za-z0-9._\-/]{1,31})\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"\((?:job\s*id|req(?:uisition)?(?:\s*id)?)\s*[:#]?\s*"
               r"(?P<id>[A-Za-z0-9][A-Za-z0-9._\-/]{1,31})\)", re.IGNORECASE),
)


_ROLE_AT_RE = re.compile(
    r"\bAs an?\s+(?P<title>[a-z][a-z /&+-]{2,44}?)\s+at\s+(?P<company>[A-Z][A-Za-z0-9&.\-]{1,28})")
_COMPANY_HIRING_RE = re.compile(
    r"\b(?P<company>[A-Z][A-Za-z0-9&.\-]{1,28})\s+is\s+(?:hiring|seeking|looking for)\s+"
    r"(?:an?\s+)?(?P<title>[A-Z][A-Za-z /&+-]{2,44})")
_TITLE_WORDS = ("engineer", "developer", "scientist", "analyst", "architect", "manager",
                "designer", "administrator", "programmer", "consultant")

# "The BAE Systems GXP Software Team, based in Rome NY is seeking highly
# motivated entry level software engineers to join our team." The company may be
# multi-word and followed by a team name; the role is usually plural here.
_TEAM_HIRING_RE = re.compile(
    r"^[^\S\n]*(?:The[^\S\n]+)?"
    r"(?P<company>[A-Z][A-Za-z0-9&.\-]*(?:[^\S\n]+[A-Z][A-Za-z0-9&.\-]*){0,3})"
    r"[^.\n]{0,120}?\bis[^\S\n]+(?:hiring|seeking|looking for)\b[^.\n]{0,60}?"
    r"(?P<title>(?:[a-z][a-z/&+-]*[^\S\n]+){0,4}(?:engineers?|developers?|scientists?|"
    r"analysts?|architects?|programmers?|designers?))", re.MULTILINE)

# Trailing team/function words are not part of the company name:
# "BAE Systems GXP Software Team" -> "BAE Systems".
_COMPANY_DESCRIPTOR = {"gxp", "software", "engineering", "product", "platform", "research",
                       "data", "cloud", "security", "hardware", "team", "group", "division",
                       "org", "organization", "department", "labs", "lab"}


def _trim_company(name: str) -> str:
    words = name.split()
    while len(words) > 2 and words[-1].lower() in _COMPANY_DESCRIPTOR:
        words.pop()
    return " ".join(words)

# Job-board chrome must never be mistaken for a company name.
_CHROME_WORDS = {"additional", "primary", "secondary", "posting", "description",
                 "apply", "onsite", "remote", "hybrid", "same", "openings"}

# Company suffixes that end the company name: "BAE Systems GXP Software Team".
_COMPANY_TAIL = re.compile(
    r"\s+(?:GXP\s+)?(?:Software|Engineering|Product|Platform|Research|Data|Cloud|"
    r"Security|Hardware)?\s*(?:Team|Group|Division|Organization|Org|Department)\b.*$",
    re.IGNORECASE)

_ROLE_NOISE = re.compile(r"^(?:highly\s+motivated|motivated|talented|experienced|"
                         r"passionate|driven|exceptional)\s+", re.IGNORECASE)


def normalize_role_title(title: str) -> str:
    """Singularize and title-case a prose role phrase."""
    cleaned = _ROLE_NOISE.sub("", re.sub(r"\s+", " ", title).strip(" .,"))
    words = cleaned.split()
    if words and words[-1].lower().endswith("s") and not words[-1].lower().endswith("ss"):
        words[-1] = words[-1][:-1]
    return " ".join(w.capitalize() if w.islower() else w for w in words)


def _team_metadata(text: str) -> tuple[str | None, str | None]:
    """Company and role from "<Company> ... is seeking ... <roles>" prose."""
    for match in _TEAM_HIRING_RE.finditer(text):
        company = _trim_company(
            _COMPANY_TAIL.sub("", match.group("company").strip(" ,.")).strip())
        title = normalize_role_title(match.group("title"))
        if not (company and title):
            continue
        if company.split()[0].lower() in _CHROME_WORDS:
            continue
        if any(w in title.lower() for w in _TITLE_WORDS):
            return company, title
    return None, None


def _prose_metadata(text: str) -> tuple[str | None, str | None]:
    """Company and title from unambiguous prose, e.g. "As a X at Y"."""
    for match in _COMPANY_HIRING_RE.finditer(text):
        title = re.sub(r"\s+", " ", match.group("title")).strip(" .,")
        if any(word in title.lower() for word in _TITLE_WORDS):
            return match.group("company").strip(), title.title()
    for match in _ROLE_AT_RE.finditer(text):
        title = re.sub(r"\s+", " ", match.group("title")).strip(" .,")
        if any(word in title.lower() for word in _TITLE_WORDS):
            return match.group("company").strip(), title.title()
    return _team_metadata(text)


def read_jd(path: Path) -> JobPosting:
    import hashlib
    text = path.read_text(encoding="utf-8", errors="replace")
    job_id = None
    for pattern in _JOB_ID_PATTERNS:
        match = pattern.search(text)
        if match:
            candidate = match.group("id").strip().rstrip(".,;")
            if any(c.isdigit() for c in candidate):
                job_id = candidate
                break

    def labelled(pattern: str) -> str | None:
        found = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        return found.group(1).strip() if found else None

    company = labelled(r"^\s*company(?:\s*name)?\s*:\s*(.+?)\s*$")
    title = labelled(r"^\s*(?:job\s*title|position|role)\s*:\s*(.+?)\s*$")
    if not (company and title):
        prose_company, prose_title = _prose_metadata(text)
        company = company or prose_company
        title = title or prose_title
    if not company:
        stem = re.sub(r"[_\-]+", " ", path.stem).strip()
        if stem and len(stem.split()) <= 3:
            company = stem.title()        # safe filename fallback

    return JobPosting(
        source_file=path.name, text=text,
        fingerprint=hashlib.sha256(re.sub(r"\s+", " ", text).strip().lower().encode()).hexdigest(),
        company_name=company, job_title=title, job_id=job_id,
    )


# ================================================================== signals

ROLE_FAMILIES = ("Backend SWE", "HealthTech / Healthcare SWE", "AI/ML-adjacent SWE",
                 "Agentic AI / AI Agent Engineer", "Data Engineer",
                 "Cloud / Infrastructure SWE", "Systems / Low-Level SWE", "General SWE")

DOMAIN_TERMS: dict[str, tuple[tuple[str, int], ...]] = {
    "Systems / Low-Level SWE": (
        ("systems software", 6), ("low-level", 5), ("low level", 5), ("kernel", 5),
        ("operating system", 5), ("memory management", 4), ("race condition", 4),
        ("synchronization", 3), ("concurrency", 3), ("processes, threads", 4),
        ("threads", 2), ("file system", 3), ("system call", 4), ("embedded", 3),
        ("process lifecycle", 4), ("c programming", 5), ("systems programming", 6)),
    "Data Engineer": (
        ("data engineer", 7), ("etl", 5), ("data pipeline", 5), ("data processing", 4),
        ("data warehouse", 4), ("airflow", 3), ("spark", 3), ("data model", 3),
        ("analytics", 3), ("datasets", 2), ("business data", 3)),
    "Cloud / Infrastructure SWE": (
        ("cloud infrastructure", 7), ("infrastructure", 4), ("devops", 5), ("ec2", 4),
        ("vpc", 4), ("terraform", 3), ("provisioning", 3), ("deployment", 2),
        ("networking", 3), ("kubernetes", 3), ("security group", 3), ("aws", 2)),
    "AI/ML-adjacent SWE": (
        ("machine learning", 5), ("deep learning", 4), ("ai/ml", 7), ("llm", 4),
        ("large language model", 4), ("pytorch", 4), ("tensorflow", 3),
        ("scikit-learn", 3), ("model evaluation", 4), ("prompt engineering", 4),
        ("nlp", 3), ("data science", 4), ("anomaly detection", 3), ("gnn", 3), ("ocr", 3)),
    "Agentic AI / AI Agent Engineer": (
        ("agentic", 8), ("ai agent", 7), ("agent workflow", 7), ("autonomous agent", 7),
        ("tool-calling", 4), ("agent framework", 6)),
    "HealthTech / Healthcare SWE": (
        ("healthcare", 7), ("healthtech", 8), ("health tech", 8), ("clinical", 6),
        ("clinician", 6), ("patient", 5), ("telemedicine", 6), ("medical", 5),
        ("ehr", 4), ("hipaa", 4), ("fhir", 4), ("hospital", 5), ("care delivery", 5),
        ("health system", 5), ("remote care", 4), ("diagnostic", 3)),
    "Backend SWE": (
        ("backend", 6), ("back-end", 6), ("back end", 5), ("rest api", 4), ("restful", 3),
        ("django", 3), ("web service", 3), ("microservice", 3), ("server-side", 4),
        ("api development", 3), ("database schema", 3), ("full-stack", 3),
        ("full stack", 3), ("web application", 2)),
}

HEALTHCARE_STRONG_TERMS = (
    "healthcare", "healthtech", "health tech", "clinical", "clinician", "clinicians",
    "patient", "patients", "telemedicine", "telehealth", "medical software", "hospital",
    "health system", "health systems", "care delivery", "remote care", "ehr", "patient care")

_BENEFITS_LINE = re.compile(
    r"health\s*(?:insurance|coverage|plan|benefit|savings)|medical[,/ ]+dental|dental[,/ ]+vision|"
    r"\bdental\b|\b401\s*\(?k\)?|\bhsa\b|\bppo\b|equal[\s-]opportunity|\beeo\b|"
    r"without regard to|wellness\s+(?:stipend|program)|paid\s+(?:parental|family)\s+leave|"
    r"mental[\s-]health\s+(?:benefit|support|resources)|benefits\s+(?:include|package)|"
    r"comprehensive\s+benefits|union\s+employees|collective\s+bargaining|\bCBA\b|"
    r"service\s+contract\s+act|\bSCA\b|mcnamara|employee\s+assistance\s+program|"
    r"savings\s+plan|paid\s+holidays|recognition\s+(?:program|awards)|"
    r"(?:intern|employee)\s+benefits\s*:", re.IGNORECASE)

# Job-board chrome. These are navigation labels, never requirements, so they are
# matched as whole short lines rather than as substrings of real prose.
_UI_NAV_LINE = re.compile(
    r"^(?:primary|additional|secondary)(?:\s+posting)?$|^posting\s*\d*$|"
    r"^\d+\s+openings?$|^(?:apply|share|save|print|back)(?:\s+now)?$|"
    r"^(?:onsite|remote|hybrid|entry\s+level|full[\s-]time|part[\s-]time)$|"
    r"^(?:description|responsibilities|qualifications|benefits)$|^same role[, ]",
    re.IGNORECASE)

CODE_QUALITY_SIGNALS: dict[str, tuple[str, ...]] = {
    "pull_request_review": (r"review(?:ing|s)?\s+(?:the\s+)?pull[\s-]?requests?",
                            r"pull[\s-]?requests?\s+review", r"\bpr\s+review",
                            r"pull[\s-]?request[\s-]?based\s+development",
                            r"merg(?:e|ing)\s+pull[\s-]?requests?"),
    "peer_code_review": (r"code[\s-]review", r"peer[\s-]review", r"review(?:ing|s)?\s+code",
                         r"code\s+reviews?"),
    "code_quality_ownership": (r"code[\s-]quality", r"quality\s+of\s+(?:the\s+)?code(?:base)?",
                               r"maintain(?:ing)?\s+(?:high\s+)?(?:code\s+)?quality",
                               r"coding\s+standards"),
    "review_mentoring": (r"feedback\s+to\s+(?:other\s+)?engineers",
                         r"mentor(?:ing)?\s+(?:other\s+)?engineers", r"constructive\s+feedback"),
    "testing_quality_ownership": (r"test(?:ing)?\s+(?:and|&)\s+quality", r"quality\s+ownership",
                                  r"test\s+coverage", r"quality\s+(?:gates|bar)"),
}

_PRIMARY_REVIEW_HEAD = tuple(re.compile(p, re.IGNORECASE) for p in (
    r"^\W*review(?:ing|s)?\b",
    r"^\W*(?:perform|conduct|provide|participate\s+in|lead|own)\w*\s+(?:\w+\s+){0,2}?(?:code|peer|pull)",
    r"^\W*(?:strong|excellent|solid|proven|deep)\s+code[\s-]review",
    r"^\W*code[\s-]review", r"^\W*peer[\s-]review", r"^\W*pull[\s-]?requests?\b",
    r"^\W*maintain(?:ing)?\s+code[\s-]quality"))

_ACTIVITY_VERB = re.compile(
    r"\b(build|builds|building|develop|develops|developing|design|designs|designing|debug|"
    r"debugs|debugging|test|tests|testing|deploy|deploys|deploying|maintain|maintains|"
    r"maintaining|implement|implements|implementing|review|reviews|reviewing|write|writes|"
    r"writing|monitor|monitors|monitoring|automate|automates|troubleshoot|investigate|"
    r"configure|integrate)\b", re.IGNORECASE)

GRADUATE_PATTERNS: dict[str, tuple[str, ...]] = {
    "new_grad_title": (r"\bnew[\s-]grad(?:uate)?s?\b", r"\bgrad(?:uate)?\s+hir(?:e|ing)\b",
                       r"\bnew[\s-]college[\s-]grad(?:uate)?s?\b"),
    "university_recruiting": (r"\buniversity\s+recruit(?:ing|ment)\b",
                              r"\bcampus\s+recruit(?:ing|ment)\b",
                              r"\buniversity\s+(?:hiring|talent|programs?)\b",
                              r"\bcampus\s+hiring\b", r"\bearly\s+talent\s+program\b"),
    "graduation_window": (r"\bgraduat(?:e|ing)\s+between\b",
                          r"\bgraduat(?:e|ing)\s+(?:in|by|before|no later than)\s+(?:\w+\s+)?(?:19|20)\d\d\b",
                          r"\bgraduation\s+dates?\s+(?:between|from|of)\b",
                          r"\bmust\s+graduate\s+(?:by|between|in)\b"),
    "class_of_year": (r"\bclass\s+of\s+(?:19|20)\d\d\b",),
    "recent_graduate": (r"\brecent\s+grad(?:uate)?s?\b", r"\brecently\s+graduated\b"),
    "graduating_student_gate": (r"\bmust\s+be\s+(?:a\s+)?(?:currently\s+)?enrolled\b",
                                r"\bcurrently\s+enrolled\s+(?:student|in)\b",
                                r"\bmust\s+be\s+(?:a\s+)?graduating\s+(?:student|senior)\b",
                                r"\bgraduating\s+(?:student|senior)s?\s+(?:only|are eligible|eligible)\b",
                                r"\bopen\s+(?:only\s+)?to\s+(?:current\s+)?students\b"),
}

ZERO_WEIGHT_PATTERNS: dict[str, tuple[str, ...]] = {
    "entry_level": (r"\bentry[\s-]level\b",),
    "junior_title": (r"\bjunior\b", r"\bjr\.?\s+(?:software|engineer)\b"),
    "early_career": (r"\bearly[\s-]career\b",),
    "years_experience_range": (r"\b\d\s*[-\u2013to]{1,3}\s*\d\s*years?\b",
                               r"\b\d\+?\s*years?\s+of\s+(?:relevant\s+)?experience\b"),
    "degree_requirement": (r"\b(?:bachelor|master|bs|ms)['\u2019]?s?\b[^.\n]{0,30}"
                           r"\b(?:required|preferred|degree)\b",
                           r"\bdegree\s+in\s+computer\s+science\b"),
}


@dataclass
class Signals:
    role_family: str
    healthcare: bool
    healthcare_reason: str
    code_quality: bool
    code_quality_reason: str
    jd_emphasizes_agentic_ai: bool
    jd_emphasizes_scrum: bool
    graduate_score: int
    graduate_reason: str
    graduate_matched: dict[str, list[str]]
    domain_scores: dict[str, int]

    @property
    def section_mode(self) -> str:
        return "education_first" if self._grad_ok else "experience_first"

    _grad_ok: bool = False


def _jd_lines(text: str) -> list[str]:
    return [l.strip().lstrip("-*\u2022").strip() for l in text.splitlines() if l.strip()]


def _title_line(text: str) -> str:
    match = re.search(r"^\s*(?:job\s*title|position|role)\s*:\s*(.+)$", text,
                      re.IGNORECASE | re.MULTILINE)
    return match.group(1).lower() if match else ""


def classify_jd(jd_text: str, section_order: SectionOrder) -> Signals:
    """Deterministic JD signals. The LLM may agree; Python decides."""
    lowered = jd_text.lower()
    title = _title_line(jd_text)

    scores: dict[str, int] = {}
    for family, terms in DOMAIN_TERMS.items():
        total = 0
        for term, weight in terms:
            hits = lowered.count(term)
            if hits:
                total += weight * min(hits, 3)
            if term in title:
                total += weight * 3
        if total:
            scores[family] = total

    healthcare, healthcare_reason = _detect_healthcare(jd_text)
    code_quality, code_quality_reason = _detect_code_quality(jd_text)

    if healthcare:
        family = "HealthTech / Healthcare SWE"
    elif scores:
        family = max(scores, key=lambda f: (scores[f], -list(DOMAIN_TERMS).index(f)))
        if family == "HealthTech / Healthcare SWE":
            others = {f: s for f, s in scores.items() if f != family}
            family = max(others, key=others.get) if others else "General SWE"
    else:
        family = "General SWE"

    grad_score, grad_reason, grad_matched = _detect_graduate(jd_text, section_order)
    signals = Signals(
        role_family=family, healthcare=healthcare, healthcare_reason=healthcare_reason,
        code_quality=code_quality, code_quality_reason=code_quality_reason,
        jd_emphasizes_agentic_ai=any(t in lowered for t in
                                     ("agentic", "ai agent", "agent workflow", "autonomous agent")),
        jd_emphasizes_scrum=any(t in lowered for t in ("scrum", "sprint planning", "sprints")),
        graduate_score=grad_score, graduate_reason=grad_reason, graduate_matched=grad_matched,
        domain_scores=dict(sorted(scores.items(), key=lambda kv: -kv[1])),
    )
    signals._grad_ok = grad_score >= section_order.threshold
    return signals


def _detect_healthcare(text: str) -> tuple[bool, str]:
    """Healthcare is judged on the employer/product/role, never on benefits text."""
    kept = [l for l in text.splitlines() if not _BENEFITS_LINE.search(l)]
    ignored = len(text.splitlines()) - len(kept)
    domain = "\n".join(kept)
    lowered = domain.lower()
    header = "\n".join(l.lower() for l in text.splitlines()[:12]
                       if re.match(r"^\s*(?:company(?:\s*name)?|job\s*title|position|role)\s*:",
                                   l, re.IGNORECASE))

    def present(term: str, where: str) -> bool:
        return bool(re.search(r"(?<![a-z0-9])" + re.escape(term) + r"(?![a-z0-9])", where))

    found = [t for t in HEALTHCARE_STRONG_TERMS if present(t, lowered)]
    in_header = [t for t in HEALTHCARE_STRONG_TERMS if present(t, header)]
    weighted = sum(w * min(lowered.count(t), 3)
                   for t, w in DOMAIN_TERMS["HealthTech / Healthcare SWE"])
    mentions = sum(len(re.findall(r"(?<![a-z0-9])" + re.escape(t) + r"(?![a-z0-9])", lowered))
                   for t in found)
    strong = bool(in_header) or len(found) >= 2 or mentions >= 3
    is_healthcare = strong and weighted >= 10

    if is_healthcare:
        where = "job title/company line" if in_header else "role and product description"
        reason = f"healthcare domain stated in the {where}; terms {found[:5]}, weighted {weighted}"
    elif found:
        reason = (f"healthcare words present ({found[:5]}, weighted {weighted}) but not as a "
                  f"stated employer/product/role domain")
    else:
        reason = "no healthcare-domain terms in the role/product description"
    if ignored:
        reason += f"; ignored {ignored} benefits/EEO line(s)"
    return is_healthcare, reason


def _detect_code_quality(text: str) -> tuple[bool, str]:
    """Narrow definition: generic collaboration words never trigger this."""
    lines = _jd_lines(text)
    evidence: dict[str, list[str]] = {}
    for line in lines:
        for name, patterns in CODE_QUALITY_SIGNALS.items():
            if any(re.search(p, line, re.IGNORECASE) for p in patterns):
                evidence.setdefault(name, []).append(line)

    for line in lines:
        if not any(re.search(p, line, re.IGNORECASE)
                   for pats in CODE_QUALITY_SIGNALS.values() for p in pats):
            continue
        if any(rx.search(line) for rx in _PRIMARY_REVIEW_HEAD):
            return True, f"condition A: review is the subject of a requirement line ({line[:60]!r})"
        verbs = {v.lower() for v in _ACTIVITY_VERB.findall(line)}
        review_verbs = {v for v in verbs if v.startswith("review")}
        if review_verbs and not (verbs - review_verbs):
            return True, f"condition A: review-only responsibility line ({line[:60]!r})"

    if len(evidence) >= 2:
        return True, f"condition B: {len(evidence)} strong signals ({', '.join(sorted(evidence))})"
    if evidence:
        return False, (f"only 1 strong signal ({next(iter(evidence))}) and no review-focused line; "
                       f"generic collaboration language is insufficient")
    return False, "no code-review/quality-ownership signal"


def _detect_graduate(text: str, order: SectionOrder) -> tuple[int, str, dict[str, list[str]]]:
    matched: dict[str, list[str]] = {}
    score = 0
    for signal, patterns in GRADUATE_PATTERNS.items():
        hits = [re.sub(r"\s+", " ", m.group(0)) for p in patterns
                for m in re.finditer(p, text, re.IGNORECASE)]
        if hits:
            matched[signal] = hits
            score += order.weights.get(signal, 0)
    zero = [s for s, patterns in ZERO_WEIGHT_PATTERNS.items()
            if any(re.search(p, text, re.IGNORECASE) for p in patterns)]
    if matched:
        reason = (", ".join(f"{s} (+{order.weights.get(s, 0)})" for s in matched)
                  + f"; score {score} vs threshold {order.threshold}")
    else:
        reason = "no graduate-recruiting signal"
    if zero:
        reason += f"; zero-weight seniority/degree language ({', '.join(zero)}) never triggers it"
    return score, reason, matched


# ====================================================== professional experience


@dataclass
class ExperienceDecision:
    latex_block: str
    shipped: list[tuple[str, str, str]]        # (bullet_id, action, latex)
    swaps: list[tuple[str, str, str]]          # (source_id, target_id, rule label)
    rule: str
    allowed_plain: set[str]
    expected_plain: list[str]
    shipped_ids: list[str] = field(default_factory=list)

    @property
    def swap_summary(self) -> str:
        return ", ".join(f"{source} -> {target}" for source, target, _ in self.swaps) or "no swap"


def experience_rule_for(signals: Signals) -> tuple[str, str]:
    """The policy label that applies, and which spreadsheet table it lives in.

    Precedence is unchanged: healthcare + code-quality, then healthcare alone,
    then non-healthcare + code-quality, then the detected role family.
    """
    if signals.healthcare and signals.code_quality:
        return "Healthcare + code-quality-heavy", "jd_signal"
    if signals.healthcare:
        return "Healthcare company (regardless of role type)", "jd_signal"
    if signals.code_quality:
        return "Non-healthcare + code-quality/collaboration-heavy", "jd_signal"
    return signals.role_family, "job_role"


def select_experience(template: Template, policy: Policy, signals: Signals) -> ExperienceDecision:
    """Build Professional Experience from spreadsheet-approved ids.

    Swaps are `source_id -> target_id`, so editing the wording of a bullet can
    never break a mapping. The template supplies no wording at all: this
    assembles the whole block, which is what keeps the final PDF verification
    independent of the template.
    """
    label, applies_when = experience_rule_for(signals)
    rules = policy.rules_for(label, applies_when)
    replacements = {rule.source_id: rule for rule in rules}
    rule_description = (f"{label} ({applies_when})" if rules
                        else f"{label} ({applies_when}): no swap authorized")

    shipped: list[tuple[str, str, str]] = []
    swaps: list[tuple[str, str, str]] = []
    shipped_ids: list[str] = []
    chunks: list[str] = []
    open_itemize = False

    def close() -> None:
        nonlocal open_itemize
        if open_itemize:
            chunks.append("\t" + r"\end{itemize}")
            open_itemize = False

    for entry in policy.experience_library:
        if entry.is_role_header:
            close()
            chunks.append("\t" + entry.latex)
            chunks.append("\t" + r"\begin{itemize}")
            chunks.append("\t\t" + r"\setlength{\itemsep}{0pt}")
            open_itemize = True
            continue
        rule = replacements.get(entry.bullet_id)
        if rule:
            alternate = policy.alternate_entry(rule.target_id)
            if alternate is None:                    # integrity check runs first
                raise PolicyError(f"rule {rule.label!r} targets unknown alternate id="
                                  f"{rule.target_id}")
            chunks.append("\t\t" + alternate.latex)
            shipped.append((entry.bullet_id, "SWAP", alternate.latex))
            shipped_ids.append(alternate.bullet_id)
            swaps.append((entry.bullet_id, alternate.bullet_id,
                          f"{rule.label} (sheet row {rule.row})"))
        else:
            chunks.append("\t\t" + entry.latex)
            shipped.append((entry.bullet_id, "KEEP_EXACT", entry.latex))
            shipped_ids.append(entry.bullet_id)
    close()

    # Approved wording comes only from the spreadsheet.
    allowed = {entry.plain for entry in policy.experience_library if not entry.is_role_header}
    allowed |= {entry.plain for entry in policy.alternate_library}

    return ExperienceDecision(
        latex_block="\n".join(chunks), shipped=shipped, swaps=swaps, rule=rule_description,
        allowed_plain=allowed, expected_plain=[latex_to_plain(l) for _, _, l in shipped],
        shipped_ids=shipped_ids,
    )


def verify_experience_in_pdf(pdf_bullets: list[str], decision: ExperienceDecision
                             ) -> tuple[bool, list[str]]:
    """Compare the PDF's Experience bullets against the spreadsheet wording.

    Only wrapping/whitespace and glyph rendering are normalized; every word
    must match the approved text of the id this run shipped, or another
    spreadsheet-approved bullet. The template is never consulted.
    """
    problems: list[str] = []
    rendered = [normalize_plain(b) for b in pdf_bullets]
    expected = [normalize_plain(e) for e in decision.expected_plain]
    allowed = {normalize_plain(a) for a in decision.allowed_plain}

    for index, bullet in enumerate(rendered):
        if bullet not in allowed:
            closest = min(allowed, key=lambda a: _distance(a, bullet)) if allowed else ""
            problems.append(
                f"PDF Experience bullet {index + 1} is not spreadsheet-approved wording.\n"
                f"      actual   = {bullet!r}\n      closest approved = {closest!r}")
    if len(rendered) != len(expected):
        problems.append(f"PDF has {len(rendered)} Experience bullets, expected {len(expected)}")
    else:
        for index, (got, want) in enumerate(zip(rendered, expected)):
            if got != want:
                shipped = (decision.shipped_ids[index]
                           if index < len(decision.shipped_ids) else "?")
                problems.append(
                    f"PDF Experience bullet {index + 1} is not the approved text for the id "
                    f"this run selected ({shipped}).\n"
                    f"      actual   = {got!r}\n      expected = {want!r}")
    return not problems, problems


def _distance(a: str, b: str) -> int:
    common = len(os.path.commonprefix([a, b]))
    return abs(len(a) - len(b)) + (len(a) - common)


# ================================================================== projects


@dataclass
class ProjectRender:
    project: Project
    bullets: list[str]


def allocate_bullets(selected_ids: list[str], policy: Policy) -> dict[str, int]:
    """3/2/2 strictly by RELEVANCE rank - rank #1 gets the extra bullet.

    `selected_ids` must be in relevance order. The allocation is keyed by
    project id so it travels with the project when the rendering order is
    later changed to chronological.
    """
    allocation = {}
    for index, project_id in enumerate(selected_ids):
        allocation[project_id] = (policy.bullet_allocation[index]
                                  if index < len(policy.bullet_allocation)
                                  else policy.bullet_allocation[-1])
    return allocation


_MONTHS = {name.lower(): number for number, name in enumerate(
    ("January", "February", "March", "April", "May", "June", "July", "August",
     "September", "October", "November", "December"), start=1)}


def project_date_key(project: Project) -> tuple[int, int]:
    """Sortable (year, month) from the canonical project date.

    Deterministic and factual: it reads the date recorded in the master, never
    generated prose. A date the master does not state sorts oldest so it can
    never jump the queue.
    """
    text = (project.date or "").strip()
    year = re.search(r"(19|20)\d{2}", text)
    month = next((number for name, number in _MONTHS.items() if name in text.lower()), 0)
    return (int(year.group(0)) if year else 0, month)


def display_order(selected_ids: list[str], master: MasterFacts) -> list[str]:
    """Reverse-chronological rendering order, newest project first.

    Selection and bullet allocation are decided by relevance elsewhere; this
    only decides where each project is rendered. Ties fall back to the
    relevance rank, which is the incoming order of `selected_ids`.
    """
    ranks = {project_id: rank for rank, project_id in enumerate(selected_ids)}

    def sort_key(project_id: str):
        project = master.project(project_id)
        year, month = project_date_key(project) if project else (0, 0)
        return (-year, -month, ranks[project_id])

    return sorted(selected_ids, key=sort_key)


def build_projects_latex(renders: list[ProjectRender]) -> str:
    """Deterministic project headers: factual metadata only, never a tech stack."""
    chunks: list[str] = []
    for render in renders:
        parts = [r"\textbf{" + typeset(render.project.name) + "}"]
        parts += [typeset(p.strip()) for p in render.project.context.split("|") if p.strip()]
        header = r"\noindent" + " $|$ ".join(parts)
        header += r" \hfill \textit{" + typeset(render.project.date) + "}"
        chunks.append("\t" + header)
        chunks.append(itemize([typeset(b) for b in render.bullets]))
    return "\n".join(chunks)


_PROJECT_STOPWORDS = {
    "and", "the", "for", "with", "that", "this", "using", "from", "into", "our", "you",
    "are", "will", "have", "has", "your", "who", "not", "but", "all", "experience",
    "work", "working", "strong", "build", "building", "team", "teams", "engineer",
    "engineers", "engineering", "software", "role", "job", "company", "title",
    "requirements", "responsibilities", "preferred", "plus", "skills", "ability",
    "familiarity", "knowledge", "such", "other", "well",
}


def _content_tokens(text: str) -> set[str]:
    out: set[str] = set()
    for token in re.findall(r"[a-z0-9+#]+", text.lower()):
        if len(token) < 2 or token in _PROJECT_STOPWORDS:
            continue
        out.add(token)
        if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            out.add(token[:-1])
    return out


def rank_projects(master: MasterFacts, jd_text: str, role_family: str
                  ) -> list[tuple[str, float, str]]:
    """Deterministic relevance ranking, used by the mock and as a fallback."""
    jd_tokens = _content_tokens(jd_text)
    ranked = []
    for project in master.projects:
        tech = _content_tokens(" ".join(project.tech)) & jd_tokens
        tags = _content_tokens(" ".join(project.tags)) & jd_tokens
        evidence = _content_tokens(" ".join(project.evidence)) & jd_tokens
        relevance = 3.0 * len(tech) + 2.0 * len(tags) + len(evidence)
        blob = " ".join((project.name, *project.tech, *project.tags)).lower()
        if any(term in blob for term, _ in DOMAIN_TERMS.get(role_family, ())):
            relevance *= 1.2
        depth = min(len(project.evidence), 12) * 0.5
        measurable = 2.0 * sum(1 for e in project.evidence if re.search(r"\d", e)) ** 0.5
        matched = ", ".join(sorted(tech | tags)[:6]) or "no direct keyword overlap"
        ranked.append((project.project_id, round(relevance + depth + measurable, 1), matched))
    ranked.sort(key=lambda row: (-row[1], row[0]))
    return ranked


# ========================================================== technical skills


@dataclass
class SkillCandidate:
    name: str
    category: str
    tier: int                 # 1 exact JD+experience, 2 JD-supported, 3 project+JD, 4 project
    source: str
    reason: str
    rank_hint: int = 9        # selection rank of the project that evidences it


@dataclass
class SkillsState:
    categories: list[tuple[str, list[str]]]
    mandatory_keys: set[str]
    candidates: list[SkillCandidate]
    added: list[tuple[str, str, str]] = field(default_factory=list)     # name, source, reason
    removed: list[tuple[str, str]] = field(default_factory=list)        # name, reason
    withheld: list[str] = field(default_factory=list)
    rejected: list[tuple[str, str]] = field(default_factory=list)
    next_candidate: int = 0

    def lines(self) -> list[str]:
        return [f"{label}: {', '.join(items)}" for label, items in self.categories if items]

    def latex(self) -> str:
        items = [r"\textbf{" + escape_latex(label) + ":} " + escape_latex(", ".join(values))
                 for label, values in self.categories if values]
        return itemize(items, indent="\t")

    def all_skills(self) -> list[str]:
        return [s for _, items in self.categories for s in items]

    def keys(self) -> set[str]:
        return {fold_term(s) for s in self.all_skills()}

    def is_rejected(self, name: str) -> bool:
        return any(fold_term(name) == fold_term(n) for n, _ in self.rejected)

    def reject(self, name: str, reason: str) -> None:
        if not self.is_rejected(name):
            self.rejected.append((name, reason))

    def add(self, candidate: SkillCandidate) -> bool:
        if fold_term(candidate.name) in self.keys():
            return False
        for label, items in self.categories:
            if label == candidate.category:
                items.append(candidate.name)
                break
        else:
            self.categories.append((candidate.category, [candidate.name]))
        self.added.append((candidate.name, candidate.source, candidate.reason))
        return True

    def remove(self, name: str, reason: str) -> bool:
        for label, items in self.categories:
            for item in list(items):
                if fold_term(item) == fold_term(name):
                    items.remove(item)
                    self.removed.append((item, reason))
                    self.added = [a for a in self.added if fold_term(a[0]) != fold_term(item)]
                    return True
        return False

    def removable_addition(self, category: str | None = None) -> str | None:
        """Lowest-value non-mandatory addition, optionally within one category."""
        for name, _, _ in reversed(self.added):
            if fold_term(name) in self.mandatory_keys:
                continue
            if category is None:
                return name
            if any(fold_term(name) == fold_term(item)
                   for label, items in self.categories if label == category for item in items):
                return name
        return None


def build_skills(master: MasterFacts, policy: Policy, signals: Signals, jd_text: str,
                 selected: list[Project], model_ranked: list[str]) -> SkillsState:
    """Mandatory spreadsheet block plus ranked, supported, JD-relevant additions."""
    category_map = dict(zip(master.tier_categories, policy.category_labels()))

    categories: list[tuple[str, list[str]]] = []
    mandatory_keys: set[str] = set()
    withheld: list[str] = []
    seen: set[str] = set()
    for label, items in policy.mandatory:
        kept: list[str] = []
        for item in items:
            canonical = master.canonical_skill(item) or item
            skill = master.skill(canonical)
            if skill and skill.condition and not getattr(signals, skill.condition, False):
                withheld.append(f"{canonical} withheld: {skill.condition} is false for this JD")
                continue
            key = fold_term(canonical)
            if key in seen:
                continue
            seen.add(key)
            mandatory_keys.add(key)
            kept.append(canonical)
        categories.append((label, kept))

    # --- candidate additions ------------------------------------------------
    candidates: list[SkillCandidate] = []
    rejected: list[tuple[str, str]] = []
    project_tech = {fold_term(t): t for p in selected for t in p.tech}
    model_order = {fold_term(m): i for i, m in enumerate(model_ranked)}
    # A skill evidenced by the rank-1 project outranks one from rank 3, so a
    # Backend resume fills space with that project's stack instead of reaching
    # into an unrelated specialization.
    project_rank: dict[str, int] = {}
    for rank, project in enumerate(selected):
        for tech in project.tech:
            project_rank.setdefault(fold_term(tech), rank)
    jd_named_pool = tuple(sk for sk in master.skills if _names_skill(jd_text, sk.name))

    for skill in master.skills:
        key = fold_term(skill.name)
        if key in seen:
            continue
        if skill.condition and not getattr(signals, skill.condition, False):
            continue
        category = category_map.get(skill.category, skill.category)
        if category not in policy.category_labels():
            continue
        jd_named = _names_skill(jd_text, skill.name)
        from_project = key in project_tech
        if jd_named and master.supported_by_experience(_bare(skill.name)):
            tier, source = 1, "jd+experience"
            reason = "exact JD keyword, used in Professional Experience"
        elif jd_named and (from_project or master.skill_evidence.get(skill.name)):
            tier, source = 2, "jd+project" if from_project else "jd+master"
            reason = "exact JD keyword, supported by selected-project or master evidence"
        elif from_project and (sibling := _jd_named_sibling(skill, selected, jd_named_pool,
                                                            master)):
            tier, source = 3, "project"
            reason = (f"selected-project technology used in the same work as {sibling!r}, "
                      f"which this JD names")
        elif from_project:
            tier, source = 4, "project"
            reason = ("technology demonstrated by a selected project, with no direct tie to "
                      "this JD")
        else:
            continue
        candidates.append(SkillCandidate(skill.name, category, tier, source, reason,
                                         rank_hint=project_rank.get(key, 9)))

    candidates.sort(key=lambda c: (c.tier, c.rank_hint,
                                   model_order.get(fold_term(c.name), 500), c.name))

    state = SkillsState(categories=categories, mandatory_keys=mandatory_keys,
                        candidates=candidates, withheld=withheld, rejected=rejected)
    _order_by_jd_relevance(state, jd_text)
    return state


def _bare(name: str) -> str:
    return re.sub(r"\s*\([^)]*\)", "", name).strip()


def _names_skill(jd_text: str, skill: str) -> bool:
    lowered = jd_text.lower()
    forms = {skill, _bare(skill)}
    for piece in re.findall(r"\(([^)]*)\)", skill):
        forms.update(p.strip() for p in re.split(r"[/,]", piece))
    if "/" in skill and "(" not in skill:
        forms.update(p.strip() for p in skill.split("/"))
    return any(len(f) >= 2 and re.search(
        r"(?<![a-z0-9+#])" + re.escape(f.lower()) + r"(?![a-z0-9+#])", lowered) for f in forms)


def _jd_named_sibling(skill: Skill, selected: list[Project], jd_named_pool: tuple[Skill, ...],
                      master: MasterFacts) -> str | None:
    """Find a JD-named skill demonstrated by the same evidence as `skill`.

    This is what makes a filler skill relevant rather than merely true: the JD
    asks for WebSockets, the same evidence sentence that proves WebSockets also
    proves Django Channels, so Django Channels belongs on the resume. Nothing
    in an unrelated project qualifies just because the project was selected.
    """
    needles = _skill_needles(skill.name)
    for project in selected:
        for sentence in project.evidence:
            if not _mentions(sentence, needles):
                continue
            for other in jd_named_pool:
                if fold_term(other.name) == fold_term(skill.name):
                    continue
                if _mentions(sentence, _skill_needles(other.name)):
                    return other.name
    return None


def _project_tag_matches_jd(jd_text: str, selected: list[Project], skill: str) -> bool:
    lowered = jd_text.lower()
    for project in selected:
        if not any(fold_term(t) == fold_term(skill) for t in project.tech):
            continue
        for tag in project.tags:
            if len(tag) >= 4 and re.search(
                    r"(?<![a-z0-9])" + re.escape(tag.lower()) + r"(?![a-z0-9])", lowered):
                return True
    return False


def _order_by_jd_relevance(state: SkillsState, jd_text: str) -> None:
    """Within each category, strongest JD matches first (spreadsheet rule 8)."""
    for _, items in state.categories:
        original = {name: index for index, name in enumerate(items)}
        items.sort(key=lambda n: (0 if _names_skill(jd_text, n) else 1, original[n]))


# ============================================================ jd requirements

_REQ_SECTIONS = (
    ("responsibilities", "required"), ("what you will do", "required"),
    ("what you'll do", "required"), ("the role", "required"), ("requirements", "required"),
    ("qualifications", "required"), ("basic qualifications", "required"),
    ("minimum qualifications", "required"), ("eligibility", "required"),
    ("preferred qualifications", "preferred"), ("preferred", "preferred"),
    ("nice to have", "preferred"), ("bonus", "preferred"), ("plus", "preferred"),
)
_PREFERRED_INLINE = (r"\bpreferred\b", r"\bis a plus\b", r"\bnice to have\b",
                     r"\bfamiliarity with\b", r"\bbonus\b", r"\bideally\b")
_REQUIRED_INLINE = (r"\brequired\b", r"\bmust\b", r"\bstrong\b", r"\bproven\b")


_ELIGIBILITY_SECTIONS = ("eligibility", "work authorization", "legal")
_ELIGIBILITY_MARKERS = (r"\bgraduat\w*\b", r"\bclass of\b", r"\bnew grad(uate)?\b",
                        r"\bwork authorization\b", r"\bsponsor\w*\b", r"\bvisa\b",
                        r"\bmust be (?:a |an )?(?:currently )?enrolled\b",
                        r"\bdegree\b.*\b(?:required|expected)\b")


@dataclass(frozen=True)
class Requirement:
    """One JD requirement, owned by Python and keyed by a stable id.

    `original_text` is the JD's own wording and is authoritative: a model may
    describe evidence for a requirement but never restates its identity.
    """

    text: str
    importance: str                 # required | preferred | unclear
    terms: tuple[str, ...]
    section: str
    requirement_id: str = ""
    kind: str = "capability"        # capability|qualification|eligibility|logistics|condition
    authority: str = "hard"         # hard = explicit requirement, signal = role prose
    line: int = 0

    @property
    def headline(self) -> str:
        return re.sub(r"\s+", " ", self.text).strip().rstrip(".")

    @property
    def original_text(self) -> str:
        return self.headline

    @property
    def is_eligibility(self) -> bool:
        """Eligibility is assessed separately, never as a skill gap."""
        return self.kind == "eligibility"

    @property
    def subjective(self) -> bool:
        """A credential no resume text can settle, e.g. "academic excellence"."""
        return any(re.search(p, self.text, re.IGNORECASE)
                   for p in _SUBJECTIVE_QUALIFICATION)

    @property
    def scored(self) -> bool:
        """Whether this requirement may influence the fit score."""
        return (self.authority == "hard" and self.kind in FIT_SCORED_KINDS
                and not self.subjective)

    def as_dict(self) -> dict:
        return {"requirement_id": self.requirement_id, "original_text": self.original_text,
                "importance": self.importance, "kind": self.kind,
                "authority": self.authority, "terms": list(self.terms)}


@dataclass
class RequirementMatch:
    requirement: Requirement
    verdict: str                    # strong_match | partial_match | unsupported
    evidence: str = ""
    source: str = "none"            # experience | project | skills | education | none
    limitation: str = ""
    term: str = ""


REQUIREMENT_KINDS = ("capability", "qualification", "eligibility", "logistics",
                     "condition")
# Only explicit capability/qualification requirements move the fit score. The
# master cannot prove whether someone will relocate or is vaccinated, so those
# are reported for manual review instead of scored.
FIT_SCORED_KINDS = ("capability", "qualification")

_HARD_LANGUAGE = (
    r"\bmust\b", r"\brequired?\b", r"\brequires\b", r"\beligible\b", r"\beligibility\b",
    r"\bqualificat\w+\b", r"\bdegree\b", r"\bexperience with\b", r"\bproficien\w+\b",
    r"\bknowledge of\b", r"\bfamiliarity with\b", r"\bBS/BA\b", r"\bbachelor\w*\b",
    r"\bmaster'?s\b", r"\byears? of\b", r"\bability to\b", r"\bdemonstrated\b",
)
_ELIGIBILITY_KIND = (
    r"\bvisa\b", r"\bsponsor\w*\b", r"\bwork authoriz\w*\b", r"\bauthoriz\w+ to work\b",
    r"\beligible to work\b", r"\bcitizen\w*\b", r"\bTN status\b", r"\bNAFTA\b",
    r"\bsecurity clearance\b", r"\bgraduat\w*\b", r"\bclass of\b", r"\bnew grad(uate)?\b",
)
_CONDITION_KIND = (
    r"\bvaccinat\w*\b", r"\bvaccine\b", r"\bcovid\b", r"\bbackground check\b",
    r"\bdrug (?:test|screen\w*)\b", r"\bphysical exam\b", r"\bimmuniz\w*\b",
)
_LOGISTICS_KIND = (
    r"\brelocat\w*\b", r"\bon-?site\b", r"\bhybrid\b", r"\bremote\b", r"\btravel\b",
    r"\bcommut\w*\b", r"\bbased (?:in|on|at)\b", r"\bin-office\b",
)
_SUBJECTIVE_QUALIFICATION = (
    r"\bhistory of\b", r"\bacademic excellence\b", r"\bprofessional success\b",
    r"\btrack record\b", r"\bpassion\w*\b", r"\bculture fit\b",
    r"\bexcellent communicat\w+\b", r"\bself-?starter\b", r"\bteam player\b",
)

_QUALIFICATION_KIND = (
    r"\bBS/BA\b", r"\bbachelor\w*\b", r"\bmaster'?s\b", r"\bdegree\b", r"\bdiploma\b",
    r"\bGPA\b", r"\bPh\.?D\b", r"\bacademic excellence\b", r"\bprofessional success\b",
    r"\bcertificat\w+\b",
)
# Perks, customer lists, city copy and EEO text are never requirements.
_MARKETING_LINE = re.compile(
    r"accolade|best city|greenest|fittest|young professionals|sabbatical|"
    r"restaurant-quality|comfy chair|kayak|marathon|concert|our community includes|"
    r"top-ranked|u\.s\. news|world report|campus was designed|equal opportunity|"
    r"non-discrimination|merit-based compensation|stock grant|raises and bonuses|"
    r"learn more about our team|code that saves lives|about the job|"
    r"fastest growing market|state capital|customers", re.IGNORECASE)


# Technology names a JD may cite that the master does not claim. Used ONLY to
# recognize role signals; nothing here can enter Technical Skills or the fit
# score, because a signal is never a scored requirement.
_JD_SIGNAL_TERMS = (
    "JS", "JavaScript", "TS", "TypeScript", "C#", "Java", "Go", "Rust", "Ruby", "PHP",
    "Windows", "macOS", "Android", "iOS", "Unix", "Solaris", "Kotlin", "Swift",
    "Scala", "React", "Angular", "Vue", "Node.js", "Spring",
    ".NET", "Kubernetes", "Terraform", "Kafka", "Spark", "Airflow", "GraphQL",
    "machine learning", "analytics", "user-centered design", "user centered design",
    "modern development methodologies", "microservices", "CI/CD", "cloud",
    "distributed systems", "data structures", "algorithms", "object-oriented",
)


def jd_signal_terms(text: str) -> tuple[str, ...]:
    """Technology/approach names the JD itself mentions."""
    found = []
    lowered = text.lower()
    for term in _JD_SIGNAL_TERMS:
        needle = term.lower()
        pattern = r"(?<![a-z0-9+#.])" + re.escape(needle) + r"(?![a-z0-9+#])"
        if re.search(pattern, lowered):
            found.append(term)
    return tuple(sorted(set(found), key=lambda t: (-len(t), t)))


def requirement_kind(text: str) -> str:
    """Classify a requirement by what it actually demands."""
    for patterns, kind in ((_ELIGIBILITY_KIND, "eligibility"),
                           (_CONDITION_KIND, "condition"),
                           (_LOGISTICS_KIND, "logistics"),
                           (_QUALIFICATION_KIND, "qualification")):
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns):
            return kind
    return "capability"


def jd_vocabulary(master: MasterFacts) -> dict[str, str]:
    """Recognizable technology/concept terms, all drawn from the master."""
    vocab: dict[str, str] = {}
    for skill in master.skills:
        vocab[fold_term(skill.name)] = skill.name
        bare = re.sub(r"\s*\([^)]*\)", "", skill.name).strip()
        if bare:
            vocab.setdefault(fold_term(bare), bare)
        for piece in re.findall(r"\(([^)]*)\)", skill.name):
            for part in re.split(r"[/,]", piece):
                if len(part.strip()) > 1:
                    vocab.setdefault(fold_term(part.strip()), part.strip())
        if "/" in skill.name and "(" not in skill.name:
            for part in skill.name.split("/"):
                if len(part.strip()) > 1:
                    vocab.setdefault(fold_term(part.strip()), part.strip())
    for project in master.projects:
        for tech in project.tech:
            vocab.setdefault(fold_term(tech), tech)
        for tag in project.tags:
            if len(tag) >= 4:
                vocab.setdefault(fold_term(tag), tag)
    for term in master.unsupported_heads():
        vocab.setdefault(fold_term(term), term)
    return {k: v for k, v in vocab.items() if len(k) >= 2}


def _requirement_words(text: str) -> set[str]:
    stop = {"with", "and", "the", "for", "from", "into", "our", "your", "you", "are",
            "will", "have", "has", "who", "not", "but", "all", "that", "this", "their",
            "please", "note", "area", "position"}
    return {t for t in re.findall(r"[a-z]{4,}", text.lower()) if t not in stop}


# Clause openers that begin a NEW requirement when a posting's copy-paste has
# run several of them together on one line. Deliberately narrow and anchored on
# requirement-stating language, so ordinary prose is never chopped up.
_CLAUSE_OPENERS = (
    r"Understanding of", r"Knowledge (?:and experience|of)", r"Experience (?:with|in)",
    r"Familiarity with", r"Strong\s+(?:analytical|communication|written)",
    r"Proficiency (?:with|in)", r"Ability to", r"Demonstrated", r"Working knowledge",
    r"Know how to", r"Motivated to", r"Course\s*work", r"Exposure to", r"Hands[\s-]on",
)
_CLAUSE_SPLIT_RE = re.compile(
    r"(?<=[a-z)])\s+(?=(?:" + "|".join(_CLAUSE_OPENERS) + r")\b)")


def split_requirement_clauses(body: str, min_words: int = 12) -> list[str]:
    """Split a list line whose requirement clauses were concatenated.

    Only splits at a capitalized requirement-stating opener following a finished
    word, and only for a line long enough to plausibly hold several clauses, so
    a normal single requirement is returned untouched.
    """
    if len(body.split()) < min_words:
        return [body]
    parts = [part.strip(" .;,") for part in _CLAUSE_SPLIT_RE.split(body)]
    parts = [part for part in parts if part]
    return parts if len(parts) > 1 else [body]


def extract_jd_requirements(jd_text: str, master: MasterFacts) -> list[Requirement]:
    """Pull requirement statements out of a JD, with or without headings.

    Pass 1 takes list items, whether bulleted or merely indented: real postings
    often list requirements with no "Requirements:" heading at all. Pass 2 takes
    prose sentences using explicit requirement language. Pass 3 records prose
    that only names technologies as a role SIGNAL, reported but never scored.
    """
    vocab = jd_vocabulary(master)
    lines = jd_text.splitlines()

    def named_terms(text: str) -> tuple[str, ...]:
        found = [display for _folded, display in vocab.items()
                 if len(display) >= 2 and _mentions(text, _skill_needles(display))]
        return tuple(sorted(set(found), key=lambda t: (-len(t), t)))

    def importance_of(text: str, default: str) -> str:
        if any(re.search(p, text, re.IGNORECASE) for p in _PREFERRED_INLINE):
            return "preferred"
        if any(re.search(p, text, re.IGNORECASE) for p in _REQUIRED_INLINE):
            return "required"
        return default

    def ignorable(text: str) -> bool:
        return bool(_BENEFITS_LINE.search(text) or _MARKETING_LINE.search(text))

    def jd_named_terms(text: str) -> tuple[str, ...]:
        """Vocabulary terms plus JD-named technologies the master never claims.

        A named technology with no support must be visible to classification so
        it blocks its AND group instead of being silently ignored.
        """
        found = set(named_terms(text))
        for term in _JD_SIGNAL_TERMS:
            if len(term) >= 2 and _mentions(text, _skill_needles(term)):
                found.add(term)
        return tuple(sorted(found, key=lambda t: (-len(t), t)))

    candidates: list[Requirement] = []

    section, section_importance = "", "unclear"
    for index, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        heading = line.rstrip(":").strip().lower()
        if line.endswith(":") and len(heading.split()) <= 5:
            section, section_importance = heading, "unclear"
            for needle, level in _REQ_SECTIONS:
                if needle in heading:
                    section_importance = level
                    break
            continue
        bulleted = bool(re.match(r"^[-*•·]", line))
        # A long indented line is still a list item. The old length cap silently
        # discarded real requirements whose clauses had been concatenated onto
        # one physical line, so length now controls SPLITTING, never dropping.
        indented = bool(re.match(r"^\s{2,}\S", raw)) and len(line.split()) <= 60
        if not (bulleted or indented):
            continue
        body = line.lstrip("-*•· ").strip()
        if len(body.split()) < 2 or ignorable(body) or _UI_NAV_LINE.match(body):
            continue
        for clause in split_requirement_clauses(body):
            if len(clause.split()) < 2 or ignorable(clause) or _UI_NAV_LINE.match(clause):
                continue
            candidates.append(Requirement(
                clause, importance_of(clause, section_importance if section else "required"),
                jd_named_terms(clause), section or "list", kind=requirement_kind(clause),
                authority="hard", line=index))

    def prose_sentences():
        for index, raw in enumerate(lines):
            line = raw.strip()
            if not line or re.match(r"^[-*•·]", line) or re.match(r"^\s{2,}\S", raw):
                continue
            # Benefits/EEO text is boilerplate for a whole line, but marketing
            # copy is judged per sentence: one trailing "Learn more about our
            # team" must not discard the sentences naming real technologies.
            if _BENEFITS_LINE.search(line):
                continue
            for sentence in re.split(r"(?<=[.!?])\s+", line):
                sentence = sentence.strip()
                if len(sentence.split()) >= 4 and not ignorable(sentence):
                    yield index, sentence

    for index, sentence in prose_sentences():
        if any(re.search(p, sentence, re.IGNORECASE) for p in _HARD_LANGUAGE):
            candidates.append(Requirement(
                sentence, importance_of(sentence, "required"), named_terms(sentence),
                "prose", kind=requirement_kind(sentence), authority="hard", line=index))

    for index, sentence in prose_sentences():
        # A signal may cite technologies the master does not claim, so the JD's
        # own lexicon counts here even though it never reaches Skills or fit.
        terms = named_terms(sentence) or jd_signal_terms(sentence)
        if terms:
            candidates.append(Requirement(
                sentence, "unclear", terms, "prose", kind="capability",
                authority="signal", line=index))

    # An explicit list item wins over prose that repeats it.
    accepted: list[Requirement] = []
    for candidate in candidates:
        words = _requirement_words(candidate.text)
        duplicate = False
        for kept in accepted:
            other = _requirement_words(kept.text)
            if kept.kind != candidate.kind or not words or not other:
                continue
            if len(words & other) / min(len(words), len(other)) >= 0.5:
                duplicate = True
                break
        if not duplicate:
            accepted.append(candidate)

    accepted.sort(key=lambda r: (r.line, 0 if r.authority == "hard" else 1))
    return [dataclasses.replace(req, requirement_id=f"REQ-{index:03d}")
            for index, req in enumerate(accepted, start=1)]


def requirement_table(requirements: list[Requirement]) -> dict[str, Requirement]:
    return {req.requirement_id: req for req in requirements}


def scored_requirements(requirements: list[Requirement]) -> list[Requirement]:
    """Explicit capability/qualification requirements: the fit-score basis."""
    return [req for req in requirements if req.scored]


def manual_review_requirements(requirements: list[Requirement]) -> list[Requirement]:
    """Eligibility, logistics, conditions and subjective credentials.

    Reported for a human decision, never scored: nothing in the resume proves
    relocation, vaccination, work authorization or "academic excellence".
    """
    return [req for req in requirements
            if req.authority == "hard"
            and (req.kind not in FIT_SCORED_KINDS or req.subjective)]


def manual_review_reason(requirement: Requirement, master: MasterFacts) -> str:
    """Why a requirement needs a human, stated without misreporting the record.

    A subjective standard ("a history of academic excellence") cannot be settled
    deterministically, but the resume still carries real positive evidence for
    it. Saying that evidence is absent would be false, so the reason names what
    is present and attributes the uncertainty to the standard instead.
    """
    if requirement.subjective:
        present: list[str] = []
        gpa = re.search(r"GPA:?\s*([0-9.]+\s*/\s*[0-9.]+)", master.raw)
        if gpa:
            present.append(f"a recorded GPA of {' '.join(gpa.group(1).split())}")
        present.append("professional software engineering roles in Professional Experience")
        return ("the resume provides positive academic and professional evidence ("
                + "; ".join(present) + "), but the posting uses a subjective standard that "
                "cannot be resolved deterministically, so a human should judge it")
    return (f"{requirement.kind} requirement that the recorded facts cannot settle; "
            f"needs a human decision")


def role_signals(requirements: list[Requirement]) -> list[Requirement]:
    """Role prose naming technologies: supporting colour, not a requirement."""
    return [req for req in requirements if req.authority == "signal"]


def skill_requirements(requirements: list[Requirement]) -> list[Requirement]:
    """Requirements that describe capabilities or qualifications."""
    return [req for req in requirements if req.kind in FIT_SCORED_KINDS]


def jd_themes(requirements: list[Requirement], limit: int = 5) -> list[Requirement]:
    """The most important, most concrete requirements, for prompt focus."""
    weight = {"required": 0, "unclear": 1, "preferred": 2}
    ranked = sorted(enumerate(skill_requirements(requirements)),
                    key=lambda pair: (0 if pair[1].authority == "hard" else 1,
                                      weight.get(pair[1].importance, 3),
                                      -len(pair[1].terms), pair[0]))
    return [req for _index, req in ranked[:limit]]


@dataclass(frozen=True)
class ResumeEvidence:
    """What the FINAL rendered resume actually says.

    Classification reads this first, so a requirement is judged against the
    document that will be sent, not against evidence that never shipped.
    """

    experience_text: str = ""
    project_bullets: tuple[tuple[str, str], ...] = ()      # (project_id, bullet text)
    skills: tuple[str, ...] = ()

    @property
    def corpus(self) -> str:
        return " ".join([self.experience_text, " ".join(self.skills)]
                        + [text for _pid, text in self.project_bullets])


# Morphology only. These normalize word forms of the SAME concept and never
# equate two technologies.
CONCEPT_ALIASES = {
    "containerize": "container", "containerized": "container",
    "containerization": "container", "containers": "container", "container": "container",
    "test": "test", "tests": "test", "testing": "test", "tested": "test",
    "debug": "debug", "debugs": "debug", "debugging": "debug", "debugged": "debug",
    "deploy": "deploy", "deploys": "deploy", "deployment": "deploy",
    "deployments": "deploy", "deployed": "deploy",
    "review": "review", "reviews": "review", "reviewing": "review", "reviewed": "review",
    "monitor": "monitor", "monitors": "monitor", "monitoring": "monitor",
    "schema": "schema", "schemas": "schema",
    "concurrency": "concurrent", "concurrent": "concurrent", "concurrently": "concurrent",
    "collaborate": "collaborate", "collaboration": "collaborate",
    "collaborative": "collaborate",
    "team": "team", "teams": "team", "teammate": "team", "teammates": "team",
    "scale": "scale", "scaling": "scale", "scalable": "scale", "scaled": "scale",
    "optimize": "optimize", "optimization": "optimize", "optimizing": "optimize",
    "maintain": "maintain", "maintaining": "maintain", "maintenance": "maintain",
    "build": "build", "building": "build", "built": "build",
    "develop": "develop", "developing": "develop", "development": "develop",
    "environment": "environment", "environments": "environment",
    "application": "application", "applications": "application",
    "service": "service", "services": "service",
    "intern": "intern", "interns": "intern", "internship": "intern",
    "internships": "intern",
    "consistent": "consistent", "consistency": "consistent",
}

_CONCEPT_STOPWORDS = {
    "with", "and", "the", "for", "from", "into", "our", "your", "you", "are", "will",
    "have", "has", "who", "not", "but", "all", "experience", "work", "working", "strong",
    "other", "such", "that", "this", "their", "using", "used", "across", "within", "keep",
    "familiarity", "knowledge", "ability", "plus", "preferred", "required", "must",
    "including", "well", "good", "solid", "proven", "hands",
}


def _concepts(text: str) -> set[str]:
    """Content words, normalized by morphology only."""
    out = set()
    for raw_token in re.findall(r"[A-Za-z][A-Za-z0-9+#/.]{2,}", text.lower()):
        # Keep internal punctuation (node.js, ci/cd, c++) but drop edge
        # punctuation: "tests." must alias to "test".
        token = raw_token.strip("./,;:")
        if not token or token in _CONCEPT_STOPWORDS:
            continue
        canonical = CONCEPT_ALIASES.get(token)
        if canonical is None and len(token) > 5:
            for suffix in ("ing", "ed", "es", "s"):
                if token.endswith(suffix):
                    canonical = token[: -len(suffix)]
                    break
        out.add(canonical or token)
    return out


# Generic engineering concepts whose bare lexical presence proves nothing: a
# clinical "plate testing" or "review test results" is not software testing.
_GENERIC_CONCEPTS = {"test", "debug"}

_SOFTWARE_CONTEXT = (
    "software", "code", "codebase", "kernel", "api", "apis", "unit", "integration",
    "regression", "automated", "concurrency", "concurrent", "memory", "edge-case",
    "edge case", "load", "deadlock", "panic", "bug", "merge", "pull request", "module",
    "service", "system", "pipeline", "deployment", "production", "latency", "endpoint",
    "schema", "query", "backend", "server", "compiler", "runtime", "suite", "coverage",
)

_NON_SOFTWARE_CONTEXT = (
    "plate testing", "test results", "clinical", "diagnostic", "ishihara", "patient",
    "patients", "blood", "specimen", "medical test",
)


def _generic_concept_in_software_context(text: str, canonical: str) -> bool:
    """Whether a generic concept appears in a software/system sentence.

    Scoped to the sentence that carries the concept, because a context word
    somewhere else in the resume would otherwise wave everything through.
    """
    for sentence in re.split(r"(?<=[.;:])\s+", text):
        if canonical not in _concepts(sentence):
            continue
        lowered = sentence.lower()
        if any(phrase in lowered for phrase in _NON_SOFTWARE_CONTEXT):
            continue
        if any(phrase in lowered for phrase in _SOFTWARE_CONTEXT):
            return True
    return False


def _term_present(text: str, term: str) -> bool:
    """Literal match, or the same concept in a different word form.

    Morphology only: "testing" finds a shipped "tests", but no alias ever
    equates two technologies. A generic concept additionally has to sit in a
    software/system context to count.
    """
    if not text:
        return False
    canonical = CONCEPT_ALIASES.get(term.strip().lower())
    if canonical in _GENERIC_CONCEPTS:
        return _generic_concept_in_software_context(text, canonical)
    if _mentions(text, _skill_needles(term)):
        return True
    return bool(canonical) and canonical in _concepts(text)


# Wording that makes an enumerated list alternatives rather than a checklist.
_ANY_OF = (r"\(any\)", r"\bany\s+(?:one|of)\b", r"\bone\s+or\s+more\b",
           r"\bat\s+least\s+one\b", r"\bone\s+of\b", r"\beither\b")


def any_of_list(text: str) -> bool:
    """True when the line says its enumeration is satisfied by one member."""
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in _ANY_OF)


def requirement_groups(requirement: Requirement) -> list[tuple[str, ...]]:
    """Split a compound requirement into AND groups of OR alternatives.

    "PostgreSQL or MySQL, REST API development, Docker, Linux, and Git" becomes
    (PostgreSQL|MySQL) AND (Docker) AND (Linux) AND (Git), so a single matched
    term can no longer satisfy the whole line.
    """
    if not requirement.terms:
        return []
    # "Python, Java, C++, or JavaScript (any)" and "Course work one or more: ..."
    # are satisfied by a single member, so the whole enumeration is ONE group.
    if any_of_list(requirement.text):
        return [tuple(requirement.terms)]
    groups: list[tuple[str, ...]] = []
    claimed: set[str] = set()
    for segment in re.split(r"[,;]|\band\b", requirement.text, flags=re.IGNORECASE):
        present = [t for t in requirement.terms if _term_present(segment, t)]
        if not present:
            continue
        if re.search(r"\bor\b", segment, re.IGNORECASE) and len(present) > 1:
            groups.append(tuple(present))            # alternatives: any one suffices
        else:
            groups.extend((term,) for term in present)
        claimed.update(present)
    groups.extend((term,) for term in requirement.terms if term not in claimed)
    return groups


@dataclass
class _GroupVerdict:
    terms: tuple[str, ...]
    status: str                 # resume | master_only | unsupported
    term: str = ""
    source: str = "none"
    owner: str = ""
    blocked: str = ""


def _evaluate_group(group: tuple[str, ...], master: MasterFacts,
                    evidence: ResumeEvidence) -> _GroupVerdict:
    """One AND group. An unsupported technology never satisfies its group."""
    unsupported = {fold_term(t): t for t in master.unsupported_heads()}
    blocked = ""
    for term in group:
        if fold_term(term) in unsupported:
            blocked = blocked or term
            continue
        bare = re.sub(r"\s*\([^)]*\)", "", term).strip() or term
        if _term_present(evidence.experience_text, term) or \
                _term_present(evidence.experience_text, bare):
            return _GroupVerdict(group, "resume", term, "experience")
        for project_id, text in evidence.project_bullets:
            if _term_present(text, term) or _term_present(text, bare):
                return _GroupVerdict(group, "resume", term, "project", project_id)
        if any(fold_term(term) == fold_term(skill) or _term_present(skill, term)
               for skill in evidence.skills):
            return _GroupVerdict(group, "resume", term, "skills")
        if master.supported_by_experience(bare) or master.supported_anywhere(bare):
            return _GroupVerdict(group, "master_only", term, "experience")
    return _GroupVerdict(group, "unsupported", group[0] if group else "", "none",
                         blocked=blocked)


_DEGREE_ASK = re.compile(r"\b(?:bachelor\w*|BS/BA|B\.?S\.?|master'?s|degree)\b",
                         re.IGNORECASE)

# Degrees the record actually contains, most advanced first.
_INSTITUTION_WORD = (r"University|Institution|Institute|College|School|Society|Vivekanand|"
                     r"Buffalo|Mumbai|GPA|August|May|December|January|Expected|Role")

_DEGREE_RE = re.compile(
    r"(?P<level>Bachelor(?:'s)?(?:\s+of\s+[A-Za-z]+)?|Master(?:'s)?(?:\s+of\s+[A-Za-z]+)?|"
    r"\bB\.?E\.?\b|\bB\.?S\.?\b|\bM\.?S\.?\b|Ph\.?D)"
    r"(?:[^\S\n]*(?:in|,)[^\S\n]*(?P<field>(?!(?:" + _INSTITUTION_WORD + r")\b)"
    r"[A-Z][A-Za-z]*(?:[^\S\n]+(?!(?:" + _INSTITUTION_WORD + r")\b)[A-Za-z]+){0,3}))?")


def _degree_tier(level: str) -> str:
    """Canonical degree tier, so "Master's" and "Master" are not two degrees."""
    folded = fold_term(level)
    for tier, markers in (("phd", ("phd",)), ("master", ("master", "ms", "meng")),
                          ("bachelor", ("bachelor", "bs", "be", "btech"))):
        if any(folded.startswith(m) for m in markers):
            return tier
    return folded


def _degree_field_terms(text: str) -> list[str]:
    """The majors a posting enumerates for its degree requirement."""
    tail = re.split(r"\bdegree\b\s*(?:in|:)?", text, maxsplit=1, flags=re.IGNORECASE)
    if len(tail) < 2:
        return []
    fields = [f.strip(" .;") for f in re.split(r",|\bor\b", tail[1], flags=re.IGNORECASE)]
    return [f for f in fields if 2 < len(f) < 48]


def degree_evidence(requirement: Requirement, master: MasterFacts,
                    evidence: ResumeEvidence) -> "RequirementMatch | None":
    """Classify a degree requirement against the degrees on the resume.

    Returns None when the resume names no degree at all, leaving the normal path
    to handle it. Otherwise the degree's presence is stated as fact and only the
    FIELD can be the shortfall, so the limitation is accurate instead of
    claiming a degree the candidate holds is missing.
    """
    corpus = " ".join([evidence.experience_text, master.raw])
    found: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for match in _DEGREE_RE.finditer(corpus):
        pair = (match.group("level").strip(), (match.group("field") or "").strip())
        key = (fold_term(pair[0]), fold_term(pair[1]))
        if key in seen:
            continue
        seen.add(key)
        found.append(pair)
    with_field = {_degree_tier(level) for level, field in found if field}
    found = [(level, field) for level, field in found
             if field or _degree_tier(level) not in with_field]
    undergrad = next(((lvl, fld) for lvl, fld in found
                      if re.match(r"bachelor|b\.?e\.?|b\.?s\.?", lvl, re.IGNORECASE)), None)
    graduate = next(((lvl, fld) for lvl, fld in found
                     if re.match(r"master|m\.?s\.?|ph", lvl, re.IGNORECASE)), None)
    held = undergrad or graduate
    if held is None:
        return None

    wants_undergrad = bool(re.search(r"bachelor|BS/BA|\bB\.?S\.?\b", requirement.text,
                                     re.IGNORECASE))
    level, field = (undergrad if wants_undergrad and undergrad else held)
    shown = f"{level}{' in ' + field if field else ''}"
    others = [f"{lvl}{' in ' + fld if fld else ''}" for lvl, fld in found
              if (lvl, fld) != (level, field)]
    evidence_text = shown + (f"; also {', '.join(others[:2])}" if others else "")

    wanted_fields = _degree_field_terms(requirement.text)
    if not wanted_fields or not field:
        return RequirementMatch(requirement, "strong_match", evidence_text, "education",
                                term=level)
    if any(fold_term(f) == fold_term(field) for f in wanted_fields):
        return RequirementMatch(requirement, "strong_match", evidence_text, "education",
                                term=field)
    return RequirementMatch(
        requirement, "partial_match", evidence_text, "education",
        limitation=(f"the degree is held, in {field}, but the posting explicitly lists "
                    f"{', '.join(wanted_fields)}"),
        term=field)


def classify_requirement(requirement: Requirement, master: MasterFacts,
                         evidence: ResumeEvidence) -> RequirementMatch:
    """Judge one requirement against the final resume, group by group.

    Every AND group must be supported for a strong match, so one matched term
    in a compound line is not enough. Docker cannot answer Kubernetes and C
    cannot answer C++.
    """
    groups = requirement_groups(requirement)
    if groups:
        verdicts = [_evaluate_group(group, master, evidence) for group in groups]
        on_resume = [v for v in verdicts if v.status == "resume"]
        master_only = [v for v in verdicts if v.status == "master_only"]
        missing = [v for v in verdicts if v.status == "unsupported"]

        def describe(verdict: _GroupVerdict) -> str:
            if verdict.status == "master_only":
                return (f"{verdict.term} (supported by the master, not surfaced on this "
                        f"resume)")
            where = {"experience": "Professional Experience",
                     "project": f"project {verdict.owner}",
                     "skills": "Technical Skills"}.get(verdict.source, "the master")
            return f"{verdict.term} ({where})"

        source = next((v.source for v in verdicts
                       if v.source in ("experience", "project", "skills")), "skills")
        if not missing and not master_only:
            return RequirementMatch(requirement, "strong_match",
                                    "; ".join(describe(v) for v in on_resume),
                                    source, term=on_resume[0].term if on_resume else "")
        if on_resume or master_only:
            clauses = []
            unmet = []
            for verdict in missing:
                if verdict.blocked:
                    unmet.append(f"{verdict.blocked}, which the master explicitly does not "
                                 f"support")
                else:
                    unmet.append(" or ".join(verdict.terms))
            if unmet:
                clauses.append("the posting also asks for " + "; ".join(unmet))
            if master_only:
                names = ", ".join(v.term for v in master_only)
                verb = "is" if len(master_only) == 1 else "are"
                clauses.append(f"{names} {verb} supported by the master but not surfaced on "
                               f"this resume")
            shown = "; ".join(describe(v) for v in (on_resume or master_only))
            return RequirementMatch(requirement, "partial_match",
                                    shown or "related evidence exists", source,
                                    limitation="; ".join(clauses),
                                    term=(missing[0].blocked or missing[0].term)
                                         if missing else "")
        first = missing[0]
        return RequirementMatch(
            requirement, "unsupported", "", "none",
            limitation=(f"the posting asks for {first.blocked}, which the master explicitly "
                        f"does not support" if first.blocked
                        else "no supporting evidence in the final resume or the master"),
            term=first.blocked or first.term)

    # A degree requirement is settled by the degrees the resume actually lists,
    # never by concept overlap. The previous path produced a bare "unsupported"
    # for a candidate who does hold a bachelor's degree, which is false.
    if requirement.kind == "qualification" and _DEGREE_ASK.search(requirement.text):
        degree = degree_evidence(requirement, master, evidence)
        if degree is not None:
            return degree

    # No named technology: judge the concept against the final resume text.
    wanted = _concepts(requirement.text)
    if not wanted:
        return RequirementMatch(requirement, "unsupported", "", "none",
                                limitation="no supporting evidence in the final resume")
    sources = [("experience", "Professional Experience", evidence.experience_text)]
    sources += [("project", f"project {pid}", text) for pid, text in evidence.project_bullets]
    sources += [("skills", "Technical Skills", " ".join(evidence.skills))]
    best = (set(), "none", "the final resume", "")
    for source, label, text in sources:
        overlap = wanted & _concepts(text)
        overlap = {concept for concept in overlap
                   if concept not in _GENERIC_CONCEPTS
                   or _generic_concept_in_software_context(text, concept)}
        if len(overlap) > len(best[0]):
            best = (overlap, source, label, text)
    overlap, source, label, _text = best
    ratio = len(overlap) / max(len(wanted), 1)
    if ratio >= 0.6:
        return RequirementMatch(requirement, "strong_match",
                                f"{label} covers {', '.join(sorted(overlap)[:4])}", source)
    if overlap:
        return RequirementMatch(
            requirement, "partial_match",
            f"{label} covers {', '.join(sorted(overlap)[:4])}", source,
            limitation="the requirement is only partly evidenced by the shipped resume")
    return RequirementMatch(requirement, "unsupported", "", "none",
                            limitation="no supporting evidence in the final resume")


_GRAD_WINDOW_RE = re.compile(
    r"graduat\w*\s+(?:between|from)\s+(?P<lo>[A-Za-z]+\s+\d{4})\s+(?:and|to|through|-)\s+"
    r"(?P<hi>[A-Za-z]+\s+\d{4})", re.IGNORECASE)
_GRAD_BY_RE = re.compile(r"graduat\w*\s+(?:by|before|no later than)\s+"
                         r"(?P<hi>[A-Za-z]+\s+\d{4})", re.IGNORECASE)


def _month_year(text: str) -> tuple[int, int] | None:
    year = re.search(r"(19|20)\d{2}", text)
    month = next((n for name, n in _MONTHS.items() if name in text.lower()), 0)
    return (int(year.group(0)), month) if year else None


def graduation_requirement(jd_text: str) -> dict | None:
    """The JD's graduation window, when it states one."""
    window = _GRAD_WINDOW_RE.search(jd_text)
    if window:
        return {"low": _month_year(window.group("lo")), "high": _month_year(window.group("hi")),
                "text": re.sub(r"\s+", " ", window.group(0)).strip()}
    by = _GRAD_BY_RE.search(jd_text)
    if by:
        return {"low": None, "high": _month_year(by.group("hi")),
                "text": re.sub(r"\s+", " ", by.group(0)).strip()}
    year = re.search(r"class of ((?:19|20)\d{2})", jd_text, re.IGNORECASE)
    if year:
        return {"low": (int(year.group(1)) - 1, 1), "high": (int(year.group(1)), 12),
                "text": f"class of {year.group(1)}"}
    return None


_WORK_AUTH_PATTERNS = (r"\bvisa\b", r"\bsponsor\w*\b", r"\bwork authoriz\w*\b",
                       r"\bauthoriz\w+ to work\b", r"\beligible to work\b",
                       r"\bcitizen\w*\b", r"\bsecurity clearance\b")


def deterministic_assessment(requirements: list[Requirement], master: MasterFacts,
                             evidence: ResumeEvidence, jd_text: str = "") -> dict:
    """The authoritative verdict for every requirement, owned entirely by Python.

    JD -> extraction -> final resume -> this classification -> IMMUTABLE verdicts.
    A model may explain or summarize these, never change them: the buckets, the
    evidence, the limitations and the eligibility status are all decided here,
    against the resume that will actually be sent.

    `jd_text` is required for eligibility: the graduation window lives in the
    posting text, so omitting it would report a satisfied window as though the
    posting stated no requirement at all.
    """
    scored = scored_requirements(requirements)
    strong: list[dict] = []
    partial: list[dict] = []
    gaps: list[dict] = []

    for requirement in scored:
        match = classify_requirement(requirement, master, evidence)
        if match.verdict == "strong_match":
            strong.append({"requirement_id": requirement.requirement_id,
                           "evidence": match.evidence,
                           "source": match.source if match.source != "none" else "skills"})
        elif match.verdict == "partial_match":
            partial.append({"requirement_id": requirement.requirement_id,
                            "evidence": match.evidence,
                            "limitation": match.limitation
                                          or "the requirement is only partly evidenced",
                            "source": match.source if match.source != "none" else "skills"})
        else:
            detail = (f"the posting asks for {match.term}; it is not supported"
                      if match.term else match.limitation)
            gaps.append({"requirement_id": requirement.requirement_id,
                         "importance": requirement.importance, "status": "unsupported",
                         "evidence": None, "detail": detail})

    manual = [{"requirement_id": requirement.requirement_id,
               "reason": manual_review_reason(requirement, master)}
              for requirement in manual_review_requirements(requirements)]

    status, details, flags = eligibility_assessment(requirements, master, jd_text)
    return {"strong_matches": strong, "partial_matches": partial, "gaps": gaps,
            "manual_review": manual,
            "eligibility": {"status": status, "details": details},
            "eligibility_flags": flags}


def verdict_index(verdicts: dict) -> dict[str, str]:
    """requirement_id -> the bucket Python placed it in."""
    index: dict[str, str] = {}
    for bucket in ("strong_matches", "partial_matches", "gaps", "manual_review"):
        for entry in verdicts.get(bucket) or []:
            index[entry["requirement_id"]] = bucket
    return index


def verdict_block(verdicts: dict, table: dict[str, Requirement]) -> str:
    """The immutable verdicts, rendered for a prompt."""
    label = {"strong_matches": "STRONG", "partial_matches": "PARTIAL",
             "gaps": "UNSUPPORTED", "manual_review": "MANUAL REVIEW"}
    lines: list[str] = []
    for bucket, name in label.items():
        for entry in verdicts.get(bucket) or []:
            requirement = table.get(entry["requirement_id"])
            text = requirement.original_text[:110] if requirement else ""
            detail = (entry.get("evidence") or entry.get("detail")
                      or entry.get("reason") or "")
            extra = f" | limitation: {entry['limitation']}" if entry.get("limitation") else ""
            lines.append(f"  {entry['requirement_id']} [{name}] {text}\n"
                         f"      basis: {detail}{extra}")
    return "\n".join(lines) or "  (none)"


def eligibility_assessment(requirements: list[Requirement], master: MasterFacts,
                           jd_text: str) -> tuple[str, list[str], list[str]]:
    """Deterministic eligibility verdict: (status, details, risk flags).

    Never concludes more than the evidence supports. A work-authorization
    requirement against an OPT-based availability is genuinely unresolved, so
    it reports `uncertain` and asks for manual review rather than declaring the
    candidate either eligible or ineligible.
    """
    details: list[str] = []
    flags: list[str] = []
    status = "meets"
    # `not_applicable` is returned below when the posting states neither a
    # graduation window nor a work-authorization requirement.

    window = graduation_requirement(jd_text)
    expected = expected_graduation(master)
    if window and expected:
        low, high = window.get("low"), window.get("high")
        inside = (low is None or expected >= low) and (high is None or expected <= high)
        month = _MONTH_LABELS.get(expected[1], "")
        details.append(f"graduates {month} {expected[0]}, which "
                       f"{'falls inside' if inside else 'falls outside'} the posting's "
                       f"window ({window['text']})")
        if not inside:
            status = "does_not_meet"
            flags.append(f"graduation date is outside the posting's window ({window['text']})")
    elif window:
        status = "uncertain"
        details.append(f"the posting requires {window['text']} and the graduation date "
                       f"could not be confirmed")

    authorization = [req for req in requirements
                     if req.kind == "eligibility"
                     and any(re.search(p, req.text, re.IGNORECASE)
                             for p in _WORK_AUTH_PATTERNS)]
    for req in authorization:
        status = "uncertain" if status != "does_not_meet" else status
        details.append(f"{req.requirement_id} states: {req.original_text}")
        for fact in master.standing_facts:
            if "opt" in fact.lower() or "authoriz" in fact.lower():
                details.append(f"known candidate fact: {fact}")
        details.append("work authorization cannot be resolved from the recorded facts; "
                       "this needs manual review and no conclusion is drawn either way")
        flags.append(f"MANUAL REVIEW ({req.requirement_id}): work-authorization requirement "
                     f"versus the candidate's recorded OPT availability")

    if not window and not authorization:
        # Nothing to satisfy: "meets" would imply a requirement was cleared.
        return "not_applicable", [], flags
    return status, details, flags


_MONTH_LABELS = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
                 7: "July", 8: "August", 9: "September", 10: "October", 11: "November",
                 12: "December"}


def expected_graduation(master: MasterFacts) -> tuple[int, int] | None:
    """The candidate's graduation date, from the master's standing facts."""
    for fact in master.standing_facts:
        if "graduation" in fact.lower():
            return _month_year(fact)
    return None


# ==================================================================== verbs


def opening_verb(text: str) -> str:
    match = re.match(r"\s*([A-Za-z]+)", text)
    return match.group(1).lower() if match else ""


def verb_usage(experience_latex_items: list[str], project_bullets: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in experience_latex_items:
        verb = opening_verb(latex_to_plain(item))
        if verb:
            counts[verb] = counts.get(verb, 0) + 1
    for bullet in project_bullets:
        verb = opening_verb(bullet)
        if verb:
            counts[verb] = counts.get(verb, 0) + 1
    return counts


def exhausted_verbs(counts: dict[str, int], limit: int) -> list[str]:
    return sorted(v for v, n in counts.items() if n >= limit)


def count_opening_verbs(bullets: list[str]) -> dict[str, int]:
    """Count the first meaningful action verb of each rendered bullet."""
    counts: dict[str, int] = {}
    for bullet in bullets:
        verb = opening_verb(normalize_plain(bullet))
        if verb:
            counts[verb] = counts.get(verb, 0) + 1
    return counts


def verb_violations(counts: dict[str, int], limit: int) -> list[tuple[str, int]]:
    """Verbs used more than the spreadsheet's per-resume limit."""
    return sorted(((verb, uses) for verb, uses in counts.items() if uses > limit),
                  key=lambda row: (-row[1], row[0]))


def available_verbs(counts: dict[str, int], limit: int,
                    preferred: tuple[str, ...]) -> list[str]:
    """Preferred verbs that still have room under the limit."""
    return [verb for verb in preferred
            if counts.get(verb.lower(), 0) < limit]
