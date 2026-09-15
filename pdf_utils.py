"""Compile the resume and measure the REAL PDF.

This module never estimates layout. `pdftotext -bbox-layout` gives per-word
geometry, so line counts and final-line fill are read off the compiled artifact.
It is deliberately a leaf module: it takes paths and plain numbers, so it can be
used (and debugged) without importing the rest of the pipeline.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

_NS = "{http://www.w3.org/1999/xhtml}"

# poppler renders the itemize bullet of this template as U+0088; other builds
# emit a real bullet character. Either starts a new bullet.
BULLET_GLYPHS = ("\x88", "•", "●", "·")

# Section headings as they appear in the rendered PDF.
SECTION_HEADINGS = (
    "EXPERIENCE", "EDUCATION", "ACADEMIC PROJECTS", "TECHNICAL SKILLS",
    "CERTIFICATIONS, AWARDS & PUBLICATIONS",
)


# --------------------------------------------------------------------- build


@dataclass
class BuildResult:
    ok: bool
    pdf_path: Path | None
    log: str
    errors: list[str]
    returncode: int

    def summary(self) -> str:
        if self.ok:
            return f"pdflatex ok -> {self.pdf_path.name}"
        return f"pdflatex FAILED (rc={self.returncode}): {'; '.join(self.errors[:3]) or 'see log'}"


_LATEX_ERROR_RE = re.compile(r"^! .*$", re.MULTILINE)


def tools_available() -> tuple[bool, bool]:
    return shutil.which("pdflatex") is not None, shutil.which("pdftotext") is not None


def compile_pdf(tex_path: Path, workdir: Path | None = None, passes: int = 2) -> BuildResult:
    """Compile with pdflatex. Two passes settle `\\hfill` positioning."""
    if shutil.which("pdflatex") is None:
        return BuildResult(False, None, "", ["pdflatex is not installed"], -1)
    workdir = workdir or tex_path.parent
    result = None
    for _ in range(max(1, passes)):
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
             "-output-directory", str(workdir), str(tex_path)],
            capture_output=True, text=True, cwd=str(workdir), timeout=180,
        )
    log_path = workdir / (tex_path.stem + ".log")
    log = log_path.read_text(errors="replace") if log_path.exists() else (result.stdout if result else "")
    pdf_path = workdir / (tex_path.stem + ".pdf")
    errors = [line.strip() for line in _LATEX_ERROR_RE.findall(log)]
    ok = bool(result) and result.returncode == 0 and pdf_path.exists()
    return BuildResult(ok, pdf_path if ok else None, log, errors,
                       result.returncode if result else -1)


def cleanup_aux(workdir: Path, stem: str = "resume") -> None:
    for suffix in (".aux", ".out", ".log"):
        (workdir / f"{stem}{suffix}").unlink(missing_ok=True)


# ------------------------------------------------------------------ geometry


@dataclass
class RenderedLine:
    page: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    text: str

    @property
    def height(self) -> float:
        return max(self.y_max - self.y_min, 0.01)


# Two fragments belong to the same visual line when their bottom edges (which
# track the baseline) are within this fraction of the shorter fragment's height.
# Baseline proximity beats box overlap: a tall bold heading's box would
# otherwise swallow the short line beside it.
_BASELINE_TOLERANCE = 0.35


def extract_lines(pdf_path: Path) -> list[RenderedLine]:
    """True visual lines from the compiled PDF.

    poppler emits one `<line>` per glyph run, so a single rendered line can
    arrive as several fragments, and right-aligned `\\hfill` text (a date) can
    arrive out of reading order. Fragments are regrouped by baseline so a
    measured "line" is what a reader actually sees.
    """
    raw = subprocess.run(["pdftotext", "-bbox-layout", str(pdf_path), "-"],
                         capture_output=True, text=True, timeout=60).stdout
    if not raw.strip():
        return []
    root = ET.fromstring(raw)
    out: list[RenderedLine] = []
    for page_index, page in enumerate(root.iter(f"{_NS}page"), start=1):
        fragments: list[RenderedLine] = []
        for line in page.iter(f"{_NS}line"):
            words = list(line.iter(f"{_NS}word"))
            if not words:
                continue
            ordered = sorted(words, key=lambda w: float(w.get("xMin", 0)))
            fragments.append(RenderedLine(
                page=page_index,
                x_min=min(float(w.get("xMin", 0)) for w in words),
                x_max=max(float(w.get("xMax", 0)) for w in words),
                y_min=min(float(w.get("yMin", 0)) for w in words),
                y_max=max(float(w.get("yMax", 0)) for w in words),
                text=" ".join((w.text or "") for w in ordered),
            ))
        out.extend(_merge_visual_lines(fragments, page_index))
    return out


def _merge_visual_lines(fragments: list[RenderedLine], page: int) -> list[RenderedLine]:
    groups: list[list[RenderedLine]] = []
    for fragment in sorted(fragments, key=lambda f: (f.y_max, f.x_min)):
        for group in groups:
            baseline = max(f.y_max for f in group)
            shortest = min(min(f.height for f in group), fragment.height)
            if abs(baseline - fragment.y_max) <= _BASELINE_TOLERANCE * shortest:
                group.append(fragment)
                break
        else:
            groups.append([fragment])

    merged = [
        RenderedLine(
            page=page,
            x_min=min(f.x_min for f in group),
            x_max=max(f.x_max for f in group),
            y_min=min(f.y_min for f in group),
            y_max=max(f.y_max for f in group),
            text=" ".join(f.text for f in sorted(group, key=lambda f: f.x_min)),
        )
        for group in groups
    ]
    merged.sort(key=lambda line: (line.y_max, line.x_min))
    return merged


def extract_text(pdf_path: Path, layout: bool = True) -> str:
    args = ["pdftotext"] + (["-layout"] if layout else []) + [str(pdf_path), "-"]
    return subprocess.run(args, capture_output=True, text=True, timeout=60).stdout


def page_count(pdf_path: Path) -> int:
    raw = subprocess.run(["pdftotext", "-bbox", str(pdf_path), "-"],
                         capture_output=True, text=True, timeout=60).stdout
    return max(1, raw.count("<page "))


def body_right_edge(lines: list[RenderedLine]) -> float:
    return max((line.x_max for line in lines), default=0.0)


# ------------------------------------------------------- sections and bullets


def _fold(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _is_heading(line: RenderedLine) -> str | None:
    folded = _fold(line.text)
    for heading in SECTION_HEADINGS:
        if folded.startswith(_fold(heading)):
            return heading
    return None


def section_order_in_pdf(lines: list[RenderedLine]) -> list[str]:
    """Section headings in the order the reader meets them."""
    order: list[str] = []
    for line in lines:
        heading = _is_heading(line)
        if heading and heading not in order:
            order.append(heading)
    return order


def section_lines(lines: list[RenderedLine], heading: str) -> list[RenderedLine]:
    """Every rendered line belonging to one section, heading excluded."""
    collected: list[RenderedLine] = []
    inside = False
    for line in lines:
        found = _is_heading(line)
        if found:
            inside = _fold(found) == _fold(heading)
            continue
        if inside:
            collected.append(line)
    return collected


def starts_bullet(text: str) -> bool:
    return text.lstrip().startswith(BULLET_GLYPHS)


def strip_bullet(text: str) -> str:
    stripped = text.lstrip()
    for glyph in BULLET_GLYPHS:
        if stripped.startswith(glyph):
            return stripped[len(glyph):].strip()
    return stripped


@dataclass
class BulletRender:
    """One rendered bullet, as it actually appears on the page."""

    text: str                 # joined visual lines, bullet glyph removed
    lines: int
    fill_pct: float           # how much of the available width its last line uses
    index_in_section: int

    def verdict(self, orphan_max: float, acceptable_max: float, ideal_max: float) -> str:
        if self.lines <= 1:
            return "single_line"
        if self.fill_pct < orphan_max:
            return "hard_orphan"
        if self.fill_pct <= acceptable_max:
            return "acceptable"
        if self.fill_pct <= ideal_max:
            return "ideal"
        return "healthy"


def extract_bullets(lines: list[RenderedLine], heading: str,
                    right_edge: float | None = None) -> list[BulletRender]:
    """Bullets of one section, with measured line count and final-line fill."""
    section = section_lines(lines, heading)
    if not section:
        return []
    right = right_edge if right_edge is not None else body_right_edge(lines)

    groups: list[list[RenderedLine]] = []
    bullet_indent: float | None = None
    for line in section:
        if starts_bullet(line.text):
            groups.append([line])
            bullet_indent = line.x_min if bullet_indent is None else min(bullet_indent, line.x_min)
        elif groups and bullet_indent is not None and line.x_min > bullet_indent + 0.5:
            # Indented past the bullet glyph: a wrapped continuation line.
            groups[-1].append(line)
        # anything else (a role/project header at the margin) closes the bullet

    bullets: list[BulletRender] = []
    for index, group in enumerate(groups):
        indent = min(line.x_min for line in group[1:]) if len(group) > 1 else group[0].x_min
        available = max(right - indent, 1.0)
        last = group[-1]
        fill = max(0.0, min(100.0, 100.0 * (last.x_max - indent) / available))
        text = " ".join([strip_bullet(group[0].text)] + [line.text.strip() for line in group[1:]])
        bullets.append(BulletRender(text=re.sub(r"\s+", " ", text).strip(),
                                    lines=len(group), fill_pct=round(fill, 1),
                                    index_in_section=index))
    return bullets


def project_blocks(lines: list[RenderedLine], heading: str,
                   right_edge: float | None = None) -> list[tuple[str, list[BulletRender]]]:
    """Each header in a section paired with the bullets rendered beneath it.

    `extract_bullets` and `extract_non_bullet_lines` each walk the section on
    their own, so neither can say which bullet belongs to which header. This
    groups them in one pass, which is what proves a project kept the bullet
    allocation its relevance rank earned, no matter where it is displayed.
    """
    section = section_lines(lines, heading)
    if not section:
        return []
    right = right_edge if right_edge is not None else body_right_edge(lines)

    blocks: list[tuple[str, list[list[RenderedLine]]]] = []
    bullet_indent: float | None = None
    for line in section:
        if starts_bullet(line.text):
            bullet_indent = line.x_min if bullet_indent is None else min(bullet_indent, line.x_min)
            if blocks:
                blocks[-1][1].append([line])
            continue
        if bullet_indent is not None and line.x_min > bullet_indent + 0.5:
            if blocks and blocks[-1][1]:
                blocks[-1][1][-1].append(line)       # wrapped bullet text
            continue
        if line.text.strip():
            blocks.append((re.sub(r"\s+", " ", line.text).strip(), []))

    grouped: list[tuple[str, list[BulletRender]]] = []
    for header, groups in blocks:
        rendered = []
        for index, group in enumerate(groups):
            indent = (min(line.x_min for line in group[1:]) if len(group) > 1
                      else group[0].x_min)
            available = max(right - indent, 1.0)
            last = group[-1]
            fill = max(0.0, min(100.0, 100.0 * (last.x_max - indent) / available))
            text = " ".join([strip_bullet(group[0].text)]
                            + [line.text.strip() for line in group[1:]])
            rendered.append(BulletRender(text=re.sub(r"\s+", " ", text).strip(),
                                         lines=len(group), fill_pct=round(fill, 1),
                                         index_in_section=index))
        grouped.append((header, rendered))
    return grouped


def extract_non_bullet_lines(lines: list[RenderedLine], heading: str) -> list[str]:
    """Section lines that are not bullets or continuations - i.e. the headers."""
    section = section_lines(lines, heading)
    headers: list[str] = []
    bullet_indent: float | None = None
    for line in section:
        if starts_bullet(line.text):
            bullet_indent = line.x_min if bullet_indent is None else min(bullet_indent, line.x_min)
            continue
        if bullet_indent is not None and line.x_min > bullet_indent + 0.5:
            continue                      # wrapped bullet text
        if line.text.strip():
            headers.append(re.sub(r"\s+", " ", line.text).strip())
    return headers


# --------------------------------------------------------- technical skills


@dataclass
class CategoryMetric:
    label: str
    lines: int
    last_line_fill: float


def measure_skill_categories(lines: list[RenderedLine], labels: list[str],
                             right_edge: float | None = None) -> list[CategoryMetric]:
    """Rendered line count and final-line fill for each Technical Skills category.

    `labels` are the emitted `"Category: a, b, c"` strings in order. A category
    owns its own label line plus every continuation line up to the next label.
    """
    if not lines or not labels:
        return []
    right = right_edge if right_edge is not None else body_right_edge(lines)
    section = section_lines(lines, "TECHNICAL SKILLS")
    if not section:
        return []

    starts: list[int] = []
    cursor = 0
    for label in labels:
        needle = _fold(label)[:18]
        index = next((i for i in range(cursor, len(section))
                      if _fold(section[i].text).startswith(needle)), None)
        if index is None:
            return []
        starts.append(index)
        cursor = index + 1

    metrics: list[CategoryMetric] = []
    for position, label in enumerate(labels):
        start = starts[position]
        end = starts[position + 1] if position + 1 < len(starts) else len(section)
        group = section[start:end]
        indent = min(line.x_min for line in group[1:]) if len(group) > 1 else group[0].x_min
        available = max(right - indent, 1.0)
        fill = max(0.0, min(100.0, 100.0 * (group[-1].x_max - indent) / available))
        metrics.append(CategoryMetric(label=label.split(":", 1)[0].strip(),
                                      lines=len(group), last_line_fill=round(fill, 1)))
    return metrics


def total_skill_lines(metrics: list[CategoryMetric]) -> int:
    return sum(m.lines for m in metrics)


# ----------------------------------------------------------------- scanning


_ARTIFACTS = (
    ("�", "unicode replacement character"),
    ("[AUTO:", "unreplaced template placeholder"),
    ("\\textbackslash", "escaped backslash leaked into output"),
    ("%% ==== BLOCK", "template block marker leaked into output"),
)


def extraction_artifacts(text: str) -> list[str]:
    return [label for needle, label in _ARTIFACTS if needle in text]


def find_terms(text: str, terms: list[str], masked: list[str] | None = None) -> list[tuple[str, str]]:
    """Scan rendered text for forbidden terms, masking approved proper nouns first.

    Masking stops an employer name such as "Go Digital Technology Consulting"
    from tripping the unsupported term "Go".
    """
    scannable = text
    for noun in sorted(masked or [], key=len, reverse=True):
        if noun and noun in scannable:
            scannable = scannable.replace(noun, " " * len(noun))
    hits: list[tuple[str, str]] = []
    for term in terms:
        if not term:
            continue
        found = re.search(r"(?<![A-Za-z0-9+#])" + re.escape(term) + r"(?![A-Za-z0-9+#])",
                          scannable, re.IGNORECASE)
        if found:
            start = scannable.rfind("\n", 0, found.start()) + 1
            end = scannable.find("\n", found.start())
            hits.append((term, scannable[start: end if end != -1 else len(scannable)].strip()))
    return hits
