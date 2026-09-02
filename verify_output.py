#!/usr/bin/env python3
"""
verify_output.py -- verification + standalone page-overflow recovery for a
job's finalized PDF.

verify_job() is pure measurement: page count (pypdf) + proofread (Groq),
writes verification_report.json, never touches the .tex or recompiles.
Called directly by process_job() in tailor_resumes.py after every compile
attempt (page count) and once more at the end (page count + proofread).

Note: as of the current tailor_resumes.py, page-count overflow is already
caught and auto-corrected INSIDE process_job()'s own retry loop -- the
fix_page_overflow()/verify_and_fix_job() functions below are now mainly a
standalone safety net: for older output folders that predate that wiring,
for a job that exhausted MAX_LAYOUT_RETRIES without converging, or for
manually re-checking a folder without spending another Gemini call on the
full pipeline. Run directly via `python verify_output.py <output_folder>`.
"""
import os
import re
import json
import subprocess
from pathlib import Path


GROQ_MODEL = "openai/gpt-oss-120b"

def check_page_count(pdf_path: Path) -> dict:
    from pypdf import PdfReader
    reader = PdfReader(str(pdf_path))
    n = len(reader.pages)
    return {"page_count": n, "one_page": n == 1}


def extract_text(pdf_path: Path) -> str:
    result = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"],
        capture_output=True, text=True,
    )
    text = result.stdout

    # pdftotext inserts a form feed (\f) as its own page-break marker on any
    # multi-page PDF -- that's pdftotext's convention, not corrupted resume
    # text.
    text = text.replace("\f", "")

    # pdftotext renders \item bullet glyphs from Latin Modern/Symbol fonts as
    # Private Use Area codepoints (U+E000-U+F8FF), which have no real meaning
    # outside that font.
    text = re.sub(r"[\uE000-\uF8FF\x80-\x9F\uFFFD]", "", text)

    # pdflatex's own justification hyphenates a word across a line break when
    # it doesn't fit -- normal, correct typesetting, and the PDF renders it
    # properly. pdftotext captures that hyphen+linebreak literally though, so
    # e.g. "configurations" wrapped mid-word comes back as
    # "con-\nfigurations", indistinguishable from a real typo to Groq. Rejoin
    # any hyphen immediately followed by a line break and a lowercase
    # continuation.
    # Caveat: this can't perfectly tell a line-wrap hyphen apart from a
    # genuine compound word that happens to wrap right after its own hyphen
    # (e.g. "real-\ntime") -- rare, but if a compound-word bullet ever reads
    # oddly in a proofread flag, worth checking the PDF directly before
    # trusting the flag.
    text = re.sub(r"(\w)-\n\s*([A-Za-z]\w*)", r"\1\2", text)

    return text


def groq_proofread(pdf_text: str, api_key: str) -> dict:
    from groq import Groq
    client = Groq(api_key=api_key)
    prompt = f"""You are proofreading the FINAL, already-compiled text of a resume PDF.
Flag ONLY objectively broken things: misspelled/garbled words, dropped
letters, duplicated words, or sentences that don't grammatically parse.

Do NOT comment on line counts, page fill percentage, layout, or wording
style/content choices -- those are verified separately and are not your job.

RESUME TEXT:
{pdf_text}

Return ONLY valid JSON on a single line, no markdown fences, no line breaks
inside string values:
{{"issues": [{{"snippet": "...", "problem": "..."}}], "clean": true|false}}
"""
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=4096,
    )
    raw = resp.choices[0].message.content.strip()
    raw = re.sub(r"^```json\s*|\s*```$", "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Model likely emitted a raw newline/control character inside a
        # string value -- strict json.loads rejects this even though it's
        # a common, harmless model quirk. strict=False tolerates control
        # characters inside strings instead of failing outright.
        try:
            return json.loads(raw, strict=False)
        except json.JSONDecodeError as e:
            return {
                "issues": [],
                "clean": None,
                "parse_error": str(e),
                "raw_response": raw,
            }


def verify_job(out_dir: Path, pdf_path: Path) -> dict:
    """
    Runs both checks for a single job's finalized PDF and writes
    verification_report.json into that job's own output folder. Pure
    measurement -- never writes to the .tex or recompiles. Called directly
    by process_job() in tailor_resumes.py -- no manual invocation, no
    folder discovery, no separate step for you to remember.

    "needs_review" is the single source of truth for whether this job's
    output is trustworthy as-is -- True if the PDF isn't 1 page, or if the
    Groq proofread came back non-clean (or couldn't be parsed, since a
    parse_error means "clean" is unknown, not confirmed true). Callers
    should read this field directly instead of re-deriving it from
    page_check/proofread separately.
    """
    page_report = check_page_count(pdf_path)

    api_key = os.environ.get("GROQ_API_KEY")
    proofread = None
    if api_key:
        pdf_text = extract_text(pdf_path)
        proofread = groq_proofread(pdf_text, api_key)

    needs_review = not page_report["one_page"]
    if proofread is not None and not proofread.get("clean", True):
        needs_review = True

    report = {
        "pdf": str(pdf_path),
        "page_check": page_report,
        "needs_review": needs_review,
    }
    if proofread is not None:
        report["proofread"] = proofread
    (out_dir / "verification_report.json").write_text(json.dumps(report, indent=2))

    return report


def fix_page_overflow(out_dir: Path, pdf_path: Path, max_retries: int = 3) -> dict:
    """
    Standalone recovery pass for a PDF that's still >1 page -- e.g. an older
    output folder that predates page-count checking being wired into
    process_job()'s own retry loop, or a job that exhausted
    MAX_LAYOUT_RETRIES there without converging. Reloads the content this
    job was built from (content.json) and the JD it was tailored against
    (jd_text.txt) -- both saved by process_job() so this can run
    independently, without re-calling Gemini's content-tailoring step.
    Requires GEMINI_API_KEY_1 (and optionally _2..._5), since fix_layout()
    itself calls Gemini.

    Returns {"attempted": bool, "reason": str (if not attempted),
             "retries_used": int, "final_page_count": int, "resolved": bool}.
    """
    content_path = out_dir / "content.json"
    jd_path = out_dir / "jd_text.txt"
    if not content_path.exists() or not jd_path.exists():
        missing = content_path.name if not content_path.exists() else jd_path.name
        return {
            "attempted": False,
            "reason": f"missing {missing} -- this job's output predates "
                      "content.json/jd_text.txt being saved; cannot "
                      "reconstruct what to fix. Re-run tailor_resumes.py on "
                      "this job to regenerate with the current pipeline.",
        }

    from tailor_resumes import (
        fill_template, compile_tex, parse_hbox_warnings, measure_fill,
        check_verb_repetition, fix_layout, force_page_overflow_target,
        TEMPLATE_PATH,
    )

    content = json.loads(content_path.read_text())
    jd_text = jd_path.read_text()
    template = Path(TEMPLATE_PATH).read_text()
    tex_path = pdf_path.with_suffix(".tex")
    log_path = tex_path.with_suffix(".log")

    for attempt in range(1, max_retries + 1):
        page_report = check_page_count(pdf_path)
        if page_report["one_page"]:
            content_path.write_text(json.dumps(content, indent=2))
            return {
                "attempted": True, "retries_used": attempt - 1,
                "final_page_count": page_report["page_count"], "resolved": True,
            }

        print(f"  -> page-fix attempt {attempt}/{max_retries}: PDF is "
              f"{page_report['page_count']} pages, requesting a targeted trim...")
        fill_report = measure_fill(pdf_path, content, verbose=False)
        if not fill_report["overflow_sections"]:
            fill_report["overflow_sections"] = force_page_overflow_target(fill_report)
        hbox_warnings = parse_hbox_warnings(log_path.read_text(errors="ignore")) if log_path.exists() else []
        verb_report = check_verb_repetition(content)

        content = fix_layout(content, hbox_warnings, fill_report, verb_report, jd_text)
        if content.get("flagged"):
            print(f"  !! flagged during page-fix: {content['flagged']}")

        tex_path.write_text(fill_template(template, content))
        compile_tex(tex_path)

    final_page_report = check_page_count(pdf_path)
    content_path.write_text(json.dumps(content, indent=2))
    return {
        "attempted": True, "retries_used": max_retries,
        "final_page_count": final_page_report["page_count"],
        "resolved": final_page_report["one_page"],
    }


def verify_and_fix_job(out_dir: Path, pdf_path: Path) -> dict:
    """
    Runs verify_job() first (pure measurement, ground truth). If page count
    is the problem -- and ONLY page count, since a proofread failure is
    genuinely corrupted text that a page-trim can't fix -- attempts
    fix_page_overflow() and then re-runs verify_job() against the
    recompiled PDF so the final report reflects what's actually on disk
    now, not the pre-fix state.
    """
    report = verify_job(out_dir, pdf_path)
    proofread_ok = report.get("proofread", {}).get("clean", True) is not False
    if report["needs_review"] and not report["page_check"]["one_page"] and proofread_ok:
        fix_result = fix_page_overflow(out_dir, pdf_path)
        report["page_fix"] = fix_result
        if fix_result.get("attempted") and fix_result.get("resolved"):
            print("  -> page overflow resolved, re-verifying final PDF...")
            report = verify_job(out_dir, pdf_path)
            report["page_fix"] = fix_result
    return report


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python verify_output.py <output_folder>")
    out_dir = Path(sys.argv[1])
    pdfs = list(out_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDF found in {out_dir}")
    pdf_path = pdfs[0]
    print(f"=== Verifying {pdf_path.name} ===")
    report = verify_and_fix_job(out_dir, pdf_path)
    print(f"  -> page count: {report['page_check']['page_count']} "
          f"({'OK' if report['page_check']['one_page'] else 'FAIL, not 1 page'})")
    if "proofread" in report:
        if report["proofread"].get("clean"):
            print("  -> proofread clean")
        else:
            for issue in report["proofread"].get("issues", []):
                print(f"      - \"{issue['snippet']}\" -- {issue['problem']}")
    else:
        print("  -> GROQ_API_KEY not set, proofread skipped")
    if "page_fix" in report:
        pf = report["page_fix"]
        if not pf.get("attempted"):
            print(f"  -> page-fix skipped: {pf['reason']}")
        else:
            print(f"  -> page-fix: {pf['retries_used']} retr{'y' if pf['retries_used']==1 else 'ies'} used, "
                  f"final page count {pf['final_page_count']} "
                  f"({'resolved' if pf['resolved'] else 'NOT resolved -- review manually'})")
    print(f"  -> needs_review: {report['needs_review']}")
    print(f"  -> report saved: {out_dir}/verification_report.json")
