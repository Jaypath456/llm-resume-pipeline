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


    text = text.replace("\f", "")
    text = re.sub(r"[\uE000-\uF8FF\x80-\x9F\uFFFD]", "", text)

    PROTECTED_HYPHENATED_TERMS = [
        "scikit-learn", "real-time", "self-attention", "co-located",
        "e-commerce", "state-of-the-art", "full-stack",
    ]
    PLACEHOLDER = "\x00HY\x00"
    protected_map = {}
    for i, term in enumerate(PROTECTED_HYPHENATED_TERMS):
        key = f"{PLACEHOLDER}{i}{PLACEHOLDER}"
        pattern = re.escape(term).replace(r"\-", r"-\s*\n?\s*")
        text, n = re.subn(pattern, key, text, flags=re.IGNORECASE)
        if n:
            protected_map[key] = term

    text = re.sub(r"(\w)-\n\s*([A-Za-z]\w*)", r"\1\2", text)

    for key, term in protected_map.items():
        text = text.replace(key, term)

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
        # Model occasionally emits a raw control character inside a string
        # value; strict=False tolerates that instead of failing outright.
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
    Runs both checks for a job's finalized PDF and writes
    verification_report.json. Pure measurement -- never writes to the .tex
    or recompiles.

    "needs_review" is the source of truth for whether this job's output is
    trustworthy as-is: True if the PDF isn't 1 page, or if the Groq
    proofread came back non-clean or couldn't be parsed.
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
    Standalone recovery pass for a PDF still >1 page. Reloads content.json
    and jd_text.txt (saved by process_job()) so this can run independently
    without re-calling Gemini's content-tailoring step. Requires
    GEMINI_API_KEY_1 (and optionally _2..._5), since fix_layout() calls
    Gemini.

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
    Runs verify_job() first. If page count is the ONLY problem (a proofread
    failure is genuine corrupted text that a page-trim can't fix), attempts
    fix_page_overflow() and re-runs verify_job() so the final report
    reflects what's actually on disk now.
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
