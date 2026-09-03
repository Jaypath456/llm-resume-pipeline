# Resume Tailoring Pipeline

Tailors a resume to a specific job description using Gemini, verifies the
*actual compiled PDF* against real layout and content rules (not the
model's self-report), and auto-corrects only what's genuinely broken —
before falling back to a human review flag.

## Project Scope

This is a side project exploring reliable verification loops for
LLM-generated output, using resume tailoring as the test case. It does not
generate or submit my actual job applications.

The focus is on the engineering problem: how to verify an LLM-generated
artifact against deterministic ground truth, catch failures that the model
misses, and retry or escalate only when necessary.

## Why this exists

LLMs are confident about their own output being correct, even when it
isn't. Ask one "does this fit on one page?" and it'll say yes regardless of
what the real PDF says. This pipeline doesn't trust that:

```
Gemini generates -> real pdflatex + pdftotext measurement ->
deterministic guardrails -> targeted fix (only if needed) -> re-verify
```

Nothing is marked "done" until the actual compiled artifact says so.

## What it checks, deterministically (no API call required)

- **Layout**: real page count, real per-section line counts, sparse
  "orphan" last lines, via `pdftotext -layout` on the compiled PDF
- **Honesty**: any technology/skill the resume claims must trace back to
  your source content — unverifiable claims are auto-stripped, never
  silently kept
- **Narrative coherence**: flags bullets that quietly merge two separate
  accomplishments into one inflated sentence
- **Verb repetition**: flags the same action verb appearing more than
  twice across the whole resume

## What it checks with a second, independent model

Groq proofreads the final compiled PDF text for genuine typos or garbled
words — separate from, and after, Gemini's writing pass, so the same
model never grades its own work.

## Cost-conscious by design

Built against Gemini's free tier across multiple rotated accounts, so the
pipeline actively avoids wasting calls:

- Skips a Gemini call entirely when a fix is purely deterministic (e.g.
  stripping an unverified skill) — just recompiles instead
- Stops re-asking Gemini a question it already answered "I can't fix this
  honestly" to on a prior attempt
- Groq handles proofread fixes first (free); Gemini is only a last-resort
  fallback after two failed Groq attempts

## One-time setup

```bash
git clone <this-repo>
cd <this-repo>
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install system dependencies:

```bash
# TeX (for pdflatex)
sudo apt install texlive-full        # Debian/Ubuntu
# or: brew install --cask mactex     # macOS

# poppler (for pdftotext)
sudo apt install poppler-utils       # Debian/Ubuntu
# or: brew install poppler           # macOS
```

Confirm both work:

```bash
pdflatex --version
pdftotext -v
```

## Configure your API keys

```bash
cp .env.example .env
```

Open `.env` and paste in your real keys. Then load them into your shell
**every time you open a new terminal session** (this project does not use
python-dotenv, so the keys must be real environment variables before you
run the script):

```bash
export $(grep -v '^#' .env | xargs)
```

What this does: `grep -v '^#' .env` reads `.env` and strips out comment
lines, `xargs` turns each `KEY=value` line into an argument, and
`export $(...)` sets each one as a real shell environment variable. Do
**not** type `.env` directly as a command — it is a data file, not a
script, and your shell will fail trying to execute it.

To verify it worked:

```bash
echo $GEMINI_API_KEY_1
```

If that prints your key, you're set. If it prints nothing, the export
step didn't run in this shell session — re-run the `export $(grep ...)`
command above.

Get your keys here:
- Gemini (free): https://aistudio.google.com/apikey — only
  `GEMINI_API_KEY_1` is required; `_2` through `_5` enable automatic
  rotation when one account's daily free quota is exhausted.
- Groq (free): https://console.groq.com/keys

## Set up your content

```bash
cp Master_Resume_Content.example.md Master_Resume_Content.md
```

Edit `Master_Resume_Content.md` with your real experience. This file is
the **only** source of truth the model is allowed to draw from — it's
listed in `.gitignore`, so your real resume content never leaves your
machine or gets committed.

Edit `Resume_Template.tex` to match your own LaTeX resume layout. Section
names and line-count targets are currently hardcoded to match a specific
template structure — see `LINE_TARGETS` and `PROJECTS` in
`tailor_resumes.py` if you're adapting this to a different resume shape.

Add job descriptions, one per file, inside `jobs/`:

```
jobs/
├── acme_backend_engineer.txt
└── globex_ml_engineer.txt
```

## Run it

```bash
python tailor_resumes.py
```

Each `output/<position>_<company>/` folder will contain:

| File | What it is |
|---|---|
| `<name>.pdf` | The compiled resume |
| `<name>.tex` | The LaTeX source |
| `cover_letter.txt` | Tailored cover letter |
| `content.json` | Tailored content as structured JSON |
| `keyword_audit.json` | Per-JD-requirement supported/partial/unsupported breakdown |
| `verification_report.json` | Page count + proofread result |
| `jd_text.txt` | Copy of the job description used |

A `.processed_jobs.json` manifest tracks status across runs, so
re-running the script never redoes finished work. `needs_review` jobs are
automatically retried on the next run, up to twice, before being left
alone for manual review.

## Reading the output

- **`tier1_gaps`** printed to console = a JD requirement with genuinely no
  backing in your master content. This is intentional — the pipeline
  won't fabricate coverage to close the gap.
- **`needs_review`** status = something couldn't be fully auto-resolved.
  Check the job's `.tex` file and the console log; it's often a cosmetic
  layout issue (a skills line slightly short of its target length) rather
  than a factual problem.
- **No PDF produced** = an actual LaTeX compile error, not a layout
  warning — check the `.log` file in that job's output folder.

## Adapting this to your own resume

This was built around one specific LaTeX template and a fixed set of
experience/project sections. To reuse it:

1. Swap in your own `Resume_Template.tex`.
2. Update `LINE_TARGETS` and `PROJECTS` in `tailor_resumes.py` to match
   your sections.
3. Write your own `Master_Resume_Content.md` following the structure in
   `Master_Resume_Content.example.md` — especially the
   `## Explicitly Unsupported` section, which powers the hard-banned-term
   guardrail.

## Known limitations

- Layout-fit checking is tuned to a single page; multi-page resumes
  aren't supported out of the box.
- A skills category with genuinely few real, verifiable entries can end
  up cosmetically short rather than perfectly filled — this is the
  honesty guardrail refusing to pad it with an invented skill, not a bug.
- `needs_review` retries re-tailor the job from scratch rather than
  patching just the part that failed.

## Cost note

This is built to run comfortably within free tiers (Gemini + Groq), but
usage limits and pricing can change. Check current terms before running
large batches:
- https://ai.google.dev/pricing
- https://console.groq.com/docs/rate-limits
