# Resume Tailoring & Verification Pipeline

A production-oriented resume tailoring pipeline that combines LLM generation with deterministic policy, evidence grounding, real PDF verification, independent auditing, and safe application tracking.

The goal is not to let an LLM freely rewrite a resume. The pipeline gives models narrow responsibilities, keeps factual and policy authority outside the models, and verifies the actual compiled artifact before a run is considered successful.

> This project generates and audits tailored application artifacts. It does **not** automatically submit job applications.

---

## Why this exists

Most resume-tailoring workflows stop at:

```text
JD -> LLM -> resume
```

That is convenient, but difficult to trust. A model can:

- invent unsupported skills or metrics,
- subtly rewrite approved experience,
- choose evidence that does not match the job,
- claim a one-page layout without checking the PDF,
- return confident company/visa research without reliable sources,
- or report success even when tracking state is inconsistent.

This pipeline instead uses:

```text
LLMs for semantic understanding and bounded writing
+
Python for authority, policy, grounding, verification, and persistence
```

The actual compiled PDF, not the model's self-report, is the final source of truth for layout and rendering.

---

## High-level architecture

```text
                         Job Description
                               |
                 +-------------+-------------+
                 |                           |
        Deterministic parsing         Gemini semantic pass
        + requirement extraction      + project relevance
                 |                    + supplemental JD signals
                 |                           |
                 +-------------+-------------+
                               |
                     Validated merged signals
                               |
                     Deterministic policy engine
                               |
             +-----------------+------------------+
             |                                    |
      Experience selection                  Project selection
      exact approved wording                exactly 3 projects
      deterministic swaps                   3 / 2 / 2 bullets
             |                                    |
             |                              Gemini writes bullets
             |                                    |
             +-----------------+------------------+
                               |
                       Skills + section order
                               |
                         LaTeX rendering
                               |
                    pdflatex + pdftotext checks
                               |
                       Cover letter generation
                       + deterministic grounding
                       + bounded repair
                               |
                   +-----------+------------+
                   |                        |
          Qwen application audit    GPT-OSS company research
          independent evaluator     browser-required research
                   |                        |
                   +-----------+------------+
                               |
                         strategy.json
                               |
                   tracking / reconciliation
```

---

## Responsibility split

### Python: authority layer

Python owns decisions that must be deterministic or auditable:

- JD requirement extraction
- deterministic role-family classification
- semantic-signal validation and merge rules
- Experience bullet selection
- approved Experience swaps
- section order
- exact Experience wording
- skills selection and line targeting
- project bullet allocation
- grounding checks
- metric/qualifier validation
- source-isolation checks
- PDF/layout verification
- application requirement verdicts
- tracking and reconciliation
- final success / needs-review status

### Gemini: semantic understanding + writing

Gemini is used for:

- ranking/selecting the three most relevant projects
- proposing supplemental semantic JD signals
- writing project bullets from approved evidence
- writing and repairing the cover letter

Gemini **does not**:

- choose Experience bullet IDs directly
- rewrite Professional Experience wording
- bypass deterministic swap rules
- decide the final section order
- decide whether unsupported evidence is acceptable

### Qwen: independent application assessment

The final resume/application audit uses:

```text
qwen/qwen3.8-27b
```

Qwen evaluates the finished application, including:

- resume tailoring
- Experience selection
- project selection
- project bullet quality
- requirement coverage
- callback likelihood
- risks / missing evidence

Its requirement classifications are advisory. Python remains authoritative when the model and deterministic evidence disagree.

### GPT-OSS: browser-backed company research

Company/job research uses:

```text
openai/gpt-oss-20b
```

for:

- visa sponsorship policy
- STEM OPT support
- job-posting metadata

Browser search is required.

An unbrowsed model response is **never** allowed to establish a current company or job policy.

---

## Source-of-truth hierarchy

### `Master_Resume_Context.md`

The factual evidence base.

It contains the candidate's supported:

- experience
- projects
- technologies
- metrics
- education
- certifications
- standing facts
- unsupported / disallowed claims

Generated content must trace back to this evidence.

### `Resume_analysis.xlsx`

The human policy specification used at runtime for items such as:

- approved Experience wording
- Experience alternatives
- Experience swap rules
- mandatory skills baseline
- layout targets
- project/bullet targets
- action-verb policy

The workbook is read fresh from disk.

### `Resume_Template.tex`

Presentation only.

The template controls how approved content is rendered; it is not the authority for resume facts or Experience policy.

---

## Professional Experience: deterministic by design

Professional Experience is intentionally the least generative section.

The normal flow is:

```text
JD
 |
 +-> deterministic role/signals
 |
 +-> Gemini supplemental semantic signals
          |
          +-> Python validates evidence against the JD
 |
 +-> merge:
       deterministic True cannot be erased by Gemini
       Gemini may only add a validated missed signal
 |
 +-> deterministic Experience rule
 |
 +-> approved Experience IDs
 |
 +-> exact wording from Resume_analysis.xlsx
```

This prevents the model from silently "improving" work history.

After rendering, the pipeline verifies that every Experience bullet in the PDF matches an approved variant.

---

## Semantic JD signals

The semantic taxonomy includes:

- backend
- agentic AI
- AI/ML-adjacent
- data engineering
- cloud/infrastructure
- systems/low-level
- code-quality/collaboration-heavy
- healthcare

Gemini acts as a **recall layer**, not an authority layer.

A proposed signal must include grounded JD evidence. Python rejects unsupported semantic additions.

Some high-risk signals have additional meaning-specific validation. For example, generic phrases such as "collaboration is key" or "test your own code" are not enough to establish a code-quality/collaboration-heavy role. Team-level engineering-quality evidence such as code reviews, PR standards, mentoring, engineering standards, or shared quality ownership is required.

---

## Project selection

Exactly **3 projects** are selected.

Projects are ranked by relevance to the JD, then receive:

```text
rank 1 -> 3 bullets
rank 2 -> 2 bullets
rank 3 -> 2 bullets
```

The most relevant project therefore gets the most resume space.

Selection order and display order are intentionally separate:

- **relevance** decides which projects are chosen and how many bullets they receive
- **chronology** decides how selected projects are displayed on the resume

Project headers remain clean and do not carry technology-stack keyword dumps.

---

## Project bullet generation

Gemini writes project bullets from project-specific evidence only.

Every generated bullet is checked for:

- factual grounding
- project ownership
- supported technologies
- supported metrics
- action-verb limits
- cross-project leakage
- unsupported claims

A bullet that fails deterministic validation is repaired or rejected rather than silently accepted.

---

## Technical Skills

The pipeline starts from the approved baseline and adds only supported, JD-relevant skills.

Current layout contract:

```text
Technical Skills = exactly 8 rendered lines
```

The system compiles the resume and measures the actual result.

If there is unused space but no supported JD-relevant skill worth adding, it leaves the line short rather than padding the resume with unrelated keywords.

---

## Cover letter

The cover letter is written by Gemini but validated deterministically.

Current contract includes:

- 140-300 words
- grounded only in candidate evidence or clearly attributed employer/JD facts
- no unsupported technologies
- no ownership fusion between unrelated projects/roles
- quantitative claims preserve their original qualifiers
- strongest supported role priority should be addressed
- normally covers the top two supported distinctive priorities
- priority 3 is optional
- each substantive evidence paragraph stays within one role or project

### Qualified metrics

Approximation is part of the fact.

For example, if the evidence says:

```text
approximately 27.6:1
```

the model may use forms such as:

```text
approximately 27.6:1
about 27.6:1
~27.6:1
```

but may not strengthen it to:

```text
27.6:1
```

If a generation attempt drops a qualifier, the retry prompt receives the rejected form and the safe supported replacement. A failed numeric form is also remembered within that letter-generation session so later attempts cannot repeat the same mistake.

---

## Real PDF verification

The pipeline does not ask an LLM whether the resume fits.

It actually:

1. generates LaTeX
2. compiles with `pdflatex`
3. extracts rendered text with `pdftotext`
4. measures the compiled artifact
5. checks section order, bullet counts, line counts, and exact Experience wording
6. applies bounded repairs only when deterministic checks show a real problem
7. recompiles and verifies again

Core output contract:

```text
1 page
3 selected projects
3 / 2 / 2 project bullets
8 rendered Technical Skills lines
exact approved Experience wording
```

Action-verb repetition is also checked.

---

## Company research: fail closed, not confidently wrong

Company research is intentionally stricter than ordinary LLM output.

### Browser requirement

Every `company_research` request requires browser search.

There is no fallback to:

```text
web_search=False
```

and no Gemini/Qwen fallback for current company policy.

### Recovery behavior

At most two browser-backed research generations are allowed:

```text
attempt 1:
    browser required
    normal research format

if output parsing fails:

attempt 2:
    browser still required
    simplified output format
```

If research still fails:

```text
research_available = false
visa sponsorship   = UNKNOWN
STEM OPT support   = UNKNOWN
job posted         = UNKNOWN
confidence         = LOW
```

Research failure does not corrupt an otherwise valid resume run.

### Provenance enforcement

A URL is not trusted just because the model prints it.

Non-UNKNOWN conclusions must be supported by URLs recorded in the provider's actual browser/search execution metadata.

Unfetched or fabricated URLs are discarded.

### Research policy

The validator also enforces scope:

- current company-wide policy requires reliable company-level evidence
- an unrelated posting cannot establish company-wide `NO`
- exact-job restrictions require exact-job evidence
- exact URL / job ID / requisition ID is strongest
- company + title + JD-stated location can be accepted when IDs are unavailable
- title alone is insufficient when multiple postings may exist
- geography-specific evidence cannot identify an exact job if the audited JD provides no matching geography
- historical H-1B activity does not prove current sponsorship policy
- E-Verify alone does not prove STEM OPT support
- no reliable evidence -> `UNKNOWN`

---

## Independent audits

After the generation artifacts exist, the pipeline launches:

```text
Qwen assessment
GPT-OSS company research
```

in parallel.

The workers are intentionally pure:

- they do not write `strategy.json`
- they do not write tracking state
- they do not mutate generation artifacts

The coordinator waits for both, merges their returned results in memory, and performs the final atomic persistence.

A company-research failure therefore does not destroy a successful Qwen assessment.

---

## Tracking

Applications/runs are keyed by a deterministic JD fingerprint.

The tracking index is the authority for duplicate/skip decisions.

A successful normal run records the application without creating duplicate history.

### Stale references

If the index points to a run folder that no longer exists, the pipeline does **not** guess.

Instead it:

- preserves the old tracking state
- does not invent a new application row
- parks the conflict in `.tracking_reconcile.json`
- keeps the newly generated artifacts
- reports:

```text
SUCCESS_WITH_TRACKING_WARNING
```

This means generation succeeded, but tracking requires operator attention.

### Explicit reconciliation

Reconciliation is deliberately separate from normal generation.

Dry run:

```bash
python run_pipeline.py \
  --reconcile-tracking output/<RUN_FOLDER>
```

The command verifies the candidate replacement without mutating state.

Apply:

```bash
python run_pipeline.py \
  --reconcile-tracking output/<RUN_FOLDER> \
  --apply
```

Validation includes:

- run folder exists
- required artifacts exist
- exactly one finalized/renamed PDF exists
- run is a clean success or tracking-only warning
- JD fingerprint recomputes and matches
- reconciliation entry exists
- fingerprint exists in the tracking index
- currently indexed folder is actually missing
- no rival authoritative successful run exists
- company/title metadata are compatible
- reconciliation entry is still unresolved

A successful reconciliation:

- atomically repoints the index
- preserves the missing run under history
- marks the reconciliation event resolved
- does not append a duplicate application row
- is idempotent

Running reconciliation again should report that there is nothing left to do.

---

## CLI examples

### Production run

```bash
python run_pipeline.py jobs/bae.txt
```

### Mock smoke test

```bash
python run_pipeline.py jobs/bae.txt --mock --smoke-test
```

Mock/smoke runs do not mutate production tracking.

### Re-assess an existing run

```bash
python run_pipeline.py --assess \
  output/BAE_Systems_Entry_Level_Software_Engineer_2026-09-16_034911459620
```

`--assess`:

- does not regenerate the resume
- does not call Gemini
- recomputes/audits the existing artifacts
- reruns Qwen + company research
- verifies generation artifacts remain unchanged

### Reconcile stale tracking

Dry run:

```bash
python run_pipeline.py \
  --reconcile-tracking output/<RUN_FOLDER>
```

Apply:

```bash
python run_pipeline.py \
  --reconcile-tracking output/<RUN_FOLDER> \
  --apply
```

Use:

```bash
python run_pipeline.py --help
```

for the full current CLI.

---

## Typical run artifacts

A completed run folder contains artifacts such as:

```text
output/<Company>_<Role>_<timestamp>/
├── <Company>_<Role>.pdf
├── resume.tex
├── resume.txt
├── cover_letter.txt
├── assessment.txt
├── strategy.json
├── job_description.txt
└── run.log
```

LaTeX build intermediates are removed after finalization.

### `strategy.json`

`strategy.json` acts as the structured audit trail for a run and records items such as:

- parsed JD signals
- deterministic/Gemini semantic signal decisions
- Experience rule and shipped IDs
- selected projects
- bullet allocation
- generated project bullets
- rendered skills
- verification results
- cover-letter checks
- Qwen audit
- company research
- final assessment metadata

---

## Provider configuration

Secrets belong in `.env` and should never be committed.

Example names used by the current setup:

```env
GEMINI_API_KEY_1=
GEMINI_API_KEY_2=
# ...
GEMINI_API_KEY_8=

GROQ_API_KEY_1=
GROQ_API_KEY_2=

GROQ_ASSESSMENT_MODEL="qwen/qwen3.8-27b"
GROQ_RESEARCH_MODEL="openai/gpt-oss-20b"

GROQ_OTPM_LIMIT__QWEN_QWEN3_8_27B=1000
GROQ_ITPM_LIMIT__QWEN_QWEN3_8_27B=7000
```

### Gemini rotation

Multiple configured Gemini accounts are supported.

Daily/account-specific exhaustion rotates immediately to the next configured account. Temporary server failures may retry with the existing backoff policy.

### Groq limits

Groq limits are organization/model limits, so changing API keys does not create new token quota.

The pipeline therefore distinguishes key-specific failures from organization-wide token/rate limits and does not rotate keys when rotation cannot help.

---

## Logging and secret hygiene

Logs intentionally identify credential slots rather than secret values:

```text
gemini account #1
groq key=#1
```

API keys are not written into run artifacts.

Provider error messages may still contain non-secret metadata such as:

- model names
- token usage
- quota limits
- provider organization identifiers

Sanitize personal/application data and provider identifiers before publishing raw production logs publicly.

---

## Testing

The current hardened implementation has:

```text
603 passed
1 skipped
```

The four regression smoke fixtures also complete successfully.

Regression checks preserve the generated resume across:

- section order
- role family
- Experience rule
- Experience IDs
- Experience wording
- project selection
- project display order
- 3/2/2 allocation
- project bullets
- Technical Skills
- page count
- extracted PDF text

The company-research and tracking changes are tested independently so that operational hardening cannot silently change resume generation.

Run the test suite with:

```bash
python -m pytest tests/test_pipeline.py
```

---

## Design principles

### 1. Models propose; deterministic code decides

LLMs are used where semantic judgment or natural-language generation is useful. They do not own policy.

### 2. Evidence beats fluency

A polished unsupported sentence is still wrong.

### 3. Verify the artifact, not the prompt

The compiled PDF is what an employer receives, so the compiled PDF is what the pipeline verifies.

### 4. Fail closed for external research

If current company policy cannot be proven through browser-backed evidence, return `UNKNOWN`.

### 5. Preserve history instead of guessing

Tracking inconsistencies become explicit reconciliation states rather than silent rewrites.

### 6. Repairs are bounded and targeted

A failed check should cause the smallest possible repair rather than a full uncontrolled regeneration.

---

## Current status

The core tailoring pipeline is production-usable:

- deterministic Experience selection is enforced
- semantic JD recovery is evidence-validated
- project generation is grounded
- skills and layout are verified from the compiled PDF
- cover-letter repairs are bounded and deterministic
- Qwen provides an independent application audit
- company research requires browser provenance and fails safely to `UNKNOWN`
- stale tracking has an explicit dry-run/apply reconciliation workflow
- regression tests protect the working resume-generation behavior

The main remaining runtime variability comes from external model/provider availability and quota limits, not from the resume policy or verification architecture.
