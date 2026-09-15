# Resume Tailoring Pipeline

A deterministic, LLM-assisted pipeline for tailoring a one-page technical
resume to individual software engineering job descriptions.

The system uses LLMs for relevance ranking and controlled writing, while
Python owns factual grounding, resume invariants, verification, assessment
truth, and final acceptance.

The core principle is simple:

> Use models for judgment and wording.  
> Use deterministic code for truth and verification.

---

## What It Does

Given a job description, the pipeline:

1. Parses the posting and extracts requirements and role signals.
2. Selects approved Professional Experience bullets deterministically.
3. Selects the 3 most relevant academic projects.
4. Generates grounded project bullets.
5. Builds JD-relevant Technical Skills.
6. Renders the resume into the fixed LaTeX template.
7. Compiles the actual PDF.
8. Verifies the rendered PDF rather than trusting model output.
9. Generates and validates a cover letter.
10. Produces a deterministic job-fit assessment.
11. Records a successful production application only after every required
    invariant passes.

---

## Architecture

The repository intentionally separates factual authority from model authority.

### `Master_Resume_Context.md`

Canonical factual evidence.

Contains supported:

- work history
- projects
- technologies
- metrics
- education
- coursework
- accomplishments

LLMs may reword supported facts but may not invent facts outside this source.

### `Resume_analysis.xlsx`

Canonical resume policy and approved Professional Experience wording.

Contains:

- Experience bullet IDs
- approved alternate bullets
- Experience swap mappings
- Skills policy
- overflow priorities
- writing rules
- tailoring rules

Professional Experience is selected by ID and never rewritten by an LLM.

### `Resume_Template.tex`

Presentation only.

The template controls layout and contains placeholders such as:

```text
[AUTO:EXPERIENCE]