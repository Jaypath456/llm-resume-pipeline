"""The only place this pipeline talks to a language model.

Three transports share one request path: Gemini (multi-account rotation),
Groq (bounded retries, Gemini as fallback) and a mock used by `--mock` runs.
The mock replaces provider RESPONSES only - prompt construction, response
validation, grounding and the retry loop are the same code in every mode.

What a model is allowed to decide:
  * which 3 projects to select, and their rank;
  * the wording of project bullets (inside that project's own evidence);
  * the cover letter draft, and the closing assessment.

What a model never touches: Professional Experience, Technical Skills,
section order, project headers, layout. Those are deterministic.

Secrets: a key is read from `Credentials` at call time and never logged.
Logs identify an account as `#1`, `#2`, and never carry key material.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import grounding
import resume_engine as engine
from resume_engine import (BACKOFF_BASE_SECONDS, BACKOFF_MAX_SECONDS, Credentials,
                           GEMINI_MODEL, GROQ_MAX_ATTEMPTS, GROQ_MAX_OUTPUT_TOKENS,
                           GROQ_MODEL, MasterFacts, Policy, Project,
                           RATE_LIMIT_MAX_ATTEMPTS, SERVER_ERROR_MAX_ATTEMPTS, Signals)



def dotenv_settings() -> dict:
    """The .env file as a mapping, or empty when it cannot be read."""
    try:
        from dotenv import dotenv_values

        return dict(dotenv_values(engine.PROJECT_ROOT / ".env") or {})
    except Exception:                        # noqa: BLE001 - optional dependency
        return {}


def configured(name: str, *, default: str = "") -> str:
    """One setting, resolved at CALL time with os.environ winning over .env.

    engine's module constants bind at import time from os.environ only, while
    load_credentials() separately merges .env - a project configured purely in
    .env therefore had working keys and empty model names. Everything the
    provider layer needs is resolved through here instead.
    """
    return (os.getenv(name) or dotenv_settings().get(name) or default or "").strip()


def configured_groq_model() -> str:
    """Legacy single-model setting, kept for compatibility."""
    return configured("GROQ_MODEL")


def assessment_model() -> str:
    """The evaluator. No silent default: unset means the audit is unavailable."""
    return configured("GROQ_ASSESSMENT_MODEL")


def research_model() -> str:
    """The web researcher. No silent default, same reason."""
    return configured("GROQ_RESEARCH_MODEL")


def groq_key_pool() -> tuple[str, ...]:
    """Every configured Groq key, in deterministic order.

    GROQ_API_KEY_1, GROQ_API_KEY_2, ... first, then legacy GROQ_API_KEY, with
    duplicates collapsed. Order is stable across runs so a failure is
    reproducible. Key material is never logged: callers identify a key as #1
    or #2 and nothing else.
    """
    settings = dotenv_settings()

    def read(name: str) -> str:
        return (os.getenv(name) or settings.get(name) or "").strip()

    pool: list[str] = []
    index = 1
    while True:
        key = read(f"GROQ_API_KEY_{index}")
        if not key:
            break
        if key not in pool:
            pool.append(key)
        index += 1
    legacy = read("GROQ_API_KEY")
    if legacy and legacy not in pool:
        pool.append(legacy)
    return tuple(pool)


# Categories that mean "this key is no good", as opposed to "this ACCOUNT is
# over its limit". Only the former justifies trying another key: two keys in
# one organization share every rate and daily ceiling, so rotating on a limit
# error just burns the second key for nothing.
KEY_SPECIFIC_CATEGORIES = frozenset({"auth_permission", "invalid_key"})

# Never retried: the same request cannot succeed, and on a daily ceiling no
# amount of waiting inside this run will help either.
# Failures that belong to the ACCOUNT, not to the request. Retrying the same
# bytes against the same account cannot help, so Gemini rotates immediately.
# Transient categories (server_error, rate_limited) keep the existing backoff:
# a per-minute limit clears, a daily one does not.
GEMINI_ROTATE_CATEGORIES = frozenset({
    "quota_exhausted", "daily_limit_exhausted", "auth_permission", "invalid_key",
})

NON_RETRYABLE_CATEGORIES = frozenset({
    "request_too_large", "output_truncated", "bad_request",
    "daily_limit_exhausted", "quota_exhausted",
    # A rejected schema is a bug in our request, not bad luck.
    "schema_rejected",
})


BULLET_ATTEMPTS = 3
LETTER_ATTEMPTS = 3
ASSESSMENT_ATTEMPTS = 3

# Groq bills a request as prompt + RESERVED output, and the on-demand tier's
# tokens-per-minute budget is small. A blanket 4096-token reservation made the
# audits alone exceed it, so each audit reserves what its own answer needs:
# the three research/experience objects are small, and the application audit's
# JSON carries the requirement buckets.
# The audits are classification and scoring work, not open-ended reasoning.
# Low effort is what keeps the answer inside the reserved output budget.
AUDIT_REASONING = {"reasoning_effort": "low", "include_reasoning": False}

AUDIT_OUTPUT_TOKENS = {
    "company_research": 1000,    # one small object plus its source list
    # Qwen's tier caps OUTPUT tokens per minute at 1000, separately from the
    # combined 8K TPM ceiling. A 2600-token reservation could never be served,
    # so the assessment asks for 900 and the prompt demands a compact answer.
    "assessment": 900,
}

# Advisory pacing budget. Overridable because it is a per-account tier limit,
# not a property of this pipeline.
GROQ_TPM_LIMIT = int((os.getenv("GROQ_TPM_LIMIT") or "8000").strip() or 8000)


# ================================================================= requests


@dataclass
class Request:
    """One model call. `context` is for the mock; real transports ignore it."""

    purpose: str
    prompt: str
    context: dict[str, Any] = field(default_factory=dict)
    temperature: float = 0.4
    # Sent to Groq as max_completion_tokens (the deprecated max_tokens spelling
    # is not used); the field name is kept for the callers that already set it.
    max_tokens: int = GROQ_MAX_OUTPUT_TOKENS
    json: bool = True          # False for purposes whose output is plain prose
    # Only the company-research audit sets this. No generation request may.
    web_search: bool = False
    # Reasoning-model controls. GPT-OSS defaults to medium effort, which spends
    # the output budget thinking instead of answering and truncated both
    # audits. The AUDITS ask for low effort; generation is left at the
    # provider default, so None means "do not send this field at all".
    reasoning_effort: str | None = None
    include_reasoning: bool | None = None
    # Mutually exclusive with include_reasoning: Groq rejects both together.
    reasoning_format: str | None = None
    # Strict Structured Outputs. When set, the provider certifies the shape and
    # json-object mode is not used.
    response_schema: dict | None = None


@dataclass
class Reply:
    """Output plus the provider that actually produced it."""

    text: str
    provider: str
    detail: str = "ok"


@dataclass
class CallRecord:
    """One provider generation, audited on two independent axes.

    `transport_ok` says the provider answered. `accepted` says the pipeline
    could actually use what came back: a schema violation or a grounding
    rejection is a successful transport with an unusable payload.
    """

    purpose: str
    provider: str
    transport_ok: bool
    accepted: bool
    detail: str = "ok"

    def as_dict(self) -> dict:
        return {"purpose": self.purpose, "provider": self.provider,
                "transport_ok": self.transport_ok, "accepted": self.accepted,
                # `ok` is retained for compatibility and means `accepted`:
                # strategy.json audits usable outputs, not HTTP status.
                "ok": self.accepted, "detail": self.detail}


class ProviderError(RuntimeError):
    def __init__(self, category: str, message: str):
        super().__init__(message)
        self.category = category


class FailFast(ProviderError):
    """A malformed or unserviceable request. Retrying cannot help."""


# The provider's several ways of saying "I ran out of completion budget".
# Checked before any schema test, because a truncated strict-mode document is
# reported as a validation failure even though the schema was accepted.
_TRUNCATED_OUTPUT = re.compile(
    r"max[_ ]completion[_ ]tokens|max completion tokens reached|"
    r"output was truncated|was truncated to fit|missing required content|"
    r"increase max_completion_tokens|finish_reason=length",
    re.IGNORECASE)


def classify_http(status: int, body: str) -> str:
    lowered = body.lower()
    if re.search(r"\bitpm\b|input tokens per minute", lowered):
        # The INPUT ceiling: a different bucket again. Whether it is the
        # request or the rolling window is decided by the transport, which
        # knows the configured limit.
        return "input_rate_limited"
    if re.search(r"\botpm\b|output tokens per minute", lowered):
        # A DIFFERENT ceiling from the combined TPM one, and from a request
        # that is simply too big. Whether it can be retried depends on the
        # configured cap, which the transport decides.
        return "output_rate_limited"
    if status == 413 or "request too large" in lowered:
        # One request bigger than the whole per-minute budget can never
        # succeed, so this must not be retried: three attempts spend three
        # times the tokens and starve the calls that would have fit.
        return "request_too_large"
    if status == 429:
        # A DAILY ceiling (TPD/RPD) will not clear inside this run, so it is
        # not the same thing as a per-minute limit that backoff can ride out.
        if re.search(r"\b(?:tpd|rpd)\b|per\s*day|daily", lowered):
            return "daily_limit_exhausted"
        return "quota_exhausted" if "quota" in lowered or "exhausted" in lowered else "rate_limited"
    if status in (401, 403):
        return "invalid_key" if "invalid api key" in lowered else "auth_permission"
    if status == 404:
        return "bad_request"          # retired model or wrong path
    if status == 400:
        # Truncation FIRST. Under strict structured output a run that hit the
        # completion cap mid-document reports json_validate_failed too, but
        # the schema was accepted and generation had already started: calling
        # that a schema rejection sends you hunting the wrong bug.
        if _TRUNCATED_OUTPUT.search(lowered):
            return "output_truncated"
        # A schema the endpoint will not accept is a configuration bug, not a
        # transient failure: it must surface, never be retried into a looser
        # mode.
        if ("json_schema" in lowered or "json_validate_failed" in lowered
                or "failed to validate json" in lowered):
            return "schema_rejected"
        return "bad_request"
    if status >= 500:
        return "server_error"
    return "transport"


def classify_exception(error: Exception) -> str:
    """Classify an SDK exception without ever touching credential material."""
    status = getattr(error, "code", None) or getattr(error, "status_code", None)
    text = str(error).lower()
    if isinstance(status, int):
        return classify_http(status, text)
    if "request too large" in text or "413" in text:
        return "request_too_large"
    if "model_not_found" in text or "not found" in text or "is not supported" in text:
        return "bad_request"
    if _TRUNCATED_OUTPUT.search(text):
        return "output_truncated"
    if re.search(r"\bitpm\b|input tokens per minute", text):
        return "input_rate_limited"
    if re.search(r"\botpm\b|output tokens per minute", text):
        return "output_rate_limited"
    if re.search(r"\b(?:tpd|rpd)\b|per\s*day|daily", text) and (
            "limit" in text or "exhaust" in text or "429" in text):
        return "daily_limit_exhausted"
    if "invalid api key" in text or "invalid_api_key" in text:
        return "invalid_key"
    if "quota" in text or "exhausted" in text:
        return "quota_exhausted"
    if "rate limit" in text or "429" in text:
        return "rate_limited"
    if "api key" in text or "permission" in text or "unauthorized" in text:
        return "auth_permission"
    if "invalid" in text or "400" in text:
        return "bad_request"
    if "internal" in text or "unavailable" in text or "500" in text or "503" in text:
        return "server_error"
    return "transport"


def _trim(text: str, limit: int = 240) -> str:
    flat = re.sub(r"\s+", " ", text).strip()
    return flat[:limit] + ("..." if len(flat) > limit else "")


_RETRY_AFTER_RE = re.compile(
    r"(?:try again|retry)(?:\s+\w+)?\s+in\s+([0-9]+(?:\.[0-9]+)?)\s*(m?s)?|"
    r"retry[-_ ]after[\"':\s]+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


def retry_after_seconds(message: str) -> float | None:
    """The provider's own wait hint, in seconds, when it gave one."""
    found = _RETRY_AFTER_RE.search(message or "")
    if not found:
        return None
    value = found.group(1) or found.group(3)
    if value is None:
        return None
    seconds = float(value)
    if (found.group(2) or "").lower() == "ms":
        seconds /= 1000.0
    # Never sleep longer than the window the limit is measured over.
    return max(0.1, min(seconds, TokenBudget.WINDOW_SECONDS))


def _backoff(attempt: int) -> float:
    return min(BACKOFF_BASE_SECONDS * (2 ** (attempt - 1)), BACKOFF_MAX_SECONDS)


# =============================================================== transports


class GeminiTransport:
    """Gemini with account rotation. Quota/auth failures move to the next key."""

    name = "gemini"

    def __init__(self, credentials: Credentials, log, model: str = GEMINI_MODEL):
        self.credentials = credentials
        self.log = log
        self.model = model
        if not credentials.gemini_accounts:
            raise ProviderError("auth_permission", "no Gemini API key is configured")

    def generate(self, request: Request) -> Reply:
        last: ProviderError | None = None
        for account in self.credentials.gemini_accounts:
            for attempt in range(1, RATE_LIMIT_MAX_ATTEMPTS + 1):
                self.log.info("gemini account #%d attempt %d model=%s purpose=%s",
                              account, attempt, self.model, request.purpose)
                try:
                    return Reply(self._generate(self.credentials.gemini_key(account), request),
                                 self.name)
                except FailFast:
                    raise
                except ProviderError as error:
                    last = error
                    self.log.warning("gemini account #%d attempt %d failed: category=%s %s",
                                     account, attempt, error.category, error)
                    if error.category == "bad_request":
                        raise FailFast(error.category,
                                       f"Gemini rejected the request ({error}); not retrying")
                    if error.category in GEMINI_ROTATE_CATEGORIES:
                        # No point spending two more identical attempts on an
                        # account whose DAILY quota is gone, or whose key is
                        # bad: rotate immediately. The live run burned three
                        # attempts per account before moving on.
                        self.log.info("gemini account #%d is out for this call "
                                      "(category=%s); rotating without retrying it",
                                      account, error.category)
                        break                            # rotate to the next account
                    limit = (SERVER_ERROR_MAX_ATTEMPTS if error.category == "server_error"
                             else RATE_LIMIT_MAX_ATTEMPTS)
                    if attempt >= limit:
                        break
                    time.sleep(_backoff(attempt))
        raise ProviderError(last.category if last else "transport",
                            f"every Gemini account failed; last category="
                            f"{last.category if last else 'unknown'}: {last}")

    def _generate(self, api_key: str, request: Request) -> str:   # pragma: no cover - live path
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        settings: dict[str, Any] = {"temperature": request.temperature}
        if request.json:
            settings["response_mime_type"] = "application/json"
        config = types.GenerateContentConfig(**settings)
        try:
            response = client.models.generate_content(
                model=self.model, contents=request.prompt, config=config)
        except Exception as error:
            raise ProviderError(classify_exception(error), _trim(str(error))) from None
        text = response.text or ""
        if not text.strip():
            raise ProviderError("malformed_response", "Gemini returned an empty response")
        return text


class TokenBudget:
    """A rolling one-minute token budget, shared by every Groq transport.

    Groq bills prompt + reserved output against a tokens-per-minute ceiling.
    Three audits that each fit individually can still exceed the minute
    together, and the failure mode is a 413 that no retry can fix. So a call
    waits just long enough for older usage to age out of the window.

    Advisory by construction: it paces, it never refuses. The pre-call figure
    is an estimate, corrected with the provider's own usage number as soon as
    a reply arrives, so one rough guess cannot skew later pacing.
    """

    WINDOW_SECONDS = 60.0

    def __init__(self, limit: int, *, sleep=time.sleep, clock=time.monotonic):
        self.limit = max(1, int(limit))
        self.used: list[tuple[float, int]] = []
        self._sleep = sleep
        self._clock = clock

    def _prune(self) -> int:
        cutoff = self._clock() - self.WINDOW_SECONDS
        self.used = [(when, tokens) for when, tokens in self.used if when > cutoff]
        return sum(tokens for _, tokens in self.used)

    def wait_for(self, tokens: int, log=None, purpose: str = "") -> float:
        """Sleep until `tokens` fits the window. Returns the seconds waited."""
        waited = 0.0
        # A request larger than the whole budget can never fit, so waiting for
        # it is pointless: let the provider answer and report the real limit.
        tokens = min(int(tokens), self.limit)
        while True:
            in_window = self._prune()
            if in_window + tokens <= self.limit or not self.used:
                return waited
            oldest = min(when for when, _ in self.used)
            pause = max(0.1, oldest + self.WINDOW_SECONDS - self._clock())
            pause = min(pause, self.WINDOW_SECONDS)
            if log is not None:
                log.info("pacing purpose=%s: %d token(s) in the last minute plus %d "
                         "requested exceeds the %d TPM budget; waiting %.1fs",
                         purpose, in_window, tokens, self.limit, pause)
            self._sleep(pause)
            waited += pause

    def record(self, tokens: int, *, replaces: int = 0) -> None:
        """Log usage. `replaces` drops an earlier estimate for the same call."""
        if replaces and self.used:
            for index in range(len(self.used) - 1, -1, -1):
                if self.used[index][1] == replaces:
                    self.used.pop(index)
                    break
        self.used.append((self._clock(), max(0, int(tokens))))

    def release(self, tokens: int) -> None:
        """Drop a reservation the provider never actually charged for."""
        for index in range(len(self.used) - 1, -1, -1):
            if self.used[index][1] == int(tokens):
                self.used.pop(index)
                return


# Calibrated against a real 413: a 36,061-character prompt was billed at about
# 4,044 prompt tokens, or 8.9 characters per token, because these prompts are
# heavily indented and structured. Dividing by 6 stays on the safe side of that
# without the 2x overestimate a prose-tuned divisor of 4 would give, and the
# provider's own usage figure replaces the estimate as soon as a reply lands.
_CHARS_PER_TOKEN = 6


def estimate_tokens(request: "Request") -> int:
    """Prompt plus RESERVED output, which is what Groq bills for a request."""
    return len(request.prompt) // _CHARS_PER_TOKEN + int(request.max_tokens or 0)


# Budgets are PER MODEL, not per account. Qwen and GPT-OSS are deliberately
# called in parallel, so a shared bucket would serialize exactly the two calls
# the architecture wants overlapping. Provider-reported 429/413 always wins
# over these local estimates; they exist only to avoid asking for the
# impossible in the first place.
_MODEL_BUDGETS: dict[str, TokenBudget] = {}


def model_tpm_limit(model: str) -> int:
    """Per-model TPM, configurable because tiers differ between models.

    GROQ_TPM_LIMIT__<MODEL> wins, then GROQ_TPM_LIMIT, then the default.
    """
    slug = re.sub(r"[^A-Z0-9]+", "_", (model or "").upper()).strip("_")
    specific = configured(f"GROQ_TPM_LIMIT__{slug}") if slug else ""
    value = specific or configured("GROQ_TPM_LIMIT") or str(GROQ_TPM_LIMIT)
    try:
        return max(1, int(value))
    except ValueError:
        return GROQ_TPM_LIMIT


def budget_for(model: str) -> TokenBudget:
    """One COMBINED-token budget object per model name, created on first use."""
    key = (model or "").strip() or "__unset__"
    if key not in _MODEL_BUDGETS:
        _MODEL_BUDGETS[key] = TokenBudget(model_tpm_limit(key))
    return _MODEL_BUDGETS[key]


# INPUT tokens per minute is a THIRD ceiling, separate from the combined TPM
# and the output OTPM. Qwen's tier allows 7000, and the first live assessment
# was refused at 7256 with a local estimate of 6169 - the generic estimator
# was 18% optimistic for this model.
#
# Calibrated from that refusal: a 31,502-character prompt was counted as 7,256
# input tokens, i.e. 4.34 characters per token. Qwen's tokenizer splits this
# kind of structured, identifier-heavy text far more finely than the generic
# 6-characters-per-token assumption. One observation is not a tokenizer, so a
# safety multiplier rides on top for anything that reserves capacity.
_MODEL_INPUT_CHARS_PER_TOKEN: dict[str, float] = {
    "qwen/qwen3.8-27b": 4.3,
}

# Applied when RESERVING capacity, never when deciding a request is impossible:
# an inflated figure must not reject a request that would really have fit.
_MODEL_INPUT_SAFETY: dict[str, float] = {
    "qwen/qwen3.8-27b": 1.25,
}

_MODEL_INPUT_BUDGETS: dict[str, TokenBudget] = {}

GROQ_ITPM_DEFAULT = 0

# How much of the ITPM ceiling one request may plan to use. The rest is
# headroom for tokenizer variance, because optimizing to 6999 of 7000 would
# fail on the first prompt that formats slightly differently.
SAFE_INPUT_FRACTION = 0.9


def input_chars_per_token(model: str) -> float:
    return _MODEL_INPUT_CHARS_PER_TOKEN.get((model or "").strip(),
                                            float(_CHARS_PER_TOKEN))


def input_safety_factor(model: str) -> float:
    return _MODEL_INPUT_SAFETY.get((model or "").strip(), 1.0)


def request_input_text(request: "Request") -> str:
    """Everything the provider counts as prompt input for this request.

    The prompt itself plus the serialized response_format: a strict JSON
    schema is request-side text and is billed like any other, so pretending
    it is free is how a preflight passes and the call still fails.
    """
    text = request.prompt
    if request.response_schema:
        text += json.dumps(request.response_schema, separators=(",", ":"))
    return text


def estimate_input_tokens(request: "Request", model: str = "", *,
                          safety: bool = True) -> int:
    """The PROMPT alone, in this model's tokens.

    Deliberately not the combined estimate: the combined ceiling counts prompt
    plus reserved output, and conflating them was why a request that fit the
    8K TPM budget still broke the 7K input one. `safety=False` gives the
    calibrated figure with no multiplier, for deciding whether a request is
    outright impossible.
    """
    raw = len(request_input_text(request)) / input_chars_per_token(model)
    if safety:
        raw *= input_safety_factor(model)
    return int(raw) + 1


def model_itpm_limit(model: str) -> int:
    """Per-model ITPM, or 0 when none is configured.

    GROQ_ITPM_LIMIT__<MODEL> wins, then GROQ_ITPM_LIMIT, then unlimited.
    """
    slug = re.sub(r"[^A-Z0-9]+", "_", (model or "").upper()).strip("_")
    specific = configured(f"GROQ_ITPM_LIMIT__{slug}") if slug else ""
    value = specific or configured("GROQ_ITPM_LIMIT") or str(GROQ_ITPM_DEFAULT)
    try:
        return max(0, int(value))
    except ValueError:
        return GROQ_ITPM_DEFAULT


def safe_input_ceiling(model: str) -> int:
    """The per-request input target: 90% of the ITPM limit, or 0 when unset."""
    limit = model_itpm_limit(model)
    return int(limit * SAFE_INPUT_FRACTION) if limit else 0


def input_budget_for(model: str) -> TokenBudget | None:
    """The model's INPUT budget, or None when it has no configured ceiling."""
    key = (model or "").strip() or "__unset__"
    limit = model_itpm_limit(key)
    if not limit:
        return None
    if key not in _MODEL_INPUT_BUDGETS or _MODEL_INPUT_BUDGETS[key].limit != limit:
        _MODEL_INPUT_BUDGETS[key] = TokenBudget(limit)
    return _MODEL_INPUT_BUDGETS[key]


# OUTPUT tokens per minute is a SEPARATE ceiling from the combined one, and on
# Qwen's tier it is much smaller (1000 vs 8000). It therefore needs its own
# budget: reserving against the combined bucket says nothing about whether the
# output allowance is free.
_MODEL_OUTPUT_BUDGETS: dict[str, TokenBudget] = {}

# No default: an unconfigured model is not assumed to have an output ceiling,
# because guessing one would pace calls that never needed pacing.
GROQ_OTPM_DEFAULT = 0


def model_otpm_limit(model: str) -> int:
    """Per-model OTPM, or 0 when none is configured.

    GROQ_OTPM_LIMIT__<MODEL> wins, then GROQ_OTPM_LIMIT, then unlimited.
    """
    slug = re.sub(r"[^A-Z0-9]+", "_", (model or "").upper()).strip("_")
    specific = configured(f"GROQ_OTPM_LIMIT__{slug}") if slug else ""
    value = specific or configured("GROQ_OTPM_LIMIT") or str(GROQ_OTPM_DEFAULT)
    try:
        return max(0, int(value))
    except ValueError:
        return GROQ_OTPM_DEFAULT


def output_budget_for(model: str) -> TokenBudget | None:
    """The model's OUTPUT budget, or None when it has no configured ceiling."""
    key = (model or "").strip() or "__unset__"
    limit = model_otpm_limit(key)
    if not limit:
        return None
    if key not in _MODEL_OUTPUT_BUDGETS or _MODEL_OUTPUT_BUDGETS[key].limit != limit:
        _MODEL_OUTPUT_BUDGETS[key] = TokenBudget(limit)
    return _MODEL_OUTPUT_BUDGETS[key]


# Retained for compatibility with callers that want the default bucket.
GROQ_BUDGET = budget_for("__default__")


class GroqTransport:
    """One Groq MODEL, with bounded retries and its own pacing budget.

    The pipeline builds two of these - the Qwen evaluator and the GPT-OSS web
    researcher - and they run concurrently. Each therefore owns a
    model-specific budget: the evaluator's local pacing reservation must not
    stall the researcher, or the parallelism buys nothing.
    """

    name = "groq"

    def __init__(self, credentials: Credentials, log, model: str = "",
                 fallback: "GeminiTransport | None" = None,
                 budget: "TokenBudget | None" = None,
                 keys: tuple[str, ...] | None = None,
                 output_budget: "TokenBudget | None" = None,
                 input_budget: "TokenBudget | None" = None):
        self.credentials = credentials
        self.log = log
        self.model = (model or "").strip()
        self.fallback = fallback
        # Flipped for the rest of the process once the endpoint proves it does
        # not accept the browsing tool, so one rejection is not paid per call.
        self.no_web_search = False
        self.budget = budget if budget is not None else budget_for(self.model)
        # OUTPUT tokens per minute, paced separately because it is a separate
        # ceiling. None when this model has no configured OTPM limit.
        self.output_budget = (output_budget if output_budget is not None
                              else output_budget_for(self.model))
        # INPUT tokens per minute, the third ceiling. None when unconfigured.
        self.input_budget = (input_budget if input_budget is not None
                             else input_budget_for(self.model))
        # The provider's own token counts for the last call, when reported.
        self.last_usage: int | None = None
        self.last_output_usage: int | None = None
        self.last_input_usage: int | None = None
        # Deterministic key order. Keys are held here and never logged; a key
        # is only ever identified by its position.
        if keys is None:
            keys = groq_key_pool() or tuple(
                k for k in (credentials.groq_key,) if k)
        self.keys = tuple(keys)
        self._key_index = 0

    @property
    def key_count(self) -> int:
        return len(self.keys)

    @property
    def active_key(self) -> str | None:
        return self.keys[self._key_index] if self.keys else None

    @property
    def available(self) -> bool:
        """Usable only with BOTH a key and an explicitly chosen model."""
        return bool(self.keys and self.model)

    def _settle_output(self, reserved: int, *, rejected: bool = False) -> None:
        """Replace the output reservation with what was actually generated.

        The provider reports completion tokens separately from the total, and
        the real figure is usually far below the cap - keeping the reservation
        would pace the next call against tokens nobody spent. A request the
        provider refused outright generated nothing at all.
        """
        if not self.output_budget or not reserved:
            return
        if self.last_output_usage:
            self.output_budget.record(self.last_output_usage, replaces=reserved)
        elif rejected:
            self.output_budget.release(reserved)

    def _settle_input(self, reserved: int, *, rejected: bool = False) -> None:
        """Replace the input reservation with the prompt tokens actually billed."""
        if not self.input_budget or not reserved:
            return
        if self.last_input_usage:
            self.input_budget.record(self.last_input_usage, replaces=reserved)
        elif rejected:
            self.input_budget.release(reserved)

    def _rotate_key(self, request: Request) -> bool:
        """Try the next key. Only for key-specific failures, never rate limits.

        Two keys in one organization share every TPM/RPM/TPD/RPD ceiling, so
        rotating on a limit error would spend the second key to hit the same
        wall. Bounded by the pool size, so there is no rotation loop.
        """
        if self._key_index + 1 >= self.key_count:
            return False
        self._key_index += 1
        self.log.warning("groq key #%d rejected for purpose=%s; trying key #%d",
                         self._key_index, request.purpose, self._key_index + 1)
        return True

    def generate(self, request: Request) -> Reply:
        if not self.available:
            reason = ("no Groq API key is configured" if not self.keys
                      else f"no Groq model is configured for purpose={request.purpose}")
            if self.fallback:
                self.log.warning("%s; using the Gemini fallback for purpose=%s",
                                 reason, request.purpose)
                return self._fallback(request)
            raise ProviderError("auth_permission", reason)

        last: ProviderError | None = None
        for attempt in range(1, GROQ_MAX_ATTEMPTS + 1):
            estimate = estimate_tokens(request)
            reserved_output = int(request.max_tokens or 0)
            reserved_input = estimate_input_tokens(request, self.model)
            itpm = model_itpm_limit(self.model)
            if itpm:
                ceiling = safe_input_ceiling(self.model)
                # The rejection test uses the CALIBRATED figure with no safety
                # multiplier: an inflated number must never refuse a request
                # that would really have fit.
                honest = estimate_input_tokens(request, self.model, safety=False)
                per_token = input_chars_per_token(self.model)
                self.log.info(
                    "%s input estimate: prompt_estimate=%d schema_estimate=%d "
                    "adjusted_input_estimate=%d configured_itpm=%d safe_ceiling=%d",
                    request.purpose, int(len(request.prompt) / per_token),
                    int((len(request_input_text(request)) - len(request.prompt))
                        / per_token), reserved_input, itpm, ceiling)
                if honest > itpm:
                    raise ProviderError(
                        "request_too_large",
                        f"purpose={request.purpose} needs about {honest} input token(s), "
                        f"over the {itpm} ITPM ceiling for model={self.model}; waiting "
                        f"cannot help - shorten the prompt or raise GROQ_ITPM_LIMIT")
                if reserved_input > ceiling:
                    self.log.warning(
                        "%s input estimate %d exceeds the %d safe ceiling (ITPM %d); "
                        "proceeding, but this prompt has little headroom",
                        request.purpose, reserved_input, ceiling, itpm)
            # A cap larger than the whole output allowance can never be served,
            # so this is a configuration error rather than something to pace.
            if self.output_budget and reserved_output > self.output_budget.limit:
                raise ProviderError(
                    "bad_request",
                    f"max_completion_tokens={reserved_output} for purpose="
                    f"{request.purpose} exceeds the configured OTPM ceiling of "
                    f"{self.output_budget.limit} for model={self.model}; lower the cap "
                    f"or raise GROQ_OTPM_LIMIT")
            self.budget.wait_for(estimate, self.log, request.purpose)
            if self.input_budget:
                self.input_budget.wait_for(reserved_input, self.log,
                                           f"{request.purpose} (input)")
            if self.output_budget:
                self.output_budget.wait_for(reserved_output, self.log,
                                            f"{request.purpose} (output)")
            self.log.info("groq attempt %d model=%s purpose=%s key=#%d "
                          "(~%d token(s) reserved, %d output)", attempt, self.model,
                          request.purpose, self._key_index + 1, estimate,
                          reserved_output)
            self.budget.record(estimate)
            if self.input_budget:
                self.input_budget.record(reserved_input)
            if self.output_budget:
                self.output_budget.record(reserved_output)
            try:
                self.last_usage = None
                self.last_output_usage = None
                self.last_input_usage = None
                text = self._generate(request)
                if self.last_usage:
                    # Replace the estimate with what the provider actually
                    # billed, so later pacing works off real numbers.
                    self.budget.record(self.last_usage, replaces=estimate)
                self._settle_output(reserved_output)
                self._settle_input(reserved_input)
                return Reply(text, self.name)
            except ProviderError as error:
                last = error
                # A completion that arrived and was then rejected - a truncated
                # answer, most often - still cost real tokens, and the provider
                # reported how many. Correct the reservation with that figure
                # either way: over-reporting starves the next call, and
                # under-reporting invites the 413 this pacing exists to avoid.
                if self.last_usage:
                    self.budget.record(self.last_usage, replaces=estimate)
                elif error.category in ("request_too_large", "auth_permission",
                                        "invalid_key", "output_rate_limited",
                                        "input_rate_limited", "schema_rejected"):
                    # Rejected before generation: nothing was consumed, so the
                    # reservation must not sit in the window pretending it was.
                    self.budget.release(estimate)
                rejected = error.category in ("request_too_large", "auth_permission",
                                              "invalid_key", "output_rate_limited",
                                              "input_rate_limited", "schema_rejected")
                self._settle_output(reserved_output, rejected=rejected)
                self._settle_input(reserved_input, rejected=rejected)
                self.log.warning("groq attempt %d failed: category=%s %s",
                                 attempt, error.category, error)
                if error.category in KEY_SPECIFIC_CATEGORIES:
                    if self._rotate_key(request):
                        continue
                    # Every key was rejected. Sending the same credential again
                    # cannot help, so stop instead of burning the attempts.
                    self.log.warning("every configured Groq key was rejected for "
                                     "purpose=%s", request.purpose)
                    break
                if error.category in ("bad_request", "schema_rejected") \
                        and request.web_search and not self.no_web_search:
                    # Browsing is optional: retry once without it so the
                    # research audit degrades to UNKNOWN rather than failing.
                    self.no_web_search = True
                    self.log.warning("groq rejected the browsing tool; retrying "
                                     "purpose=%s without web search", request.purpose)
                    continue
                if error.category in ("output_rate_limited", "input_rate_limited"):
                    # The request itself already fits the ceiling (checked
                    # before sending), so this is the rolling window rather
                    # than the request: pace and retry. Keys are never rotated
                    # for a rate limit, because one organization shares them.
                    if attempt < GROQ_MAX_ATTEMPTS:
                        # The provider's own retry-after wins; otherwise wait for
                        # the rolling output window to clear.
                        pause = retry_after_seconds(str(error))
                        if pause is None:
                            paced = (self.input_budget if error.category ==
                                     "input_rate_limited" else self.output_budget)
                            pause = (TokenBudget.WINDOW_SECONDS if paced
                                     else _backoff(attempt))
                        self.log.warning("groq output rate limit for purpose=%s; waiting "
                                         "%.1fs before retrying (keys are NOT rotated "
                                         "for a rate limit)", request.purpose, pause)
                        time.sleep(pause)
                        continue
                    break
                if error.category in NON_RETRYABLE_CATEGORIES:
                    # The same bytes cannot succeed, and a daily ceiling will
                    # not clear inside this run.
                    break
                if attempt < GROQ_MAX_ATTEMPTS:
                    time.sleep(_backoff(attempt))
        if self.fallback:
            self.log.warning("groq unavailable (category=%s); falling back to Gemini for purpose=%s",
                             last.category if last else "unknown", request.purpose)
            return self._fallback(request)
        raise ProviderError(last.category if last else "transport", f"Groq failed: {last}")

    def _fallback(self, request: Request) -> Reply:
        """Gemini answered, so the reply must say Gemini - not the wrapper."""
        reply = self.fallback.generate(request)
        return Reply(reply.text, reply.provider, "fallback_from_groq")

    def _generate(self, request: Request) -> str:                 # pragma: no cover - live path
        from groq import Groq

        client = Groq(api_key=self.active_key)
        # Only ask for a JSON object when the parser needs one. This model
        # rejects its own generation (400 json_validate_failed) when a prose
        # answer like a cover letter is forced through JSON mode.
        settings: dict[str, Any] = {}
        if request.response_schema:
            # Strict Structured Outputs. The provider validates the shape, so
            # json_object mode is not sent alongside it.
            settings["response_format"] = request.response_schema
        elif request.json:
            settings["response_format"] = {"type": "json_object"}
        # Reasoning effort is the setting that matters: at the default medium
        # effort GPT-OSS spent the audits' output budget thinking and truncated
        # before finishing the JSON. Sent only when a request asks for it, so
        # generation keeps the provider default untouched.
        if request.reasoning_effort:
            settings["reasoning_effort"] = request.reasoning_effort
        # reasoning_format and include_reasoning are mutually exclusive; the
        # format wins when both are somehow set.
        if request.reasoning_format:
            settings["reasoning_format"] = request.reasoning_format
        elif request.include_reasoning is not None:
            settings["include_reasoning"] = request.include_reasoning
        if request.web_search and not self.no_web_search:
            # Provider-side browsing, used by the company-research audit ONLY.
            # tool_choice="required" forces the model to actually search rather
            # than answer from memory: stale sponsorship policy is exactly the
            # failure this audit exists to avoid. A model or endpoint that
            # rejects either field produces a bad_request, which generate()
            # retries once with browsing off so research degrades to UNKNOWN.
            settings["tools"] = [{"type": "browser_search"}]
            settings["tool_choice"] = "required"
        try:
            completion = client.chat.completions.create(
                model=self.model, temperature=request.temperature,
                # max_completion_tokens, not the deprecated max_tokens.
                max_completion_tokens=request.max_tokens,
                messages=[{"role": "user", "content": request.prompt}], **settings)
        except Exception as error:
            raise ProviderError(classify_exception(error), _trim(str(error))) from None
        usage = getattr(completion, "usage", None)
        self.last_usage = int(getattr(usage, "total_tokens", 0) or 0) or None
        # Completion tokens are what the OTPM ceiling counts.
        self.last_output_usage = int(
            getattr(usage, "completion_tokens", 0) or 0) or None
        # Prompt tokens are what the ITPM ceiling counts.
        self.last_input_usage = int(getattr(usage, "prompt_tokens", 0) or 0) or None
        choice = completion.choices[0]
        # A reasoning model can spend the whole budget before answering; a
        # truncated cover letter is worse than a retry or a fallback.
        if choice.finish_reason == "length":
            # Its own category: resending an identical request produces an
            # identical truncation, and the wasted tokens are what pushed the
            # following attempt over the per-minute ceiling.
            raise ProviderError("output_truncated",
                                f"Groq hit the {request.max_tokens}-token completion cap "
                                f"(finish_reason=length)")
        return choice.message.content or ""


# ==================================================================== mock

# Curated, evidence-grounded bullet candidates. Every number here is copied
# from that project's own evidence with its exact structure (`45+`, `sub-500 ms`,
# `approximately 26%`). Candidates are offered per project so the mock can
# respect the resume-wide action-verb budget, exactly as a real model is asked to.
MOCK_BULLETS: dict[str, list[str]] = {
    "lms": [
        "Engineered a decoupled learning management system with a Django REST Framework backend and a React SPA frontend, using PostgreSQL for durable persistence and Redis for ephemeral real-time state and ranking.",
        "Implemented the real-time subsystem with Django Channels, Daphne, Redis, and WebSockets, serving live quizzes, dynamic leaderboards, and rate-limited chat to 45+ concurrent users with sub-500 ms broadcast latency.",
        "Optimized REST and submission paths under Locust and asyncio load tests of up to 100 concurrent users, cutting p95 REST latency by approximately 26% and quiz-submission processing from approximately 2000 ms to 1500 ms.",
        "Designed stateless JWT authentication with a custom JWT-over-WebSocket handshake, and added server-side grading with idempotent quiz submissions and a backtracking scheduler for conflict-free schedules.",
    ],
    "temp": [
        "Developed an end-to-end IoT environmental monitoring system during a weekend hackathon, with a Python, Django, and Django REST Framework backend that parsed and validated incoming sensor data.",
        "Integrated Arduino microcontrollers and MPI 3118A sensors to transmit localized ambient-temperature data, storing history and location metadata in PostgreSQL for an interactive ReactJS dashboard.",
        "Delivered a ReactJS dashboard that polled the REST API for real-time building temperatures, with Auth0 protecting access to the system.",
    ],
    "pintos": [
        "Implemented the user-programs layer of Pintos, an x86 teaching operating-system kernel written in C, covering process execution, stack setup for argument passing, parent-child synchronization, waiting, and clean termination.",
        "Engineered a system-call handler covering 13 process-control and file I/O operations, with thread-safe file-descriptor management for up to 128 open files per process and executable write-denial protection.",
        "Hardened the kernel using synchronization primitives and byte-wise memory validation to prevent deadlocks, kernel panics, and invalid-memory failures, passing 100% of 80 concurrency, memory-fault, and edge-case tests.",
    ],
    "fraud": [
        "Engineered a graph-based fraud detection pipeline over the IEEE-CIS dataset of 590,540 transactions with an approximately 3.5% fraud rate, converting tabular records into a graph across shared card, address, and email-domain identifiers.",
        "Implemented Focal Loss, class weighting, and threshold optimization to handle an approximately 27.6:1 class imbalance, capping large groups to control memory use during graph construction.",
        "Benchmarked a baseline MLP, GraphSAGE, and GAT in PyTorch Geometric, where GraphSAGE reached 0.9259 AUC-ROC and 0.5641 fraud F1 while GAT underperformed with hub-collapse behavior.",
    ],
    "tailor_pipeline": [
        "Engineered an automated resume-tailoring pipeline that treats model output as untrusted until independently verified, compiling generated LaTeX with pdflatex and re-extracting the rendered PDF with pdftotext so checks read the real artifact.",
        "Implemented deterministic factual-grounding and narrative-coherence checks that reject unsupported technology claims and bullets merging unrelated accomplishments into inflated claims.",
        "Integrated an independent Groq model to proofread compiled text, with targeted repair logic and cost-aware retries that send only genuinely failing sections back for revision.",
    ],
    "music": [
        "Developed a machine learning pipeline classifying raw audio into music genres, extracting MFCC, Spectral Centroid, Spectral Rolloff, Zero-Crossing Rate, and Chroma features with Librosa.",
        "Evaluated CatBoost and KNN models on the extracted feature set, reaching 97.68% classification accuracy and publishing the results in IJRAR.",
        "Implemented the feature-extraction and evaluation workflow in Python with Pandas, NumPy, and scikit-learn.",
    ],
}

MOCK_LETTER = """Dear Hiring Manager,

{paragraph_one}

{paragraph_two}

{project_paragraph}

I would be glad to talk through any of this in more detail.

Sincerely,
Jay Niketan Pathare"""

MOCK_OPENING = (
    "I am applying for the {job_title} role at {company}. My experience building Python "
    "backend systems, REST APIs and real-time services lines up closely with what this "
    "role asks for.{eligibility}")

MOCK_EXPERIENCE_PARAGRAPH = (
    "At Thesis Mumbai Tech I led platform module delivery end to end, from gathering "
    "requirements in client meetings through prototyping to validate scope and reviewing "
    "and merging the team's work into production. I built healthcare modules supporting "
    "10k+ records and designed a real-time WebSocket and Redis pipeline that fed IoT "
    "sensor data into PostgreSQL with ~2s live monitoring, and I architected the "
    "relational schemas and SQL queries behind those services.")


_MONTH_NAMES = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June",
                7: "July", 8: "August", 9: "September", 10: "October", 11: "November",
                12: "December"}

# A repair-capable letter fixture. The live BAE run failed three times on the
# same unqualified metric, so the mock now reproduces that first mistake and
# then ACTS on the repair feedback, which is the behaviour under test: read
# the prompt, obey the SAFE form, obey the prohibition list, add the missing
# priority as its own paragraph.
MOCK_FRAUD_EXACT = (
    "I also built a graph-based fraud detection pipeline over the IEEE-CIS dataset "
    "of 590,540 transactions, where GraphSAGE handled a 27.6:1 class imbalance and "
    "reached 0.9259 AUC-ROC against an MLP baseline.")

MOCK_FRAUD_QUALIFIED = (
    "I also built a graph-based fraud detection pipeline over the IEEE-CIS dataset "
    "of 590,540 transactions, where GraphSAGE handled an approximately 27.6:1 class "
    "imbalance and reached 0.9259 AUC-ROC against an MLP baseline.")

# Pintos gets its OWN paragraph. Splicing it into the fraud paragraph is what
# failed attempt 2 of the live run on source isolation.
MOCK_PINTOS_PARAGRAPH = (
    "Separately, I implemented the user-programs layer of Pintos, an x86 teaching "
    "kernel written in C, covering process execution, parent-child synchronization "
    "and a system-call handler, and passed 100% of 80 concurrency and memory-fault "
    "tests.")


def _letter_repair_state(prompt: str) -> dict:
    """What the repair feedback in this prompt is actually asking for."""
    return {
        "safe_form": "SAFE 'approximately 27.6'" in prompt or "approximately 27.6" in prompt,
        "metric_rejected": "REJECTED '27.6'" in prompt,
        "form_prohibited": "ALREADY REJECTED IN THIS SESSION" in prompt,
        "needs_priority_two": "priority 2 is still missing" in prompt,
        "source_isolation": "mixes evidence from" in prompt,
    }


MOCK_PROJECT_PARAGRAPHS = {
    "lms": ("Outside work I built a decoupled learning management system whose real-time "
            "layer runs on Django Channels, Daphne, Redis and WebSockets, serving 45+ "
            "concurrent users with sub-500 ms broadcast latency. Load testing it with "
            "Locust and asyncio at up to 100 concurrent users is where I learned most "
            "about keeping latency predictable under concurrency."),
    "pintos": ("I also implemented the user-programs layer of Pintos, an x86 teaching "
               "kernel written in C, covering process execution, parent-child "
               "synchronization and a system-call handler, and passed 100% of 80 "
               "concurrency and memory-fault tests. That work is the closest thing I have "
               "to low-level debugging under real correctness pressure."),
    "tailor_pipeline": ("I also built an automated resume-tailoring pipeline that treats "
                        "model output as untrusted until independently verified, compiling "
                        "LaTeX with pdflatex and re-extracting the rendered PDF with "
                        "pdftotext so every check reads the real artifact."),
    "temp": ("I also built an end-to-end IoT monitoring system in a weekend hackathon, with "
             "a Python, Django and Django REST Framework backend that parsed and validated "
             "incoming sensor data and stored history in PostgreSQL."),
    "fraud": ("I also built a graph-based fraud detection pipeline over the IEEE-CIS dataset "
              "of 590,540 transactions, where GraphSAGE reached 0.9259 AUC-ROC against an "
              "MLP baseline."),
    "music": ("I also built an audio classification pipeline using Librosa features with "
              "CatBoost and KNN models, reaching 97.68% accuracy and a published paper."),
}

# The AI-assisted-development evidence lives in the general skills bank rather
# than in any one project, so it has no project paragraph of its own. Wording
# tracks that capsule exactly: tools used, for what, and nothing more.
MOCK_AI_PARAGRAPH = (
    "I work with AI coding agents day to day rather than around them. I use GitHub "
    "Copilot for code generation and completion, and Cursor and Claude Code for "
    "repository-level development, debugging, test-driven iteration and code review, "
    "which is how I keep a fast feedback loop on work I still have to verify myself.")

# Which already-written paragraph carries each distinctive theme's evidence.
# customer_facing and healthcare are absent on purpose: the experience
# paragraph already carries both, so no extra paragraph is needed.
MOCK_THEME_PARAGRAPHS = {
    "ai_native": ("tailor_pipeline",),
    "low_level_systems": ("pintos",),
    "ml_data": ("fraud", "music"),
    "realtime": ("lms", "temp"),
    "cloud_infra": ("temp",),
}

# Fallbacks for themes whose strongest evidence is a role or the skills bank
# rather than a project, so there is no project paragraph to reach for. Each
# stays inside ONE source and carries no metric.
MOCK_SOURCE_PARAGRAPHS = {
    "ai_native": MOCK_AI_PARAGRAPH,
    "ml_data": ("As a data engineering intern I developed an ETL workflow in Python with "
                "Pandas, NumPy and MySQL to process and clean large business datasets, "
                "then generated the reports that surfaced key operational trends from "
                "them. Extraction, cleaning, transformation and reporting were all mine "
                "to get right, which is where I learned to distrust data I had not "
                "validated myself."),
}



class MockTransport:
    """Deterministic, realistic structured output. Makes zero network calls."""

    name = "mock"

    def __init__(self, master: MasterFacts, log, seed: int = 7):
        self.master = master
        self.log = log
        self.random = random.Random(seed)

    def generate(self, request: Request) -> Reply:
        self.log.info("mock transport serving purpose=%s (no network call)", request.purpose)
        # Dispatch on PURPOSE, not on the json flag: company_research also asks
        # for prose-wrapped JSON, because Groq refuses JSON mode with tools.
        if request.purpose == "cover_letter":
            return Reply(self._letter(request.context), self.name)
        handler: Callable[[dict], dict] = {
            "project_selection": self._select,
            "project_bullets": self._bullets,
            "bullet_repair": self._repair,
            "assessment": self._assess,
            "company_research": self._company_research,
        }[request.purpose]
        return Reply(json.dumps(handler(request.context)), self.name)

    def _select(self, context: dict) -> dict:
        ranked = engine.rank_projects(self.master, context["jd_text"], context["role_family"])
        count = context["count"]
        chosen = ranked[:count]
        return {
            "role_family": context["role_family"],
            "career_stage": "early_career_masters_student",
            "selected": [
                {"project_id": pid, "llm_rank": index + 1,
                 "reason": f"strongest overlap with the job description ({matched})"}
                for index, (pid, _score, matched) in enumerate(chosen)],
            "considered": [
                {"project_id": pid, "selected": pid in {c[0] for c in chosen},
                 "reason": matched} for pid, _score, matched in ranked],
            "skill_priority": context.get("skill_hints", []),
            "semantic_jd_signals": self._semantic_signals(context["jd_text"]),
        }

    def _semantic_signals(self, jd_text: str) -> dict:
        """Taxonomy signals with VERBATIM evidence, or nothing.

        The fixture reports only what it can actually quote from the posting,
        which is the same contract the live prompt imposes. It deliberately
        adds nothing the deterministic classifier does not already see, so a
        mock run's Experience decision is provably unchanged by this path.
        """
        signals = engine.classify_jd(jd_text, self.master.section_order)
        deterministic = grounding.deterministic_signal_map(signals)
        lowered = jd_text.lower()
        out: dict[str, dict] = {}
        for name, present in deterministic.items():
            if not present:
                out[name] = {"present": False, "evidence": []}
                continue
            family = grounding.SEMANTIC_SIGNALS[name].get("family")
            terms = [t for t, _w in engine.DOMAIN_TERMS.get(family, ())] if family else []
            if name == "code_quality_collaboration_heavy":
                terms = ["code review", "pull request", "best practice", "pair programming"]
            elif name == "healthcare":
                terms = ["patient", "clinical", "healthcare", "hospital"]
            quote = ""
            for term in terms:
                position = lowered.find(term)
                if position < 0:
                    continue
                window = jd_text[max(0, position - 40): position + len(term) + 40]
                words = window.split()
                # A verbatim interior slice, so the quote is grounded even
                # though the window edges may have clipped a word.
                if len(words) >= 5:
                    quote = " ".join(words[1:-1])
                    break
            out[name] = ({"present": True, "evidence": [quote]} if quote
                         else {"present": False, "evidence": []})
        return out

    def _bullets(self, context: dict) -> dict:
        project_id = context["project_id"]
        candidates = list(MOCK_BULLETS.get(project_id, []))
        wanted = context["count"]
        budget: dict[str, int] = dict(context.get("verb_budget", {}))
        limit = context.get("verb_limit", 2)
        chosen: list[str] = []
        for bullet in candidates:
            if len(chosen) == wanted:
                break
            verb = engine.opening_verb(bullet)
            if budget.get(verb, 0) >= limit:
                continue
            budget[verb] = budget.get(verb, 0) + 1
            chosen.append(bullet)
        for bullet in candidates:                       # budget could not be met
            if len(chosen) == wanted:
                break
            if bullet not in chosen:
                chosen.append(bullet)
        return {"project_id": project_id, "bullets": chosen[:wanted]}

    def _repair(self, context: dict) -> dict:
        """Repair one bullet without touching facts, as the real prompt asks."""
        bullet = context["bullet"]
        goal = context.get("goal", "shorten")
        if goal == "reword_verb":
            # Swap only the opening verb; every fact stays exactly as it was.
            options = context.get("available_verbs") or []
            head, _, rest = bullet.partition(" ")
            if options and rest:
                return {"bullet": f"{options[0]} {rest}"}
            return {"bullet": bullet}
        if goal == "shorten":
            # Drop the trailing subordinate clause, keeping the main claim and
            # its metrics rather than truncating mid-thought.
            best = None
            for connector in (" so ", " while ", ", plus ", ", and ", " alongside ",
                              ", with ", " for "):
                head, sep, tail = bullet.rpartition(connector)
                if sep and len(head) > 110 and (best is None or len(head) > len(best)):
                    best = head
            if best:
                return {"bullet": best.rstrip(" ,") + "."}
            trimmed = re.sub(r"\s*\([^()]*\)", "", bullet, count=1)
            return {"bullet": trimmed if trimmed != bullet else bullet}
        alternatives = [b for b in MOCK_BULLETS.get(context["project_id"], [])
                        if b not in context.get("siblings", []) and b != bullet]
        return {"bullet": alternatives[0] if alternatives else bullet}

    def _letter(self, context: dict) -> str:
        """Assemble a letter from the current approved facts, per JD.

        Reads the repair feedback in the prompt, exactly as a real model
        would: the first draft of a fraud-project letter states the class
        imbalance without its qualifier, and the retry corrects it to the SAFE
        form and adds the missing priority as its own paragraph.
        """
        jd_text = (context.get("jd_text") or "").lower()
        repair = _letter_repair_state(context.get("prompt") or "")
        ranked = list(context.get("project_ids") or ["lms"])
        # A real model reads ROLE-DISTINCTIVE PRIORITIES and leads with the
        # evidence it points at; this fixture imitates that so the mock proves
        # the selection, not just the prompt text.
        project_paragraph = ""
        for priority in grounding.supported_priorities(context.get("priorities")):
            if priority["theme"] not in MOCK_THEME_PARAGRAPHS:
                # The experience paragraph already carries this theme, so spend
                # the project paragraph on the next priority instead.
                continue
            candidates = MOCK_THEME_PARAGRAPHS[priority["theme"]]
            pick = next((pid for pid in candidates if pid in ranked), "")
            if pick:
                project_paragraph = MOCK_PROJECT_PARAGRAPHS[pick]
            else:
                project_paragraph = MOCK_SOURCE_PARAGRAPHS.get(priority["theme"], "")
            if project_paragraph:
                break
        # Fallback: the pre-priority keyword heuristic, unchanged.
        keyed = [("concurrency", "pintos"), ("kernel", "pintos"), ("systems", "pintos"),
                 ("real-time", "lms"), ("websocket", "lms"), ("latency", "lms"),
                 ("machine learning", "fraud"), ("model", "fraud"),
                 ("verification", "tailor_pipeline"), ("llm", "tailor_pipeline")]
        chosen = next((pid for needle, pid in keyed
                       if needle in jd_text and pid in ranked), ranked[0])
        # The metric-repair path. A first draft that uses the fraud project
        # states the class imbalance unqualified, which is the live defect;
        # once the prompt carries the SAFE form (or prohibits the rejected
        # one), the fixture writes the qualified form instead.
        if "fraud" in ranked and not project_paragraph:
            project_paragraph = (MOCK_FRAUD_QUALIFIED
                                 if (repair["safe_form"] or repair["form_prohibited"])
                                 else MOCK_FRAUD_EXACT)
        # The breadth repair: priority 2 arrives as its OWN paragraph, never
        # spliced into another source's.
        extra = ""
        if repair["needs_priority_two"] and "pintos" in ranked:
            extra = "\n\n" + MOCK_PINTOS_PARAGRAPH
        eligibility = ""
        if context.get("needs_eligibility"):
            eligibility = (" I am completing an M.S. in Computer Science at the University "
                           "at Buffalo, expected February 2027, after earning a B.E. in "
                           "Information Technology.")
        body = project_paragraph or MOCK_PROJECT_PARAGRAPHS.get(
            chosen, MOCK_PROJECT_PARAGRAPHS["lms"])
        return MOCK_LETTER.format(
            paragraph_one=MOCK_OPENING.format(
                job_title=context.get("job_title") or "Software Engineer",
                company=context.get("company") or "your team",
                eligibility=eligibility),
            paragraph_two=MOCK_EXPERIENCE_PARAGRAPH,
            project_paragraph=body + extra)

    # -- audit fixtures --------------------------------------------------
    # Deterministic stand-ins for the three Groq audit calls. They imitate a
    # reviewer reading the same inputs a live model gets, so the pipeline's
    # audit path is exercised end to end with no network call.

    def _company_research(self, context: dict) -> dict:
        """No network access in mock mode, so everything is honestly UNKNOWN."""
        return {
            "company_visa_sponsorship": "UNKNOWN",
            "company_visa_confidence": "LOW",
            "company_stem_opt_support": "UNKNOWN",
            "company_stem_opt_confidence": "LOW",
            "job_posted": context.get("jd_posted") or "UNKNOWN",
            "job_posted_confidence": "LOW",
            "checked_at": "mock-run (no web search performed)",
            "sources": [],
        }

    def _assess(self, context: dict) -> dict:
        """Classify Python's authoritative requirements against the final resume.

        Every entry cites a requirement_id, so a requirement identity can never
        be invented here either. Only explicit capability/qualification
        requirements reach the fit score; logistics, conditions and eligibility
        are routed to manual review.
        """
        all_requirements = list(context.get("requirements") or [])
        skills = list(context.get("skills") or [])
        tailoring = context.get("tailoring") or {}
        evidence = context.get("evidence") or engine.ResumeEvidence(skills=tuple(skills))

        scored = engine.scored_requirements(all_requirements)
        # One authority for both paths: the same deterministic verdicts the live
        # assessment is forced to adopt.
        verdicts = engine.deterministic_assessment(all_requirements, self.master, evidence,
                                                   context.get("jd_text") or "")
        strong = verdicts["strong_matches"][:12]
        partial = verdicts["partial_matches"][:8]
        gaps = verdicts["gaps"][:8]
        manual = verdicts["manual_review"]

        # Role signals the posting only describes: colour, not requirements.
        complementary = []
        for requirement in engine.role_signals(all_requirements):
            for term in requirement.terms:
                if self.master.canonical_skill(term):
                    complementary.append(f"{term} (role signal {requirement.requirement_id})")
        for extra in ("AWS (EC2/RDS)", "PostgreSQL", "WebSockets"):
            if extra in skills and not any(extra in c for c in complementary):
                complementary.append(f"{extra} (supporting evidence the posting does not ask "
                                     f"for)")

        required_total = sum(1 for r in scored if r.importance == "required") or 1
        strong_ids = {m["requirement_id"] for m in strong}
        required_strong = sum(1 for r in scored
                              if r.importance == "required"
                              and r.requirement_id in strong_ids)
        coverage = required_strong / required_total
        required_gaps = [g for g in gaps if g["importance"] == "required"]
        score = round(min(9.6, 6.0 + 3.6 * coverage) - 0.4 * len(required_gaps), 1)
        score = max(3.0, score)
        recommendation = ("strong_apply" if score >= 9 else "apply" if score >= 7.5
                          else "borderline" if score >= 6 else "skip")

        # Eligibility is computed once, by deterministic_assessment, with the
        # posting text in hand. Recomputing it here risked two answers.
        status = verdicts["eligibility"]["status"]
        details = list(verdicts["eligibility"]["details"])
        flags = list(verdicts["eligibility_flags"])

        notes = []
        if tailoring.get("skills_rendered_lines"):
            notes.append(f"Technical Skills renders "
                         f"{tailoring['skills_rendered_lines']} lines as required")
        if tailoring.get("experience_exact"):
            notes.append("Professional Experience matches approved wording exactly")
        if tailoring.get("projects"):
            notes.append(f"projects selected for this posting: {tailoring['projects']}")
        notes.append(f"{len(strong)} of {len(scored)} scored requirement(s) are directly "
                     f"evidenced on the resume")
        if manual:
            notes.append(f"{len(manual)} requirement(s) routed to manual review")
        tailoring_score = 9.5 if not tailoring.get("issues") else 7.0

        # ---- audit #2 dimensions (see grounding.validate_application_audit).
        # Judged from the SAME inputs a live model gets: the whole catalogue,
        # the ranking, each project's own evidence and the final bullets.
        catalogue = list(context.get("project_catalogue") or [])
        detail = context.get("selection_detail") or {}
        bullets_by_id = context.get("project_bullets") or {}
        jd_words = set(re.findall(r"[a-z]{5,}", (context.get("jd_text") or "").lower()))
        chosen = [e for e in catalogue if e.get("selected")]
        passed_over = [e for e in catalogue if not e.get("selected")]

        def _overlap(entry: dict) -> int:
            text = (" ".join(entry.get("tech") or []) + " " + (entry.get("evidence") or "")
                    + " " + (entry.get("name") or "")).lower()
            return sum(1 for word in jd_words if word in text)

        best_available = sorted(catalogue, key=_overlap, reverse=True)[:len(chosen)]
        best_ids = {e["project_id"] for e in best_available}
        chosen_ids = {e["project_id"] for e in chosen}
        selection_notes = []
        for entry in passed_over:
            if entry["project_id"] in best_ids:
                selection_notes.append(
                    f"{entry['project_id']} overlapped the posting more than one of the "
                    f"selected projects but was not chosen")
        overlap_chosen = sum(_overlap(e) for e in chosen) or 1
        jd_relevance = min(10.0, 6.0 + overlap_chosen / 10.0)
        best_chosen = 10.0 * (len(chosen_ids & best_ids) / max(1, len(chosen_ids)))
        tech_sets = [set(map(str.lower, e.get("tech") or [])) for e in chosen]
        shared = (len(set.intersection(*tech_sets)) if len(tech_sets) > 1 else 0)
        complementary_coverage = max(5.0, 10.0 - 1.5 * shared)
        allocation = detail.get("allocation") or {}
        ranked_first = (detail.get("selected") or [None])[0]
        ranking = 9.5 if allocation and allocation.get(ranked_first) == max(
            allocation.values(), default=0) else 7.5
        selection_score = round(0.40 * jd_relevance + 0.25 * best_chosen
                                + 0.20 * complementary_coverage + 0.15 * ranking, 1)

        # Bullet quality, including a real evidence-fidelity check: every number
        # a bullet states must appear in that project's own evidence.
        fidelity_notes = []
        numbers_ok = 0
        numbers_seen = 0
        for entry in chosen:
            evidence_text = ((entry.get("evidence") or "") + " "
                             + " ".join(entry.get("tech") or [])).lower()
            for bullet in bullets_by_id.get(entry["project_id"], []):
                for number in re.findall(r"\d[\d,.]*", bullet):
                    numbers_seen += 1
                    if number.lower() in evidence_text:
                        numbers_ok += 1
                    else:
                        fidelity_notes.append(
                            f"{entry['project_id']}: {number!r} is not stated verbatim in "
                            f"that project's evidence")
        fidelity = 10.0 if not numbers_seen else round(10.0 * numbers_ok / numbers_seen, 1)
        all_bullets = [b for bs in bullets_by_id.values() for b in bs]
        bullet_text = " ".join(all_bullets).lower()
        bullet_relevance = min(10.0, 6.0 + sum(1 for w in jd_words if w in bullet_text) / 12.0)
        specificity = min(10.0, 6.0 + sum(1 for b in all_bullets
                                          if re.search(r"\d", b)) / max(1, len(all_bullets)) * 4)
        ownership = 9.0 if all_bullets else 0.0
        opening = [b.split()[0].lower() for b in all_bullets if b.split()]
        redundancy = max(4.0, 10.0 - 2.0 * (len(opening) - len(set(opening))))
        bullet_score = round(0.35 * bullet_relevance + 0.25 * specificity
                             + 0.20 * fidelity + 0.10 * ownership
                             + 0.10 * redundancy, 1)

        # Experience selection, scored as one component of this audit. Judged
        # from the deterministic context only: the fixture never proposes a
        # bullet id, exactly as the live prompt forbids.
        exp_context = context.get("experience_context") or {}
        merged_signals = exp_context.get("merged_signals") or {}
        rule = exp_context.get("rule") or ""
        shipped_ids = list(exp_context.get("shipped_ids") or [])
        positives = [name for name, value in merged_signals.items() if value]
        rule_matches_signal = any(
            token in rule.lower()
            for token in ("code-quality", "healthcare", "backend", "ai/ml", "agentic",
                          "cloud", "data")) if rule else False
        experience_score = round(min(10.0, 6.0
                                     + (2.0 if rule_matches_signal else 0.0)
                                     + (1.0 if shipped_ids else 0.0)
                                     + min(1.0, 0.25 * len(positives))), 1)
        experience_notes = [
            f"the {rule or 'selected'} rule fits the merged signals "
            f"({', '.join(positives) or 'none detected'})"]

        likelihood = ("HIGH" if score >= 8.0 and not required_gaps else
                      "MEDIUM" if score >= 6.5 else "LOW")

        verdict = {"strong_apply": "a strong apply", "apply": "worth applying to",
                   "borderline": "borderline", "skip": "a poor match"}[recommendation]
        summary = (f"{len(strong)} of {len(scored)} scored requirement(s) are directly "
                   f"evidenced and {len(gaps)} are unsupported, so this posting is "
                   f"{verdict}.")
        return {
            "fit_score": score, "recommendation": recommendation, "summary": summary,
            "eligibility": {"status": status, "details": details},
            "strong_matches": strong, "partial_matches": partial, "gaps": gaps,
            "manual_review": manual, "complementary_strengths": complementary[:6],
            "tailoring_quality": {"score": tailoring_score, "notes": notes},
            "project_selection": {
                "score": selection_score,
                "components": {"jd_relevance": round(jd_relevance, 1),
                               "best_available_chosen": round(best_chosen, 1),
                               "complementary_coverage": round(complementary_coverage, 1),
                               "ranking_and_allocation": round(ranking, 1)},
                "notes": selection_notes[:4] or ["the selected projects are the strongest "
                                                 "available overlap for this posting"]},
            "project_bullets": {
                "score": bullet_score,
                "components": {"jd_relevance": round(bullet_relevance, 1),
                               "technical_specificity": round(specificity, 1),
                               "evidence_fidelity": round(fidelity, 1),
                               "impact_ownership": round(ownership, 1),
                               "non_redundancy": round(redundancy, 1)},
                "notes": fidelity_notes[:4] or ["every number traces to its own project's "
                                                "evidence"]},
            "experience_selection": {"score": experience_score,
                                      "notes": experience_notes},
            "callback_likelihood": likelihood,
            "risk_flags": flags,
        }


# ================================================================== prompts

# ================================================================== prompts

_STYLE_RULES = """WRITING RULES (follow exactly):
- Tone: direct, technical, professional, concise, evidence first. Plain past tense.
- Preferred opening verbs: Built, Developed, Designed, Engineered, Implemented, Integrated,
  Architected, Parallelized, Delivered, Optimized, Automated, Scaled, Reduced.
- Never use: leveraged, spearheaded, cutting-edge, seamless, innovative, transformative,
  world-class, results-driven, utilize. Do not write "production-grade" unless it is meaningful.
- No em dashes anywhere. No LaTeX. No markdown. No bullet characters. Plain sentences.
- One sentence per bullet, ending with a period.
- Lead with the JD-relevant concept and use the job description's own supported vocabulary,
  but never keyword stuff.

FACTUAL RULES (absolute):
- Use ONLY the evidence listed for THIS project. Do not borrow a metric, technology, or
  responsibility from another project or from professional experience.
- Copy every number exactly as the evidence states it, including its form: "3+" is not "3",
  "45+" is not "45", "approximately 90%" is not "90%", "sub-500 ms" is not "500 ms".
  Never strengthen or weaken a threshold.
- Do not invent scope, team size, ownership, business impact, or results.
- Do not name a technology the evidence does not name."""


# Repair guidance, appended only when an attempt was rejected. Says HOW to fix
# each class of failure the validators can raise, because a validator message
# states what is wrong, not what to write instead.
_LETTER_REPAIR_RULES = """HOW TO REPAIR (apply only to the problems listed above):
  * A metric flagged with a REPAIR note: use the SAFE form given, verbatim, in place of
    the REJECTED form. Do not delete the metric - it is truthful once qualified - and do
    not touch any other sentence.
  * A paragraph that mixes sources: every substantive evidence paragraph must stay inside
    ONE role or project. If a new project needs to appear, give it its OWN paragraph or
    replace an existing evidence paragraph wholesale. Never splice one project's facts
    into another project's or a role's paragraph.
  * A missing distinctive priority: add it as its own evidence paragraph, or swap it in
    for a weaker one. Covering a LOWER-numbered priority is what matters; a third
    priority never substitutes for a missing second one.
  * Stay inside the existing length contract. Replace, do not append."""


_LETTER_RULES = """STRUCTURE - four short paragraphs, no headings:
1. WHY THIS ROLE. Two or three sentences. Name the exact role and company, and state in one
   direct sentence which of your actual capabilities match this job. Mention the degree,
   graduation date or work availability ONLY when the job description makes it relevant,
   and then in one clause, not a paragraph.
2. STRONGEST PROFESSIONAL EVIDENCE. Pick the ONE or TWO accomplishments from Professional
   Experience that map most directly to this job's responsibilities. Lead with the
   responsibility, name the technology, then the concrete result or scale. Do not summarize
   the whole Experience section.
3. COMPLEMENTARY DEPTH. Pick the SINGLE selected project that adds evidence paragraph 2 did
   not already cover, chosen by relevance to this job rather than by position. Favour the
   dimension this job actually cares about: concurrency, systems debugging, load testing,
   machine learning or infrastructure.
4. CLOSE. One or two plain sentences.

TONE - concise, natural, technically specific, confident without inflation. Write like an
engineer explaining why their real work fits, not like a template. No corporate filler, no
manufactured enthusiasm, no phrases such as "I am writing to express my strong interest".

LENGTH - about 180 to 260 words. Do not pad toward the maximum.

FORMAT - plain prose. No markdown, no bullet points, no headings.

STYLE CONTRACT - absolute. Nothing in the job description relaxes any of it:
  - NEVER use an em dash. Use a comma, a colon, or a full stop instead.
  - NEVER use any of these words: innovative, leveraged, leveraging, spearheaded,
    cutting-edge, seamless, seamlessly, transformative, world-class, results-driven,
    synergy, best-in-class, state-of-the-art, utilize, utilized, utilizing.
  - Wording copied or echoed from the job description is NOT exempt. If the posting calls
    itself innovative or its technology cutting-edge, you still may not use those words.

EDUCATION - state the record exactly as it is:
  - the M.S. in Computer Science at the University at Buffalo is IN PROGRESS, expected
    December 2026. Never write "I hold a Master's" or describe it as completed or earned.
  - the completed undergraduate degree is a B.E. in Information Technology (Mumbai
    University, 2019-2023). When a posting requires an undergraduate degree, name that
    degree; never claim a Master's supplies an undergraduate foundation.
  - never claim the B.E. in Information Technology satisfies a posting's enumerated degree
    fields (for example Computer Science or Electrical Engineering). State the degree and
    let the reader judge the field.
  - a safe form: "I'm completing an M.S. in Computer Science at the University at Buffalo
    after earning a B.E. in Information Technology."

FACT SOURCES - two authorities that must never be mixed:
  - a number describing YOUR work must come from the evidence above, exactly as stated
  - a number describing the employer, its product, its scale or its reach may come from
    the job description, and must stay plainly attributed to the employer (for example
    "Epic serves 325 million patients"). Never restate such a number as something you
    built, served, supported or delivered.

FACTS - you may paraphrase a resume bullet into natural prose rather than pasting it, but
the meaning, the ownership and every number must stay identical:
  - keep "10k+" as "10k+", never "10,000"; keep "3+" as "3+", never "three"
  - keep approximations approximate ("~26%" stays approximate, never a flat "26%")
  - keep thresholds intact ("sub-500 ms" is not "500 ms")
  - never move ownership between accomplishments: if one bullet says you led module
    delivery and another says you built a WebSocket pipeline, you did NOT lead the
    WebSocket pipeline
  - an exact value of 1000 or more keeps the evidence's exact digits and separators:
    "590,540" stays "590,540" and "28,167" stays "28,167". Never rewrite an exact value as
    "590k", "590 k", "0.59 million" or "about 590,000". Only write an approximate form when
    the evidence itself is approximate, and then keep its authorized wording ("~90%",
    "10k+", "45+")
  - CANDIDATE DATES COME ONLY FROM THE STANDING FACTS. A graduation date, an employment
    start date and any immigration or work-authorization date must be the candidate's own
    recorded facts, never derived from this posting's eligibility window. Never infer an
    OPT start date. If graduation is mentioned at all, it is "expected February 2027".
    Prefer omitting OPT entirely
  - ONE SOURCE PER PARAGRAPH. Each experience or project paragraph must stay inside a
    single role or project. Never move a technology, metric or capability from one source
    into another: the OCR/LLM pipeline paragraph may not mention the AWS EC2/RDS work, and
    a project paragraph may not borrow another project's stack. If a capability is not in
    that source's own evidence, do not claim it there
  - claim no technology that the evidence below does not name
  - do not claim "profiling" unless the evidence says profiling. Supported adjacent
    wording: testing concurrent code, concurrency testing, load testing, optimizing
    systems, performance optimization\""""


def _project_evidence_block(project: Project) -> str:
    lines = [f"PROJECT: {project.name}", f"project_id: {project.project_id}",
             f"context: {project.context}", f"date: {project.date}",
             f"technologies named by the evidence: {', '.join(project.tech)}",
             "SUPPORTED EVIDENCE (the only facts you may use):"]
    lines += [f"  - {item}" for item in project.evidence]
    if project.tags:
        lines.append(f"relevance tags: {', '.join(project.tags)}")
    return "\n".join(lines)


def _json_instruction(shape: str) -> str:
    return (f"Return ONLY valid JSON, no prose and no code fences, shaped exactly like:\n{shape}")


def parse_json(text: str, purpose: str) -> dict:
    """Parse a model response, tolerating fences and surrounding prose."""
    cleaned = re.sub(r"^\s*```(?:json)?|```\s*$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            pass
    raise ProviderError("malformed_response",
                        f"{purpose} response was not JSON: {_trim(text)}")


# =================================================================== client


@dataclass
class Selection:
    selected: list[str]
    ranks: dict[str, int]
    reasons: dict[str, str]
    considered: list[tuple[str, bool, str]]
    career_stage: str | None = None
    # Raw semantic JD signals, exactly as the model sent them. Validation and
    # merging happen in the pipeline; nothing here has any authority yet.
    semantic_signals_raw: dict = field(default_factory=dict)


# ============================== strict structured output for the assessment
#
# Best-effort JSON-object mode failed live with json_validate_failed: the
# model produced something the endpoint would not certify as JSON. Groq
# supports strict JSON Schema for this model, which makes the SHAPE the
# provider's problem instead of ours.
#
# The schema expresses exactly the contract that already existed - the same
# fields validate_application_audit and assessment.txt read, with the verdict
# buckets still ids-only. No dimension is added or removed. Strict mode
# requires every property to be listed in `required` and every object to set
# additionalProperties: false, so genuinely-empty things are expressed as
# empty arrays rather than omitted keys.


# The schema carries SHAPE only. Every semantic instruction - rubrics, word
# limits, what a bucket means - stays in the prompt, where it costs the tokens
# once instead of once per property.
# No minimum/maximum: Python's own _score() already rejects anything outside
# 0-10, and repeating the bound on thirteen properties is pure input cost.
_SCORE = {"type": "number"}
_STRINGS = {"type": "array", "items": {"type": "string"}}


def _score_property(description: str = "") -> dict:
    return dict(_SCORE)


def _notes_property(limit_words: int = 0) -> dict:
    return dict(_STRINGS)


def _strict_object(properties: dict) -> dict:
    """An object with every property required and nothing else permitted."""
    return {"type": "object", "properties": properties,
            "required": sorted(properties), "additionalProperties": False}


def _id_bucket(description: str = "") -> dict:
    """A verdict bucket: requirement ids and nothing else."""
    return {"type": "array",
            "items": _strict_object({"requirement_id": {"type": "string"}})}


APPLICATION_ASSESSMENT_SCHEMA = _strict_object({
    "fit_score": _score_property("candidate-to-JD fit, per the rubric"),
    "recommendation": {"type": "string", "enum": list(grounding.RECOMMENDATIONS)},
    "summary": {"type": "string"},
    "eligibility": _strict_object({
        "status": {"type": "string", "enum": list(grounding.ELIGIBILITY_STATUS)},
        "details": dict(_STRINGS),
    }),
    "strong_matches": _id_bucket("requirement ids the resume evidences directly"),
    "partial_matches": _id_bucket("requirement ids with related but insufficient evidence"),
    "gaps": _id_bucket("requirement ids with no supporting evidence"),
    "manual_review": _id_bucket("requirement ids the recorded facts cannot settle"),
    "complementary_strengths": dict(_STRINGS),
    "tailoring_quality": _strict_object({
        "score": _score_property("how well the resume used the AVAILABLE evidence"),
        "notes": _notes_property(15),
    }),
    "experience_selection": _strict_object({
        "score": _score_property("whether the shipped Experience fits the posting"),
        "notes": _notes_property(15),
    }),
    "project_selection": _strict_object({
        "score": _score_property("overall project-selection quality"),
        "components": _strict_object({
            name: dict(_SCORE) for name in grounding.PROJECT_SELECTION_COMPONENTS}),
        "notes": _notes_property(15),
    }),
    "project_bullets": _strict_object({
        "score": _score_property("overall project-bullet quality"),
        "components": _strict_object({
            name: dict(_SCORE) for name in grounding.PROJECT_BULLET_COMPONENTS}),
        "notes": _notes_property(15),
    }),
    "callback_likelihood": {"type": "string",
                            "enum": list(grounding.CALLBACK_LIKELIHOOD)},
    "risk_flags": dict(_STRINGS),
})

ASSESSMENT_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "application_assessment",
        "strict": True,
        "schema": APPLICATION_ASSESSMENT_SCHEMA,
    },
}


# ============================== compact audit context
#
# Qwen's tier caps INPUT tokens per minute at 7000, and the first live run was
# refused at 7256. The assessor judges a FINISHED application, so it does not
# need the generation corpus a second time: the requirement table, the verdict
# evidence paragraphs, the full project evidence and the project bullets were
# each being sent twice in different shapes.
#
# Nothing scoreable is dropped. Every audit dimension keeps exactly the input
# it needs to be judged: the JD in full, the shipped resume, the final
# bullets, the deterministic verdicts, and enough about the projects that were
# NOT chosen to say whether a better one existed.

_VERDICT_LABELS = {"strong_matches": "strong", "partial_matches": "partial",
                   "gaps": "gap", "manual_review": "manual"}


def _short(text: str, limit: int) -> str:
    """One line, clipped on a word boundary rather than mid-word."""
    flat = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(flat) <= limit:
        return flat
    cut = flat[:limit].rsplit(" ", 1)[0]
    return cut + "..."


def compact_verdict_table(verdicts: dict, table: dict, requirements) -> str:
    """One line per requirement: id, importance, verdict, short text.

    Replaces both the authoritative requirement table and the verdict block,
    which restated the same requirements with long evidence paragraphs Python
    owns anyway. The model needs the verdict, not Python's reasoning for it.
    """
    importance = {req.requirement_id: req.importance for req in requirements}
    kinds = {req.requirement_id: req.kind for req in requirements}
    lines: list[str] = []
    for bucket, label in _VERDICT_LABELS.items():
        for entry in verdicts.get(bucket) or []:
            identifier = entry.get("requirement_id", "")
            requirement = table.get(identifier)
            text = requirement.original_text if requirement is not None else ""
            lines.append(
                f"  {identifier} | {importance.get(identifier, 'unclear')} | "
                f"{kinds.get(identifier, 'capability')} | {label} | "
                f"{_short(text, 90)}")
    return "\n".join(lines) or "  (none)"


def compact_selected_projects(projects, project_bullets: dict[str, list[str]],
                              detail: dict) -> str:
    """The three that shipped: what they are, how they rank, what they say.

    Their FINAL BULLETS are the artifact under review. A short evidence digest
    comes with them so the evidence-fidelity component can still be judged,
    but not the full corpus the writer was given.
    """
    ranks = {pid: index for index, pid in enumerate(detail.get("selected") or [], start=1)}
    allocation = detail.get("allocation") or {}
    blocks: list[str] = []
    for project in projects:
        pid = project.project_id
        blocks.append(
            f"  {pid} | rank {ranks.get(pid, '-')} | {allocation.get(pid, '-')} bullet(s)"
            f" | {project.name.split('(')[0].strip()}\n"
            f"      evidence digest: {_short(' '.join(project.evidence), 200)}\n"
            f"      FINAL BULLETS:\n"
            + "\n".join(f"        - {b}" for b in project_bullets.get(pid, [])))
    return "\n".join(blocks) or "  (none)"


def compact_unselected_projects(catalogue: list[dict]) -> str:
    """The alternatives, compactly: enough to spot a better available choice."""
    lines: list[str] = []
    for entry in catalogue or []:
        if entry.get("selected"):
            continue
        tech = ", ".join((entry.get("tech") or [])[:6])
        lines.append(
            f"  {entry['project_id']} | {_short(entry.get('name') or '', 60)}\n"
            f"      tech: {tech or 'none recorded'}\n"
            f"      strongest evidence: {_short(entry.get('evidence') or '', 150)}")
    return "\n".join(lines) or "  (none - every available project was selected)"


class LLMClient:
    """Prompt construction, response validation and grounding for every call."""

    def __init__(self, gemini, groq, master: MasterFacts, policy: Policy, log,
                 audit=None, research=None):
        self.gemini = gemini
        # Retained so existing callers and tests keep working; generation no
        # longer routes anything through it.
        self.groq = groq
        # The two AUDIT transports, each a distinct Groq model with no Gemini
        # fallback. An audit that quietly failed over to the generator would
        # not be independent of it.
        self.audit = audit if audit is not None else groq
        self.research = research if research is not None else self.audit
        self.master = master
        self.policy = policy
        self.log = log
        self.calls: list[CallRecord] = []

    # -- plumbing --------------------------------------------------------
    def _invoke(self, transport, request: Request) -> Reply:
        try:
            reply = transport.generate(request)
        except ProviderError as error:
            self.calls.append(CallRecord(request.purpose, transport.name, False, False,
                                         f"category={error.category}: {error}"))
            self.log.error("[%s] %s call failed: category=%s %s", request.purpose.upper(),
                           transport.name, error.category, error)
            raise
        # reply.provider is whoever answered, which is not the wrapper when a
        # fallback handled the call.
        self.calls.append(CallRecord(request.purpose, reply.provider, True, True, reply.detail))
        return reply

    def reject_last(self, detail: str) -> None:
        """Mark the most recent generation unusable, keeping its history."""
        if self.calls:
            self.calls[-1].accepted = False
            self.calls[-1].detail = detail

    def _call(self, transport, request: Request) -> dict:
        return parse_json(self._invoke(transport, request).text, request.purpose)

    # -- project selection ----------------------------------------------
    def select_projects(self, jd_text: str, signals: Signals, count: int) -> Selection:
        catalogue = "\n\n".join(_project_evidence_block(p) for p in self.master.projects)
        signal_names = ", ".join(grounding.SEMANTIC_SIGNAL_NAMES)
        prompt = f"""You are selecting which academic projects belong on a tailored resume.

JOB DESCRIPTION:
{jd_text}

DETECTED ROLE FAMILY: {signals.role_family}

ALL AVAILABLE PROJECTS (every one is factually supported):
{catalogue}

Evaluate EVERY project above and select exactly {count}, ranked 1 to {count} (1 is the most
relevant, and it receives the most resume space). Weigh: relevance to this job description,
ATS keyword overlap with the posting, technical depth, strength of the supporting evidence,
complementary coverage across the {count} picks, and whether specialized work helps or
distracts for THIS role. Give a short concrete reason for every project, selected or not.

SEMANTIC JD SIGNALS - in this SAME response, also classify the posting against our FIXED
taxonomy. This is a reading task, not a decision: Python owns every policy consequence.
  {signal_names}

For each signal report present true or false. For every present=true signal, quote 1 to 3
SHORT phrases COPIED VERBATIM from the job description above as evidence. A phrase you
paraphrase, summarise or invent will be discarded, and so will the signal. Do not invent
signal names, do not explain your reasoning, and do not mention Professional Experience,
resume bullets or bullet ids anywhere: you have not been shown them and they are not
yours to choose.

{_json_instruction('''{"role_family": "...", "career_stage": "...",
 "selected": [{"project_id": "...", "llm_rank": 1, "reason": "..."}],
 "considered": [{"project_id": "...", "selected": true, "reason": "..."}],
 "semantic_jd_signals": {"backend": {"present": true,
                          "evidence": ["exact short phrase copied from the posting"]},
                         "code_quality_collaboration_heavy": {"present": false,
                          "evidence": []}}}''')}"""

        data = self._call(self.gemini, Request(
            "project_selection", prompt, temperature=0.2,
            context={"jd_text": jd_text, "role_family": signals.role_family, "count": count}))

        valid = {p.project_id for p in self.master.projects}
        selected, ranks, reasons = [], {}, {}
        for entry in data.get("selected", []):
            pid = str(entry.get("project_id", "")).strip()
            if pid in valid and pid not in ranks:
                ranks[pid] = int(entry.get("llm_rank", len(selected) + 1))
                reasons[pid] = str(entry.get("reason", "")).strip() or "no reason given"
                selected.append(pid)
        selected.sort(key=lambda pid: ranks[pid])
        if len(selected) != count:
            raise ProviderError("malformed_response",
                                f"project selection returned {len(selected)} valid project(s), "
                                f"expected exactly {count}")
        considered = [(str(e.get("project_id", "")), bool(e.get("selected")),
                       str(e.get("reason", "")).strip())
                      for e in data.get("considered", [])
                      if str(e.get("project_id", "")) in valid]
        raw_signals = data.get("semantic_jd_signals")
        return Selection(selected, ranks, reasons, considered,
                         str(data.get("career_stage", "")).strip() or None,
                         semantic_signals_raw=raw_signals
                         if isinstance(raw_signals, dict) else {})

    # -- project bullets -------------------------------------------------
    def write_bullets(self, project: Project, count: int, jd_text: str,
                      verb_counts: dict[str, int], target_chars: tuple[int, int],
                      ) -> tuple[list[str], list[grounding.Problem]]:
        """Write `count` grounded bullets, re-prompting with the failures found."""
        exhausted = engine.exhausted_verbs(verb_counts, self.policy.verb_max_uses)
        low, high = target_chars
        base = f"""Write exactly {count} resume bullet(s) for one academic project.

JOB DESCRIPTION (for relevance and vocabulary only, never for facts):
{jd_text}

{_project_evidence_block(project)}

{_STYLE_RULES}

LENGTH: each bullet should be about {low} to {high} characters so it renders as two full
lines. Do not pad with filler and do not add facts to reach the length.
ACTION VERBS ALREADY USED ELSEWHERE ON THIS RESUME (do not start a bullet with these):
{', '.join(exhausted) or 'none'}
Do not start two bullets with the same verb.

{_json_instruction('{"project_id": "...", "bullets": ["...", "..."]}')}"""

        problems: list[grounding.Problem] = []
        bullets: list[str] = []
        for attempt in range(1, BULLET_ATTEMPTS + 1):
            prompt = base
            if problems:
                prompt += ("\n\nYour previous attempt was rejected by deterministic validation. "
                           "Fix exactly these problems and change nothing else:\n"
                           + "\n".join(f"  - {p.message}" for p in problems))
            data = self._call(self.gemini, Request(
                "project_bullets", prompt, temperature=0.35,
                context={"project_id": project.project_id, "count": count,
                         "verb_budget": dict(verb_counts),
                         "verb_limit": self.policy.verb_max_uses}))
            bullets = [re.sub(r"\s+", " ", str(b)).strip()
                       for b in data.get("bullets", []) if str(b).strip()]
            if len(bullets) != count:
                problems = [grounding.Problem("count", "error",
                                              f"returned {len(bullets)} bullets, expected {count}")]
                self.reject_last(f"{project.project_id}: {problems[0].message}")
                self.log.warning("[PROJECT WRITING] %s attempt %d: %s",
                                 project.project_id, attempt, problems[0].message)
                continue
            problems = []
            for bullet in bullets:
                problems.extend(grounding.check_bullet_grounding(
                    bullet, project, self.master, self.policy.banned_phrases))
            seen_verbs: dict[str, int] = {}
            for bullet in bullets:
                verb = engine.opening_verb(bullet)
                seen_verbs[verb] = seen_verbs.get(verb, 0) + 1
                if verb in exhausted:
                    problems.append(grounding.Problem(
                        "verb_budget", "error",
                        f"bullet starts with {verb!r}, already used "
                        f"{self.policy.verb_max_uses} times on this resume"))
            for verb, uses in seen_verbs.items():
                if uses > 1:
                    problems.append(grounding.Problem(
                        "verb_budget", "error",
                        f"{uses} bullets in this project start with {verb!r}"))
            blocking = grounding.errors(problems)
            for problem in problems:
                self.log.info("[PROJECT WRITING] %s attempt %d: %s",
                              project.project_id, attempt, problem)
            if not blocking:
                return bullets, problems
            self.reject_last(f"{project.project_id}: " + "; ".join(
                p.message for p in blocking[:2]))
            problems = blocking
        return bullets, problems

    def repair_bullet(self, project: Project, bullet: str, goal: str, reason: str,
                      siblings: list[str], available: list[str] | None = None
                      ) -> tuple[str, list[grounding.Problem]]:
        """Bounded, truthful repair of ONE bullet for layout or verb reasons."""
        options = list(available or [])
        directions = {
            "shorten": "Make it shorter so it fits on exactly two rendered lines.",
            "lengthen": ("Make it longer, using only unused evidence from the same project, so "
                         "it fills two rendered lines instead of leaving a short tail."),
            "reword_verb": (
                "Start the bullet with a DIFFERENT action verb and keep everything else "
                "intact. Every fact, number, technology and the JD-relevant idea must survive "
                f"unchanged. Natural choices still available: {', '.join(options) or 'none'}. "
                "Do not force an awkward synonym: if no natural verb fits, return the bullet "
                "unchanged."),
        }
        direction = directions[goal]
        prompt = f"""Revise ONE resume bullet for layout only. The facts must not change.

{_project_evidence_block(project)}

CURRENT BULLET:
{bullet}

OTHER BULLETS IN THIS PROJECT (do not duplicate their content):
{chr(10).join('  - ' + s for s in siblings) or '  (none)'}

WHY IT IS BEING REVISED: {reason}
WHAT TO DO: {direction}

{_STYLE_RULES}

Do not drop a number to save space; drop a clause instead. Do not add a new claim.

{_json_instruction('{"bullet": "..."}')}"""
        data = self._call(self.gemini, Request(
            "bullet_repair", prompt, temperature=0.2,
            context={"project_id": project.project_id, "bullet": bullet, "goal": goal,
                     "siblings": siblings, "available_verbs": options}))
        revised = re.sub(r"\s+", " ", str(data.get("bullet", ""))).strip()
        if not revised:
            return bullet, [grounding.Problem("repair", "error", "repair returned nothing")]
        problems = grounding.check_bullet_grounding(revised, project, self.master,
                                                    self.policy.banned_phrases)
        for sibling in siblings:
            if engine.normalize_plain(sibling).lower() == engine.normalize_plain(revised).lower():
                problems.append(grounding.Problem("repair", "error",
                                                  "repair duplicates a sibling bullet"))
        return revised, problems

    # -- cover letter ----------------------------------------------------
    def cover_letter(self, jd: engine.JobPosting, signals: Signals,
                     experience_plain: list[str], projects: list[Project],
                     themes: list[engine.Requirement] | None = None,
                     unsupported: tuple[str, ...] = (),
                     capsules: dict[str, str] | None = None,
                     priorities: list[dict] | None = None) -> tuple[str, list]:
        evidence = "\n".join(f"  - {item}" for item in experience_plain)
        project_facts = "\n\n".join(_project_evidence_block(p) for p in projects)
        facts = "\n".join(f"  - {f}" for f in self.master.standing_facts)
        themes = themes or []
        theme_block = "\n".join(
            f"  {index}. [{req.importance}] {req.headline}"
            + (f"  (names: {', '.join(req.terms)})" if req.terms else "")
            for index, req in enumerate(themes, start=1)) or "  (none extracted)"
        grad = engine.graduation_requirement(jd.text)
        # WHICH supported evidence to lead with. This selects among facts that
        # are already authorized; it never widens what may be claimed.
        priorities = list(priorities or [])
        priority_block = grounding.render_letter_priorities(priorities)
        # Qualified numbers, with the forms they may be written in. Exposed up
        # front so a writer never has to reconstruct "approximately 27.6:1"
        # from a rejection message - which is how the live letter produced a
        # bare "27.6:1" on attempts 1 and 3.
        numbers_block = grounding.render_quantitative_facts(
            grounding.quantitative_facts(self.master, capsules or {}))
        base = f"""Write a cover letter for this job, grounded strictly in the evidence below.

JOB DESCRIPTION:
{jd.text}

RESUME EXPERIENCE BULLETS (exact approved wording; these are the professional facts):
{evidence}

SELECTED PROJECTS:
{project_facts}

STANDING FACTS (state these accurately if you mention them):
{facts}

THE REQUIREMENTS THAT MATTER MOST IN THIS POSTING:
{theme_block}

Cover at least TWO of those requirements with direct evidence, ideally one from
Professional Experience and one from a project. Depth beats keyword coverage: do not try
to mention every requirement.

{priority_block}

{numbers_block}

{_LETTER_RULES}

{'ELIGIBILITY: this posting states a graduation requirement (' + grad['text'] + '). At most ONE concise sentence may mention the degree. Use the candidate standing facts verbatim and NEVER the posting window as the candidate date: expected graduation is February 2027. Omit OPT entirely unless work authorization or start timing is material to this posting; if it is, the only supported wording is "eligible to begin post-completion OPT employment from February 2, 2027, subject to OPT/EAD authorization".' if grad else 'ELIGIBILITY: this posting states no graduation or authorization requirement, so do not discuss immigration or logistics unless it strengthens the match.'}

{('CAPABILITIES THIS RESUME DOES NOT SUPPORT: ' + ', '.join(unsupported) + '. The posting names these, so you may describe them as work the EMPLOYER does or that you want to learn, but you may NEVER present them as your own experience. Do not write "my experience with" or "I built" about any of them. Use supported neighbouring wording instead.') if unsupported else ''}

Address it to "Dear Hiring Manager," and sign as Jay Niketan Pathare.
Do not claim HIPAA, FHIR, EHR, CI/CD, Kubernetes or microservices experience.

Return ONLY the letter itself as plain text: no JSON, no code fences, no commentary."""

        # Bounded repair, exactly like write_bullets(): one wording slip should
        # not cost the whole run. Only DETERMINISTIC VALIDATION failures retry
        # here; transport retry/failover stays owned by the transport layer.
        letter = ""
        problems: list[grounding.Problem] = []
        # Repair state for THIS letter only - never across jobs. Attempt 3 of
        # the live run repeated attempt 1's rejected metric because nothing
        # remembered it had already failed.
        banned_forms: list[str] = []
        history: list[str] = []
        for attempt in range(1, LETTER_ATTEMPTS + 1):
            prompt = base
            repair = [p for p in problems
                      if p.severity == "error" or p.kind == "relevance"]
            if repair:
                prompt += ("\n\nPrevious attempt was rejected. Fix exactly these problems "
                           "and change nothing else. Keep every remaining sentence "
                           "grounded in the evidence above:\n"
                           + "\n".join(f"  - {p.message}" for p in repair))
                prompt += "\n\n" + _LETTER_REPAIR_RULES
            if banned_forms:
                prompt += ("\n\nALREADY REJECTED IN THIS SESSION - these exact forms failed "
                           "validation on an earlier attempt and must NOT appear again:\n"
                           + "\n".join(f"  - {form}" for form in dict.fromkeys(banned_forms)))
            if history:
                prompt += ("\n\nAttempts so far: " + "; ".join(history)
                           + ". Do not reintroduce a problem you already fixed.")
            # Gemini writes the cover letter. Groq is the audit provider only,
            # so no cover-letter request may reach it or its fallback chain.
            reply = self._invoke(self.gemini, Request(
                "cover_letter", prompt, temperature=0.5, json=False,
                context={"job_title": jd.job_title, "company": jd.company_name,
                         "project_ids": [p.project_id for p in projects],
                         "jd_text": jd.text, "needs_eligibility": bool(grad),
                         "themes": [r.headline for r in themes],
                         "priorities": priorities,
                         # The fixture reads its own repair feedback; real
                         # transports ignore `context` entirely.
                         "prompt": prompt}))
            letter = reply.text.strip()
            if len(letter) < 200:
                raise ProviderError("malformed_response",
                                    f"cover letter was only {len(letter)} characters")
            problems = grounding.validate_cover_letter(
                letter, self.master, masked_terms=(jd.job_id,) if jd.job_id else (),
                banned=self.policy.banned_phrases, jd_text=jd.text,
                company=jd.company_name, unsupported_concepts=unsupported,
                capsules=capsules or {})
            problems += grounding.validate_letter_quality(
                letter, company=jd.company_name, job_title=jd.job_title)
            problems = grounding.dedupe_problems(problems)
            blocking = grounding.errors(problems)
            # Advisory relevance: a grounded letter that led with the wrong
            # supported evidence is repairable, never rejected. Attempts are
            # spent on it only while some remain.
            relevance = grounding.letter_relevance(letter, priorities)
            if not blocking and relevance and attempt < LETTER_ATTEMPTS:
                self.reject_last("cover-letter relevance: " + relevance[0].message)
                self.log.warning("[COVER LETTER] attempt %d retried for relevance: %s",
                                 attempt, relevance[0].message)
                history.append(f"attempt {attempt}: breadth")
                problems = problems + relevance
                continue
            if not blocking:
                return letter, problems + relevance
            # The provider answered, but an invalid letter is not a usable
            # artifact. Transport success and acceptance are audited separately.
            banned_forms.extend(grounding.rejected_metric_forms(blocking))
            history.append(f"attempt {attempt}: "
                           + ", ".join(sorted({p.kind for p in blocking})))
            self.reject_last("cover-letter validation failed: " + "; ".join(
                p.message for p in blocking[:3]))
            self.log.warning("[COVER LETTER] attempt %d rejected: %s", attempt,
                             "; ".join(p.message for p in blocking[:2]))
            if banned_forms:
                self.log.info("[COVER LETTER] forms now prohibited for the remaining "
                              "attempts: %s", ", ".join(dict.fromkeys(banned_forms)))
        return letter, problems

    # -- audit #2: current company research ------------------------------
    def research_company(self, jd: engine.JobPosting, *, jd_posted: str = "UNKNOWN",
                         today: str = "") -> dict:
        """Current sponsorship / STEM OPT / posting-date research. Audit only.

        The ONLY audit call permitted to use current web search. Whether the
        configured model can actually browse is a provider capability; when it
        cannot, it is instructed to answer UNKNOWN rather than guess, and
        Python re-checks the two rules a model most often over-reads.
        """
        shape = ('{"company_visa_sponsorship": "UNKNOWN",\n'
                 ' "company_visa_confidence": "LOW",\n'
                 ' "company_stem_opt_support": "UNKNOWN",\n'
                 ' "company_stem_opt_confidence": "LOW",\n'
                 ' "job_posted": "UNKNOWN",\n'
                 ' "job_posted_confidence": "LOW",\n'
                 ' "checked_at": "YYYY-MM-DD",\n'
                 ' "sources": [{"title": "...", "url": "https://...",\n'
                 '               "evidence": "...", "scope": "company_policy"}]}')
        prompt = f"""Research an employer's CURRENT work-authorization posture and this
posting's publication date. Today is {today or "the current date"}.

COMPANY: {jd.company_name or "unknown"}
JOB TITLE: {jd.job_title or "unknown"}
JOB ID: {jd.job_id or "none"}

THE POSTING ITSELF:
{jd.text}

Use current web search. Prefer, in this order:
  1. official company sources (careers site, immigration or FAQ pages)
  2. the current official posting for this exact role
  3. government or public records
  4. reliable third-party evidence

HARD RULES - these are the mistakes to avoid:
  * E-Verify participation is NOT STEM OPT support. If E-Verify is the only evidence you
    find, company_stem_opt_support is UNKNOWN.
  * Historical H-1B or PERM filings are NOT proof of current sponsorship policy. Treat
    them as weak evidence and lower your confidence accordingly.
  * A job-specific restriction always overrides a company-level finding. If this posting
    says sponsorship is unavailable, say so regardless of what the company has done before.
  * If you cannot browse, or find nothing, answer UNKNOWN with LOW confidence. Never guess.

SOURCE SCOPING - every source must declare what it is about, with "scope" set to exactly
one of:
  "this_job"           the exact posting being applied to
  "company_policy"     an official company-wide statement of policy
  "unrelated_posting"  a DIFFERENT job at the same company
  "context"            anything else
An unrelated posting is supporting context only and can never establish a company-wide
YES or NO, however plainly it seems to state one. A restriction in "this_job" is
authoritative for this application and overrides any company-wide finding.

For job_posted: use an explicit, trustworthy date in the posting first; otherwise search
for this exact company, title and job id. Return YYYY-MM-DD or UNKNOWN. Never substitute
today's date, a crawl date or a file date for the publication date.

Every source you cite must be a real URL you actually consulted.

{_json_instruction(shape)}"""

        reply = self._invoke(self.research, Request(
            "company_research", prompt, temperature=0.1, web_search=True,
            max_tokens=AUDIT_OUTPUT_TOKENS["company_research"],
            **AUDIT_REASONING,
            # Groq refuses response_format=json_object alongside tool calling
            # ("json mode cannot be combined with tool/function calling"), and
            # browsing is the whole point of this call. The shape is required
            # in the prompt instead, and parse_json tolerates fences or prose.
            json=False,
            context={"company": jd.company_name, "job_title": jd.job_title,
                     "job_id": jd.job_id, "jd_text": jd.text, "today": today,
                     "jd_posted": jd_posted}))
        data = parse_json(reply.text, "company_research")
        research, problems = grounding.validate_company_research(
            data, jd_text=jd.text, jd_posted=jd_posted)
        for problem in problems:
            self.log.warning("[COMPANY RESEARCH] %s", problem.message)
        for override in research["deterministic_overrides"]:
            self.log.warning("[COMPANY RESEARCH] %s", override)
        return research

    # -- assessment ------------------------------------------------------
    def assess(self, jd: engine.JobPosting, signals: Signals, selected: list[Project],
               skills: list[str], *, experience_plain: list[str],
               project_bullets: dict[str, list[str]],
               requirements: list[engine.Requirement],
               tailoring: dict | None = None,
               extra_experience: list[str] | None = None,
               project_catalogue: list[dict] | None = None,
               selection_detail: dict | None = None,
               experience_context: dict | None = None) -> dict:
        """Audit #2: the Final Application Audit. Advisory only.

        Judges candidate-to-JD fit AND how well generation used the evidence it
        had: resume tailoring, project selection, project bullet quality and
        callback likelihood. Every verdict below is still Python's - the model
        explains them and scores its own audit dimensions, and nothing it
        returns re-enters generation.
        """
        scored = engine.scored_requirements(requirements)
        manual = engine.manual_review_requirements(requirements)
        signal_reqs = engine.role_signals(requirements)
        signal_block = "\n".join(
            f"  {req.requirement_id} names {', '.join(req.terms)}"
            for req in signal_reqs) or "  (none)"
        evidence = engine.ResumeEvidence(
            experience_text=" ".join(list(experience_plain) + list(extra_experience or [])),
            project_bullets=tuple((pid, " ".join(bullets))
                                  for pid, bullets in project_bullets.items()),
            skills=tuple(skills))
        # Python classifies first. These verdicts are immutable: the model is
        # asked to explain them, never to decide them.
        verdicts = engine.deterministic_assessment(requirements, self.master, evidence,
                                                   jd.text)
        table = engine.requirement_table(requirements)
        # Kept for the final integrity gate in the pipeline.
        self.last_verdicts = verdicts
        # ONE compact table instead of the requirement list, the manual-review
        # list and the verdict block, which restated the same requirements
        # three times with evidence paragraphs Python already owns.
        verdicts_text = compact_verdict_table(verdicts, table, requirements)
        resume_experience = "\n".join(f"  - {item}" for item in experience_plain)
        grad = engine.graduation_requirement(jd.text)
        expected = engine.expected_graduation(self.master)
        facts = "\n".join(f"  - {f}" for f in self.master.standing_facts)
        # ---- the Experience selection this audit also scores --------------
        # The separate Groq experience audit is gone: this call judges the
        # shipped result as one component, and is given the deterministic
        # context it needs to do so. It cannot change any of it.
        context_block = experience_context or {}
        merged = context_block.get("merged_signals") or {}
        signal_summary = ", ".join(
            f"{name}={'yes' if value else 'no'}" for name, value in sorted(merged.items())
        ) or "not recorded"
        experience_rule = context_block.get("rule") or "not recorded"
        experience_swaps = ", ".join(
            f"{source} -> {target}" for source, target in
            (context_block.get("swaps") or [])) or "no swap"
        experience_ids = ", ".join(context_block.get("shipped_ids") or []) or "not recorded"

        # ---- audit #1 inputs the model must see in full -------------------
        # Project selection cannot be judged from the 3 that were picked, so
        # the COMPLETE candidate catalogue, the ranking and the rationale go
        # in. Evidence is abridged HERE and given in full below for the three
        # that shipped, which is where fidelity is actually judged: the
        # on-demand token budget does not allow saying everything twice.
        # The ALTERNATIVES only. The three that shipped get their own block
        # below with their final bullets, so listing them twice was pure
        # duplication - and the writer's full evidence corpus is not something
        # an assessor that cannot rewrite a bullet needs.
        catalogue = compact_unselected_projects(project_catalogue)
        detail = selection_detail or {}
        allocation = detail.get("allocation") or {}
        selection_block = (
            f"  selected (relevance order): {', '.join(detail.get('selected') or []) or '-'}\n"
            f"  display order on the page:  {', '.join(detail.get('display_order') or []) or '-'}\n"
            f"  bullet allocation:          "
            f"{', '.join(f'{k}={v}' for k, v in allocation.items()) or '-'}")
        bullet_evidence = compact_selected_projects(selected, project_bullets, detail)

        prompt = f"""Assess how well this candidate fits this job. Advisory only: it must never
change the resume.

JOB DESCRIPTION:
{jd.text}

ROLE SIGNALS - the posting only describes these; they are NOT requirements. Never score
them. If the candidate happens to be strong there, say so in `complementary_strengths`:
{signal_block}

THE FINAL TAILORED RESUME - Professional Experience:
{resume_experience}

THE FINAL TAILORED RESUME - Academic Projects: listed with their final bullets under
THE SELECTED PROJECTS below, so they are not repeated here.

THE FINAL TAILORED RESUME - Technical Skills:
  {', '.join(skills)}

EDUCATION AND STANDING FACTS:
{facts}
{f"  - the posting states a graduation requirement: {grad['text']}" if grad else "  - the posting states no graduation requirement"}

EXPLICITLY UNSUPPORTED - never credit the candidate with any of these:
  {', '.join(self.master.unsupported_heads())}

AUTHORITATIVE REQUIREMENTS AND THEIR IMMUTABLE VERDICTS - Python extracted every
requirement from the posting, owns its identity, and has already classified it against the
shipped resume. One line each: id | importance | kind | verdict | requirement. These are
FACTS for you. Cite requirements ONLY by these ids; you may never invent, rename, merge,
split or rephrase one, nor add one that is absent here. "manual" requirements are never
scored and are never gaps, because the recorded facts cannot settle them:
{verdicts_text}

{len(scored)} requirement(s) reach fit_score; {len(manual)} are manual review.

Summarize these verdicts; never change them. Do not move a requirement between buckets,
add one or remove one, and never write that a degree, qualification or skill is missing
when the table shows it present - a field or scope mismatch is not an absent degree. Never
infer an upgrade: Docker is NOT Kubernetes, C is NOT C++, AWS EC2/RDS is NOT general
cloud expertise, Django is NOT FastAPI, generic testing is NOT pytest.

FIT SCORE RUBRIC - this scores CANDIDATE-TO-JD FIT, not how polished the resume looks:
  9-10  nearly all important requirements evidenced
  8-8.9 core requirements strongly matched, only minor gaps
  7-7.9 meaningful match with several weaker or missing areas
  6-6.9 borderline but viable
  <6    substantial mismatch, or a hard requirement is not met
Do not penalize a technology the posting never asks for, and do not treat generic related
experience as an exact match.

A requirement_id may appear in exactly ONE bucket.

Evidence the posting never asked for belongs in `complementary_strengths`, not in a
requirement match, and must not raise fit_score.

tailoring_quality is a SEPARATE judgement about how well the resume presents the candidate
for this posting. A well-tailored resume can still have a real gap.

THE PROJECTS THAT WERE AVAILABLE BUT NOT SELECTED - judge the selection against these too,
not only against the three that shipped. Every one is factually supported; this is the
complete set of alternatives:
{catalogue}

WHAT GENERATION DECIDED ABOUT PROJECTS:
{selection_block}

THE SELECTED PROJECTS - rank, bullet allocation, an evidence digest and the FINAL BULLETS
that shipped. The bullets are the artifact you are scoring; the digest is there so you can
check a claim against its own project, not so you can rewrite anything:
{bullet_evidence}

AUDIT SCORES - these are YOUR judgements about how well generation used the evidence it
had. They are separate from fit_score and they change nothing:

  tailoring_quality.score - RESUME TAILORING. How effectively did this resume use the
  AVAILABLE SUPPORTED evidence for this posting? Do NOT lower it because the candidate
  genuinely lacks a technology; a missing unsupported qualification belongs to fit_score
  alone. Judge presentation, emphasis and evidence choice.

  project_selection.score - weigh approximately:
      40% relevance of the selected projects to the important JD requirements
      25% whether the BEST AVAILABLE projects were chosen from the full catalogue
      20% complementary coverage, avoiding three projects that prove the same thing
      15% ranking and the bullet allocation
  Report each as a component score. If a better available project was passed over, name it
  in notes by project_id.

  project_bullets.score - weigh approximately:
      35% direct JD relevance
      25% technical specificity
      20% evidence fidelity: every claim traceable to that project's own evidence above
      10% impact and ownership clarity
      10% non-redundancy across bullets
  Report each as a component score. A bullet claiming anything absent from its project's
  evidence is an evidence_fidelity failure and must be named in notes.

  callback_likelihood - ONE value: "LOW", "MEDIUM", "HIGH" or "UNKNOWN". A holistic read of
  whether this application plausibly earns a first conversation. Never a percentage.


HOW THE SHIPPED PROFESSIONAL EXPERIENCE WAS CHOSEN - judge the RESULT, not the process.
Python resolved this deterministically from an approved spreadsheet policy; the wording is
fixed approved wording and neither you nor any model may change or propose either.
  merged JD signals: {signal_summary}
  policy rule that fired: {experience_rule}
  swaps applied: {experience_swaps}
  shipped bullet ids: {experience_ids}

Score experience_selection: was the rule that fired the right reading of this posting, and
does the shipped evidence speak to it? You are READ ONLY here. Do not name a different
bullet id, do not propose a swap, and do not rewrite any wording: say only whether the
result fits the posting and why, in at most 15 words.

OUTPUT LENGTH - a HARD 900-token completion budget. The shape is fixed for you; keep the
prose short so it fits:
  * experience_selection.notes, project_selection.notes and project_bullets.notes: at
    most 2 entries each, at most 15 words per note
  * summary: at most 35 words
  * complementary_strengths: at most 3 entries, three words each
  * risk_flags: at most 3 entries, at most 12 words each
  * no prose reasoning anywhere, no restating the posting, no explaining your method.
    Numbers and short labels only.

Python decides eligibility.status deterministically and overwrites whatever you send.



Each bucket takes requirement_ids and nothing else; Python replaces the buckets with its
own entries either way. An id you invent, or classify twice, is rejected."""

        known_ids = {req.requirement_id for req in requirements}
        # Bounded schema repair. Only STRUCTURAL provider-schema failures retry
        # here; the deterministic finalization below is untouched, and there is
        # no Python-only assessment fallback.
        data: dict = {}
        problems: list[grounding.Problem] = []
        for attempt in range(1, ASSESSMENT_ATTEMPTS + 1):
            attempt_prompt = prompt
            if problems:
                attempt_prompt += (
                    "\n\nYour previous assessment response failed deterministic schema "
                    "validation.\n\nFix exactly these structural problems:\n"
                    + "\n".join(f"  - {p.message}" for p in problems)
                    + "\n\nReturn the complete assessment JSON again. Do not alter "
                      "immutable Python verdict semantics.")
            # The AUDIT transport: no Gemini fallback. Cover-letter generation
            # above deliberately keeps self.groq (and its fallback); all three
            # audits must be Groq or UNKNOWN.
            reply = self._invoke(self.audit, Request(
                "assessment", attempt_prompt, temperature=0.3,
                max_tokens=AUDIT_OUTPUT_TOKENS["assessment"],
                # Strict Structured Outputs: the provider certifies the shape.
                # reasoning_effort="none" turns reasoning OFF rather than
                # merely hiding it - hidden reasoning still consumed the 900
                # completion tokens and truncated the document before it
                # closed. Nothing is lost: Python owns the verdicts, the
                # Experience decision and the post-validation, and the schema
                # owns the shape, so there is no chain of thought to keep.
                # Neither reasoning_format nor include_reasoning is sent.
                response_schema=ASSESSMENT_RESPONSE_FORMAT,
                reasoning_effort="none",
                context={"jd_text": jd.text, "requirements": requirements,
                         "requirement_ids": sorted(known_ids),
                         "skills": skills, "selected": selected, "evidence": evidence,
                         "tailoring": tailoring or {}, "grad": grad,
                         "expected": expected,
                         "project_catalogue": project_catalogue or [],
                         "selection_detail": selection_detail or {},
                         "experience_context": context_block,
                         "project_bullets": project_bullets}))
            try:
                data = parse_json(reply.text, "assessment")
            except ProviderError as error:
                # A transport failure is not ours to retry. Under strict
                # Structured Outputs unparseable output is a PROVIDER contract
                # break, not something a reworded prompt can fix, so it is not
                # retried in a looser mode either - that would only hide a
                # schema bug behind best-effort JSON.
                if error.category != "malformed_response":
                    raise
                self.reject_last(f"malformed JSON: {error}")
                raise ProviderError(
                    "malformed_response",
                    f"the assessment reply was not valid JSON despite strict "
                    f"structured output: {error}") from None
            data, normalizations = grounding.canonicalize_assessment(data)
            for event in normalizations:
                self.log.info("[ASSESSMENT] normalized %s", event)
            problems = grounding.errors(
                grounding.validate_assessment(data, jd.text, known_ids))
            if not problems:
                break
            detail = ("schema validation failed: "
                      + "; ".join(p.message for p in problems[:4]))
            self.reject_last(detail)
            self.log.warning("[ASSESSMENT] attempt %d rejected: %s", attempt, detail)
        if problems:
            raise ProviderError(
                "malformed_response",
                f"assessment could not be validated after {ASSESSMENT_ATTEMPTS} "
                f"attempt(s): " + "; ".join(p.message for p in problems[:4]))

        # The model explained; Python decides. Verdicts, eligibility, the cap,
        # breadth and the user-facing summary are all applied here in order, so
        # assess() returns a fully finalized assessment.
        for note in grounding.finalize_assessment(
                data, verdicts=verdicts, table=table, requirements=requirements,
                scored=len(engine.scored_requirements(requirements)),
                signals=len(signal_reqs),
                manual=len(engine.manual_review_requirements(requirements))):
            self.log.info("[ASSESSMENT] %s", note)
        data.setdefault("eligibility_flags", verdicts.get("eligibility_flags") or [])
        # The audit dimensions are read defensively and kept in their own block,
        # so a missing score degrades to UNKNOWN without disturbing the verdict
        # architecture that validate_final_assessment checks.
        audit, audit_problems = grounding.validate_application_audit(data)
        for problem in audit_problems:
            self.log.warning("[APPLICATION AUDIT] %s", problem.message)
        fit = data.get("fit_score")
        audit["fit_score"] = round(float(fit), 1) if isinstance(fit, (int, float)) else None
        data["application_audit"] = audit
        return data


def build_client(master: MasterFacts, policy: Policy, log, *, mock: bool,
                 credentials: Credentials | None = None) -> LLMClient:
    if mock:
        transport = MockTransport(master, log)
        return LLMClient(transport, transport, master, policy, log, audit=transport,
                         research=transport)
    credentials = credentials or engine.load_credentials()
    gemini = GeminiTransport(credentials, log)
    # Two purpose-specific Groq models, each with its own pacing budget so the
    # parallel audits do not serialize each other. NEITHER has a Gemini
    # fallback: an audit is Groq or it is UNKNOWN.
    assessor = GroqTransport(credentials, log, model=assessment_model(), fallback=None)
    researcher = GroqTransport(credentials, log, model=research_model(), fallback=None)
    log.info("groq audit models: assessment=%s research=%s (keys configured: %d)",
             assessor.model or "UNSET", researcher.model or "UNSET", assessor.key_count)
    # Generation is Gemini-only now: projects, bullets and the cover letter all
    # go to Gemini, so no generation path can reach Groq.
    return LLMClient(gemini, gemini, master, policy, log, audit=assessor,
                     research=researcher)
