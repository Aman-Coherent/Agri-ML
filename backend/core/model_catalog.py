"""
Single source of truth for which free-tier models exist per provider, and
their per-model rate limits.

Both Groq and Mistral rate-limit **per model, per organization** — not one
combined number for the whole account (verified directly against
`console.groq.com/docs/rate-limits` and `docs.mistral.ai/admin/billing-usage/
usage-limits`). That means every additional model added here is a genuinely
separate quota bucket on the exact same API key, not a slice of an existing
one. `llm_router.py` fans every configured key out across every model in its
tier; `usage_tracker.py` uses the same catalog to compute "remaining today"
without depending on provider response headers (which Groq's chat-completion
path does not currently surface through litellm's Router — see
`usage_tracker.py` docstring).

Tiering: PRIMARY models are comparable-or-better than the previous single
default (`openai/gpt-oss-20b`) — safe to load-balance across interchangeably.
OVERFLOW models trade quality for a much bigger daily budget and are only
reached once every primary deployment is already rate-limited (wired via
`fallbacks` in `llm_router.py`), so a normal day looks identical in quality
to before this change, and a saturated day degrades gracefully instead of
failing outright. (Groq currently has no suitable overflow model - see
GROQ_OVERFLOW_MODELS.)
"""
from __future__ import annotations

from typing import NamedTuple


class ModelSpec(NamedTuple):
    litellm_model: str  # value passed as litellm_params.model
    tag: str  # short id used in key_ref__tag identifiers and the usage UI
    rpm: int
    tpm: int
    rpd: int | None  # None = provider doesn't publish/enforce a daily request cap
    tpd: int | None  # None = provider doesn't publish/enforce a daily token cap
    # Model-specific litellm_params baked into every deployment of this model
    # (never passed per-call - see llm_router._deployment's docstring).
    extra: dict | None = None


# gpt-oss is a reasoning model: hidden reasoning tokens count toward TPM/TPD
# like any other output. "low" keeps that short - classification against a
# fixed list doesn't need long deliberation, and every token saved is quota
# for another company.
_GPT_OSS_EXTRA = {"reasoning_effort": "low"}


# Confirmed 2026-10-20 against Groq's live GET /openai/v1/models (with a real
# free-tier key), https://console.groq.com/docs/rate-limits and
# https://console.groq.com/docs/deprecations. Removed after Groq shut them
# down for free/developer tiers (every call now 404s `model_not_found`):
#   - llama-3.3-70b-versatile  (shut down 2026-08-16; Groq's replacement: gpt-oss-120b)
#   - qwen/qwen3.6-27b         (shut down 2026-09-14; replacement: qwen/qwen3.8-27b)
# Excludes: the llama-prompt-guard-2-* moderation classifiers,
# gpt-oss-safeguard-20b (safety-tuned, 3 RPM), allam-2-7b (Arabic-focused,
# 4K context), and the whisper/orpheus audio models — none are good fits for
# plain classify/generate completions.
# rpm here is what the ROUTER allows, not Groq's published 30 RPM: one
# analysis call is ~5-6K tokens against an 8K TPM bucket, so a 2nd call in
# the same minute is always a 429. litellm counts rpm before sending (its
# token counts only update after a response), so rpm=1 is what actually
# stops 50 concurrent companies from bursting into the same Groq bucket.
GROQ_PRIMARY_MODELS: list[ModelSpec] = [
    ModelSpec("groq/openai/gpt-oss-20b", "gpt-oss-20b", rpm=1, tpm=8_000, rpd=1_000, tpd=200_000, extra=_GPT_OSS_EXTRA),
    ModelSpec("groq/openai/gpt-oss-120b", "gpt-oss-120b", rpm=1, tpm=8_000, rpd=1_000, tpd=200_000, extra=_GPT_OSS_EXTRA),
    # Groq lists this as a "preview" model (may be discontinued at short
    # notice) - re-check the deprecations page if it starts 404ing.
    ModelSpec("groq/qwen/qwen3.8-27b", "qwen3.8-27b", rpm=1, tpm=8_000, rpd=1_000, tpd=200_000),
]

# Previously llama-3.1-8b-instant (bigger daily budget, weaker model), but
# Groq shut it down for free/developer tiers on 2026-08-16 and its official
# replacement, gpt-oss-20b, is already in PRIMARY above. Groq currently offers
# no other suitable free chat model, so this tier is empty; llm_router.py
# only wires a fallback to a group that actually has deployments, so an empty
# tier is safe.
GROQ_OVERFLOW_MODELS: list[ModelSpec] = []

# Mistral rate-limits per model too, but publishes no free-tier table, so
# rpm/tpm below are the REAL limits Mistral's API reports back in its
# `x-ratelimit-limit-req-minute` / `x-ratelimit-limit-tokens-minute` response
# headers, read live on 2026-10-20 (identical on every free key tested).
# Getting these wrong reproduces the GROQ_TPM/MISTRAL_TPM misconfiguration
# bug fixed earlier (rate-limited router never rotating away from an
# exhausted deployment). rpd/tpd are None since Mistral enforces a monthly
# (not daily) allowance instead, and no longer publishes its size.
#
# Removed: `mistral-small-latest` (now Mistral Small 4 / mistral-small-2603).
# Every free key reports `x-ratelimit-limit-req-minute: 0` for it and 429s
# every call - it is no longer usable on Mistral's free tier. It was still
# being load-balanced here, so roughly 1 in 3 generation calls burned a
# guaranteed 429 + cooldown/retry before reaching a working model.
# Added in its place: `ministral-14b-latest` (Ministral 3 14B, same family as
# the 3B/8B below, verified returning valid json_object output).
# All three `-latest` aliases currently resolve to the 2512 (Ministral 3)
# releases; the older 2410 versions behind them were retired 2025-12-31.
#
# IMPORTANT: "ministral-3-8b-latest" (previously here) is NOT a real Mistral
# model id - it was a typo conflating the two actually-separate "Ministral 3"
# edge models, "ministral-3b-latest" (3B) and "ministral-8b-latest" (8B).
# Every single Mistral call hit Mistral's hard `400 Invalid model` error as a
# result - 100% of the time, not intermittently like a rate limit.
# Confirmed against Mistral's own docs (docs.mistral.ai) before fixing.
MISTRAL_PRIMARY_MODELS: list[ModelSpec] = [
    ModelSpec("mistral/ministral-14b-latest", "ministral-14b", rpm=30, tpm=937_500, rpd=None, tpd=None),
    ModelSpec("mistral/ministral-8b-latest", "ministral-8b", rpm=188, tpm=625_000, rpd=None, tpd=None),
]
# Deliberately NOT used: "mistral/ministral-3b-latest" (rpm=750,
# tpm=1_300_000 on free keys). In live tests against the agriculture
# taxonomy it was the one model that returned placeholder product names
# ("Herbicide XYZ", "Insecticide ABC") and the least consistent roles. Its
# extra capacity isn't needed either: 8B + 14B already allow ~1,700 requests
# per minute across 8 free keys, far more than website reading can feed.

MISTRAL_OVERFLOW_MODELS: list[ModelSpec] = []

# Jina Reader isn't an LLM and is never routed through litellm's Router (see
# url_reader.py — it's a raw httpx call), so `litellm_model` here is a plain
# label, never actually passed to litellm. This single entry only exists so
# usage_tracker.py can reuse the exact same per-key/per-"model" daily-usage
# machinery it already has for Groq/Mistral instead of a second bespoke code
# path. rpm=500 is Jina's own published authenticated-key limit
# (jina.ai/reader); rpd/tpd are None because Jina publishes no daily cap for
# Reader, only the per-minute one — same "Not published" display as Mistral.
JINA_MODELS: list[ModelSpec] = [
    ModelSpec("jina/reader", "reader", rpm=500, tpm=0, rpd=None, tpd=None),
]


def specs_for_provider(provider: str) -> list[ModelSpec]:
    """All catalog specs (primary + overflow) that a key for this provider
    fans out across — used by usage_tracker.py to know which deployment ids
    to look up for a given physical key."""
    if provider == "groq":
        return [*GROQ_PRIMARY_MODELS, *GROQ_OVERFLOW_MODELS]
    if provider == "mistral":
        return [*MISTRAL_PRIMARY_MODELS, *MISTRAL_OVERFLOW_MODELS]
    if provider == "jina":
        return JINA_MODELS
    return []


def spec_for_tag(provider: str, tag: str) -> ModelSpec | None:
    for spec in specs_for_provider(provider):
        if spec.tag == tag:
            return spec
    return None


def provider_for_tag(tag: str) -> str | None:
    """Reverse lookup used when only a deployment's `key_ref__tag` id is
    known (e.g. in usage_tracker.record_usage, which has no other way to
    tell which provider a call went to)."""
    if any(spec.tag == tag for spec in specs_for_provider("groq")):
        return "groq"
    if any(spec.tag == tag for spec in specs_for_provider("mistral")):
        return "mistral"
    if any(spec.tag == tag for spec in specs_for_provider("jina")):
        return "jina"
    return None
