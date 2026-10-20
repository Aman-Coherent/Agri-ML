"""
Per-user LiteLLM Router factory. Every key is the requesting user's own (added
in Settings): there is no shared/system key pool at all, so a user who hasn't
added a key for a provider simply has no deployments for that provider's model
group. Jobs are refused up front when a required key is missing (see
user_keys.ensure_required_keys), so build_router() raising below is only
a last-resort guard.

Every key is fanned out across every free model in its provider's catalog
(see model_catalog.py) — Groq/Mistral rate-limit per model per org, so each
extra model is an entirely separate quota bucket on the same key, not a
slice of an existing one. All Groq and Mistral deployments share ONE group
(AGRI_MODEL), so usage-based routing spreads companies across both
providers' free quotas and skips whichever deployments are rate-limited.

The Groq browser_search calls in url_reader.py bypass this Router entirely
(raw API calls) and rotate across the user's Groq keys themselves - see
`pick_groq_search_keys()` below.

RPM/TPM counters used for routing decisions (`usage-based-routing-v2`) are
process-local in-memory only - see the comment above `Router(**router_kwargs)`
below for why Redis-backed cross-process sync is deliberately NOT wired in.
Separately, actual token/request usage *display* data (Settings -> API keys)
is tracked independently in usage_tracker.py, which does use Redis and is
unaffected by that decision.
"""
from __future__ import annotations

import logging
import re
import ssl
import weakref

import litellm
from litellm import Router
from litellm.integrations.custom_logger import CustomLogger

logger = logging.getLogger("llm_router")

from backend.core.model_catalog import (
    GROQ_OVERFLOW_MODELS,
    GROQ_PRIMARY_MODELS,
    MISTRAL_PRIMARY_MODELS,
    ModelSpec,
)

# Every free Groq AND Mistral (key, model) deployment in ONE group: the
# agriculture analysis (agri_analyzer.py) is the only per-company LLM call,
# and pooling both providers lets usage-based routing spread load across all
# of them (and skip whichever are rate-limited) instead of saturating one
# provider while the other's free quota sits idle.
AGRI_MODEL = "agri"
QUALITY_MODEL = "quality"  # user's Claude/OpenAI/custom key, if configured - fallback only

# Mistral bills cached-prefix tokens at 10% instead of 100% when every call
# shares the same `prompt_cache_key`. agri_analyzer's system prompt (rules +
# ~2.7K-token taxonomy + schema) is byte-identical across every company, so
# this is a near-total-cost win on the biggest static chunk of that call.
# Baked into every Mistral deployment's litellm_params (see _deployment()'s
# docstring for why NOT a generic per-call kwarg). Version-suffixed so a
# future prompt edit starts a fresh cache instead of silently caching stale
# text.
MISTRAL_PROMPT_CACHE_KEY = "agri-analysis-v2"

# litellm has no request timeout by default (falls back to the underlying
# SDK's default, ~600s). A single hung network call at that length can pin a
# concurrency slot for ten minutes, and it silently degrades throughput to a
# crawl as more slots leak away one at a time over a large job. 30s is well
# above what a normal completion takes; a genuinely stuck provider fails
# fast and frees the slot for the next company instead of hanging the batch.
REQUEST_TIMEOUT = 30.0

# litellm creates ~4 separate httpx client objects PER deployment (sync,
# async, sync-stream, async-stream), and httpx builds a brand new SSLContext
# for each one whenever `verify` is a bool - which means re-parsing the
# entire certifi CA bundle from disk (`ssl.create_default_context()` +
# `load_verify_locations()`, NOT just opening the file) every single time.
# On this stack that parse costs ~0.6s per client. With the multi-model
# fanout in this file (each key now fanned out across every model in its
# provider's catalog), a single build_router() call creates 50+ deployments
# - 200+ httpx clients - which measured as a 130+ second hang on every job
# start (confirmed via faulthandler.dump_traceback_later, stuck in
# httpx._config.load_ssl_context_verify). httpx explicitly special-cases
# `verify=<ssl.SSLContext instance>` to reuse it as-is with zero re-parsing
# (see httpx._config.SSLConfig.load_ssl_context_verify), so building this
# context ONCE and pointing litellm's global `ssl_verify` at it collapses
# build_router() back down to ~0.06s regardless of deployment count, with
# identical certificate verification behavior (same certifi CA bundle,
# loaded once instead of N times).
if not isinstance(litellm.ssl_verify, ssl.SSLContext):
    litellm.ssl_verify = ssl.create_default_context()


def _deployment(
    model_name_group: str,
    key_ref: str,
    api_key: str,
    spec: ModelSpec,
    extra_litellm_params: dict | None = None,
) -> dict:
    """One (key, model) deployment. `model_info.id` is set to `key_ref__tag`
    so it's BOTH a globally-unique id litellm's router needs to track
    per-deployment cooldowns/rpm/tpm correctly, AND a way to recover which
    physical key + model handled a given response afterward (see
    usage_tracker.record_usage, which reads `response._hidden_params
    ["model_id"]` and splits on "__").

    IMPORTANT: `key_ref` (and this whole id) must never contain a colon.
    litellm's usage-based-routing-v2 (lowest_tpm_rpm_v2.py) builds its own
    Redis/in-memory cache keys as f"{deployment_id}:tpm:{minute}" and then
    recovers the deployment id by naively doing `cache_key.split(":")[0]` -
    it assumes the id itself has zero colons. An earlier version of this
    file used "system:groq:0" / "user:<uuid>" style refs with a "::tag"
    suffix, which silently truncated every single deployment id down to
    just "system" or "user", made every id comparison fail, and caused
    100% of calls to raise "No deployments available" regardless of actual
    usage - reproduced and confirmed via a router-only completion call, no
    real rate limiting involved. Hyphens for the ref itself and a double
    underscore for the tag suffix keep ids human-readable while staying
    entirely colon-free.

    `extra_litellm_params` (e.g. Mistral's `prompt_cache_key`) is baked into
    THIS deployment's own litellm_params rather than passed as a generic
    per-call kwarg from the caller. That distinction matters: litellm builds
    each provider request as `{**deployment_litellm_params, **call_kwargs}`,
    so a provider-specific call kwarg silently survives a `fallbacks` hop to
    a completely different provider's deployment. Reproduced live: passing
    `prompt_cache_key` as a call-level kwarg meant that the moment Mistral
    failed (e.g. a 402) and litellm fell back to Groq, Groq rejected the
    request outright with `property 'prompt_cache_key' is unsupported` -
    turning one provider's outage into a total product-generation outage
    instead of a clean fallback. Scoping it to the deployment's own
    litellm_params means it only ever appears on requests actually sent to
    that deployment."""
    return {
        "model_name": model_name_group,
        "litellm_params": {
            "model": spec.litellm_model,
            "api_key": api_key,
            "rpm": spec.rpm,
            "tpm": spec.tpm,
            **(spec.extra or {}),
            **(extra_litellm_params or {}),
        },
        "model_info": {"id": f"{key_ref}__{spec.tag}"},
    }


def _add_fanout(
    model_list: list[dict],
    model_name_group: str,
    key_ref: str,
    api_key: str,
    specs: list[ModelSpec],
    extra_litellm_params: dict | None = None,
) -> None:
    for spec in specs:
        model_list.append(_deployment(model_name_group, key_ref, api_key, spec, extra_litellm_params))


def build_router(user_keys: list[dict] | None = None) -> Router:
    """
    user_keys: list of {"id": str, "provider": "claude"|"openai"|"custom"|"groq"|"mistral",
                          "api_key": str, "model_name": str|None, "base_url": str|None}
    """
    user_keys = [uk for uk in (user_keys or []) if uk.get("api_key")]

    model_list: list[dict] = []

    def _user_key_ref(uk: dict) -> str:
        # Falls back to an object-identity-based ref for the (theoretical)
        # case of a key dict with no id, so a deployment id is still unique
        # rather than silently colliding with another anonymous key. No
        # colons anywhere here - see _deployment()'s docstring for why.
        return f"user-{uk['id']}" if uk.get("id") else f"user-anon-{id(uk)}"

    for uk in user_keys:
        provider = uk.get("provider")
        api_key = uk["api_key"]
        key_ref = _user_key_ref(uk)

        if provider == "groq":
            _add_fanout(model_list, AGRI_MODEL, key_ref, api_key, [*GROQ_PRIMARY_MODELS, *GROQ_OVERFLOW_MODELS])
        elif provider == "mistral":
            _add_fanout(
                model_list, AGRI_MODEL, key_ref, api_key, MISTRAL_PRIMARY_MODELS,
                extra_litellm_params={"prompt_cache_key": MISTRAL_PROMPT_CACHE_KEY},
            )
        elif provider == "claude":
            model_list.append(
                {
                    "model_name": QUALITY_MODEL,
                    "litellm_params": {"model": uk.get("model_name") or "claude-3-5-sonnet-latest", "api_key": api_key},
                    "model_info": {"id": f"{key_ref}__claude"},
                }
            )
        elif provider == "openai":
            model_list.append(
                {
                    "model_name": QUALITY_MODEL,
                    "litellm_params": {"model": uk.get("model_name") or "gpt-4o-mini", "api_key": api_key},
                    "model_info": {"id": f"{key_ref}__openai"},
                }
            )
        elif provider == "custom":
            params = {"model": uk.get("model_name") or "openai/custom-model", "api_key": api_key}
            if uk.get("base_url"):
                params["api_base"] = uk["base_url"]
            model_list.append(
                {"model_name": QUALITY_MODEL, "litellm_params": params, "model_info": {"id": f"{key_ref}__custom"}}
            )

    if not model_list:
        raise RuntimeError("No LLM API keys configured. Add your own Groq/Mistral keys in Settings -> API Keys.")

    # QUALITY_MODEL only has deployments if a matching key is actually
    # configured. Unconditionally referencing an empty group in `fallbacks`
    # means the moment the primary group hits a rate limit, LiteLLM's
    # fallback chain tries to route to a model_name string with no
    # deployments and hard-crashes with `litellm.BadRequestError` instead of
    # a clean rate-limit error - actively making the fallback mechanism
    # worse than having none. Only wire in fallbacks to a group that exists.
    configured_groups = {m["model_name"] for m in model_list}
    fallbacks = []
    if AGRI_MODEL in configured_groups and QUALITY_MODEL in configured_groups:
        fallbacks.append({AGRI_MODEL: [QUALITY_MODEL]})
    if QUALITY_MODEL in configured_groups and AGRI_MODEL in configured_groups:
        fallbacks.append({QUALITY_MODEL: [AGRI_MODEL]})

    router_kwargs: dict = dict(
        model_list=model_list,
        routing_strategy="usage-based-routing-v2",
        fallbacks=fallbacks,
        timeout=REQUEST_TIMEOUT,
        num_retries=3,
        retry_after=3,
        # Default cooldown when a 429 carries no Retry-After header.
        # `allowed_fails` is deliberately NOT set: setting it switches litellm
        # to its legacy "cool down only after N fails per minute" logic,
        # which kept re-sending companies to Groq keys whose DAILY token
        # limit was already spent (~4,000 TPD 429s in one 21-minute run,
        # each a wasted attempt and ~1/3 of companies failing). Without it,
        # litellm's current logic cools a deployment down on its first 429
        # for as long as the provider's Retry-After says, and
        # _DailyLimitCooldown below extends that for daily-limit errors.
        cooldown_time=15,
    )

    # Deliberately NOT wiring redis_host/redis_port here for cross-process
    # RPM/TPM sync. Empirically (see scripts/eval_pipeline.py's first run),
    # litellm's Router-level Redis usage cache hangs/fails against this
    # project's Redis (an old Windows-ported build - see usage_tracker.py's
    # HSET-compatibility note for the same root cause hitting a different
    # command), and `usage-based-routing-v2`'s TPM/RPM lookup treats a
    # failed cache read as "0 healthy deployments" rather than degrading to
    # "assume unused" - i.e. this was silently making EVERY deployment
    # look rate-limited on EVERY call, not just under real load. It's also
    # not actually needed: only one ARQ worker process is ever run (see the
    # "Zombie ARQ Worker Process" fix in job history), so a single Router
    # instance's local in-memory usage counters already see every request
    # end to end - there is no second process whose usage would otherwise
    # go untracked. If real multi-process horizontal scaling is added later,
    # this needs a genuinely compatible Redis (5+) before re-enabling.
    router = Router(**router_kwargs)
    _install_daily_limit_cooldown()
    _ROUTERS.add(router)
    return router


# ─────────── Daily-limit cooldown ───────────
#
# Groq's free tier also has a tokens-PER-DAY cap per (key, model). Once it's
# hit, every call to that deployment fails until the rolling window frees up
# - often an hour or more - and the error says how long ("Please try again
# in 1h2m3.5s"). This rests that deployment for that long on every live
# router (one per running job), so the work flows to deployments that still
# have quota instead of being retried into the same wall.

_ROUTERS: "weakref.WeakSet[Router]" = weakref.WeakSet()
_RETRY_IN_RE = re.compile(r"try again in\s+(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:([\d.]+)s)?", re.I)
_DAILY_LIMIT_RE = re.compile(r"per day|\bTPD\b|\bRPD\b", re.I)
_DEFAULT_DAILY_COOLDOWN_S = 30 * 60


def _parse_retry_in(message: str) -> float | None:
    m = _RETRY_IN_RE.search(message)
    if not m or not any(m.groups()):
        return None
    hours, minutes, seconds = (float(g) if g else 0.0 for g in m.groups())
    return hours * 3600 + minutes * 60 + seconds


class _DailyLimitCooldown(CustomLogger):
    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        try:
            message = str(kwargs.get("exception") or "")
            if not _DAILY_LIMIT_RE.search(message):
                return
            model_info = (kwargs.get("litellm_params") or {}).get("model_info") or {}
            deployment_id = model_info.get("id")
            if not deployment_id:
                return
            seconds = max(_parse_retry_in(message) or _DEFAULT_DAILY_COOLDOWN_S, 60.0)
            for router in list(_ROUTERS):
                if deployment_id in router.get_model_ids():
                    router.cooldown_cache.add_deployment_to_cooldown(
                        model_id=deployment_id,
                        original_exception=kwargs.get("exception"),
                        exception_status=429,
                        cooldown_time=seconds,
                    )
            logger.info("Daily limit hit on %s - resting it for %.0f min", deployment_id, seconds / 60)
        except Exception:  # noqa: BLE001 - a logging hook must never break a request
            logger.debug("daily-limit cooldown hook failed", exc_info=True)


def _install_daily_limit_cooldown() -> None:
    if not any(isinstance(cb, _DailyLimitCooldown) for cb in litellm.callbacks):
        litellm.callbacks.append(_DailyLimitCooldown())


def pick_groq_search_keys(user_keys: list[dict] | None = None) -> list[tuple[str, str]]:
    """Every one of the user's active Groq keys as (api_key, key_ref), for
    url_reader.GroqSearchPool - the browser_search calls (no-website web
    search + URL-read fallback) bypass the Router, so they rotate across the
    keys themselves. key_ref matches `_deployment()`'s id scheme so
    usage_tracker.py attributes these raw calls to the right physical key.
    Empty when the user has no Groq key: searching is then simply skipped."""
    return [
        (uk["api_key"], f"user-{uk['id']}" if uk.get("id") else f"user-anon-{id(uk)}")
        for uk in (user_keys or [])
        if uk.get("provider") == "groq" and uk.get("api_key")
    ]


def pick_jina_key(user_keys: list[dict] | None = None) -> tuple[str | None, str | None]:
    """Like every provider, Jina Reader has no system-level key (see config.py) - it
    is either called with the requesting USER's own key (first active one
    they've added in Settings, provider="jina") or, if they haven't added
    one, with no key at all. url_reader.py's unauthenticated fallback within
    _fetch_jina already handles the "no key" case correctly on its own, so
    returning (None, None) here is a normal, fully-supported outcome, not an
    error - it must NEVER be filled in with any shared/system credential.

    Returns (api_key, key_ref) so usage_tracker.py can attribute Reader
    calls back to the right physical key, exactly like pick_groq_search_keys.
    """
    user_jina_keys = [uk for uk in (user_keys or []) if uk.get("provider") == "jina" and uk.get("api_key")]
    if not user_jina_keys:
        return None, None
    uk = user_jina_keys[0]
    key_ref = f"user-{uk['id']}" if uk.get("id") else f"user-anon-{id(uk)}"
    return uk["api_key"], key_ref
