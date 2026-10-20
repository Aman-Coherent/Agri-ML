"""
Turns a company's website (or, with no website, its name + location) into
LLM-ready text.

With a URL (if every step fails, or the domain is dead, it is treated like
"without a URL" below):
  1. Redis cache        — same URL never re-fetched within 7 days
  2. Direct fetch       — our own httpx GET of the homepage + up to 2 linked
                          product/service pages, text extracted locally with
                          trafilatura. Free, no rate limit, hundreds in
                          parallel - the default path at 6-lakh-company scale.
  3. Jina Reader        — r.jina.ai headless-Chrome render, only for sites the
                          direct fetch can't read (JavaScript-only pages,
                          bot walls). Unauthenticated tier is 20 RPM, so it's
                          rate-limited locally and skipped when saturated.
  4. Groq browser_search on the URL (gpt-oss built-in browsing tool)
  5. name_location      — no text; the LLM works from name + location only

Without a URL:
  1. Redis cache (keyed by name + location)
  2. Groq browser_search web search for "<name> <location>"
  3. name_location

Groq browser_search costs ~13K tokens per call against an 8K-TPM / 200K-TPD
free bucket, so GroqSearchPool rotates across every (key, model) slot and
rests each slot after use; when every slot is resting it returns None
immediately and the company degrades to name_location instead of waiting.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from urllib.parse import urljoin, urlparse

import httpx
import trafilatura
from lxml import html as lxml_html
from redis.asyncio import Redis

from backend.core.circuit_breaker import ContentUnavailable, jina_breaker
from backend.core.models import UrlReadResult, UrlReadSource
from backend.core.usage_tracker import record_usage

logger = logging.getLogger("url_reader")

# The highest-signal content (hero text, about, product listings) is almost
# always front-loaded, and this block goes into every analysis prompt - on
# Groq's 8K-TPM free buckets, every extra char here is throughput lost.
MAX_MARKDOWN_CHARS = 5000
CACHE_TTL_SECONDS = 7 * 24 * 3600
MIN_CONTENT_LENGTH = 200
CIRCUIT_STATUS_KEY = "circuit_status:jina_reader"

# ─────────────────────────── Direct fetch ───────────────────────────

DIRECT_TIMEOUT = httpx.Timeout(15.0, connect=8.0)
MAX_PAGE_BYTES = 3_000_000
MAX_SUBPAGES = 2
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
)
_HEADERS = {"User-Agent": _UA, "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.5", "Accept-Language": "en"}
_SUBPAGE_PATH_RE = re.compile(
    r"product|service|solution|offering|portfolio|range|catalog|brand|crop|seed|fertili|machine|equipment|what-we-do",
    re.I,
)
_SUBPAGE_TEXT_RE = re.compile(r"product|service|solution|offering|portfolio|our range|catalog|brands|what we do", re.I)
_SKIP_PATH_RE = re.compile(
    r"blog|news|post|article|career|job|press|media|event|investor|contact|privacy|terms|cookie|login|"
    r"signin|register|cart|checkout|account|\.(pdf|jpe?g|png|gif|zip|docx?|xlsx?)$",
    re.I,
)

# One pooled client per worker process (created lazily inside the running
# event loop). A second client with certificate verification off is used only
# as a retry for sites whose TLS setup is broken (expired/self-signed certs
# are common on small business sites): we only ever READ public marketing
# pages here and feed the text to an LLM, so a forged page could at worst
# mislabel one company - an acceptable trade for not losing those sites.
_clients: dict[bool, httpx.AsyncClient] = {}


def _client(verify: bool = True) -> httpx.AsyncClient:
    client = _clients.get(verify)
    if client is None or client.is_closed:
        client = httpx.AsyncClient(
            timeout=DIRECT_TIMEOUT,
            follow_redirects=True,
            headers=_HEADERS,
            verify=verify,
            limits=httpx.Limits(max_connections=300, max_keepalive_connections=50),
        )
        _clients[verify] = client
    return client


async def _get_html(url: str) -> tuple[str, str]:
    """Returns (final_url, html). Raises ContentUnavailable for anything that
    isn't a readable HTML page."""
    for verify in (True, False):
        try:
            async with _client(verify).stream("GET", url) as resp:
                if resp.status_code >= 400:
                    raise ContentUnavailable(f"http_{resp.status_code}")
                ctype = resp.headers.get("content-type", "")
                if ctype and "html" not in ctype and "xml" not in ctype:
                    raise ContentUnavailable("not_html")
                chunks, size = [], 0
                async for chunk in resp.aiter_bytes():
                    chunks.append(chunk)
                    size += len(chunk)
                    if size > MAX_PAGE_BYTES:
                        break
                raw = b"".join(chunks)
                return str(resp.url), raw.decode(resp.encoding or "utf-8", errors="replace")
        except httpx.ConnectError as exc:
            # Certificate problems surface as ConnectError - retry once unverified.
            if verify and "certificate" in str(exc).lower():
                continue
            raise ContentUnavailable("connect_error") from exc
        except (httpx.TimeoutException, httpx.TooManyRedirects, httpx.RemoteProtocolError) as exc:
            raise ContentUnavailable(type(exc).__name__) from exc
    raise ContentUnavailable("tls_error")


def _pick_subpages(base_url: str, raw_html: str) -> list[str]:
    try:
        tree = lxml_html.fromstring(raw_html)
    except Exception:
        return []
    host = urlparse(base_url).netloc.lower().removeprefix("www.")
    base_norm = base_url.rstrip("/")
    by_path, by_text = [], []
    seen: set[str] = set()
    for a in tree.iter("a"):
        href = (a.get("href") or "").strip()
        if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
            continue
        url = urljoin(base_url, href).split("#")[0]
        parsed = urlparse(url)
        if parsed.netloc.lower().removeprefix("www.") != host or parsed.scheme not in ("http", "https"):
            continue
        if url.rstrip("/") == base_norm or url in seen or _SKIP_PATH_RE.search(parsed.path):
            continue
        seen.add(url)
        if _SUBPAGE_PATH_RE.search(parsed.path):
            by_path.append(url)
        elif _SUBPAGE_TEXT_RE.search(a.text_content() or ""):
            by_text.append(url)
    # Shortest paths first: "/products" beats "/products/x/y/some-item".
    by_path.sort(key=lambda u: urlparse(u).path.count("/"))
    return (by_path + by_text)[:MAX_SUBPAGES]


def _extract_text(raw_html: str) -> str:
    return trafilatura.extract(
        raw_html, output_format="markdown", include_links=False, include_tables=True, favor_recall=True
    ) or ""


def _merge_texts(texts: list[str]) -> str:
    """Joins page texts, dropping lines already seen (repeated nav/hero
    banners across pages) so the char budget goes to real content."""
    seen: set[str] = set()
    out: list[str] = []
    for text in texts:
        for line in text.splitlines():
            key = line.strip().lower()
            if not key:
                continue
            if key in seen:
                continue
            seen.add(key)
            out.append(line.rstrip())
    return "\n".join(out)[:MAX_MARKDOWN_CHARS]


async def _fetch_direct(url: str) -> UrlReadResult:
    final_url, raw = await _get_html(url)
    home_text, subpages = await asyncio.gather(
        asyncio.to_thread(_extract_text, raw), asyncio.to_thread(_pick_subpages, final_url, raw)
    )

    async def _sub(u: str) -> str:
        try:
            _, sub_raw = await _get_html(u)
            return await asyncio.to_thread(_extract_text, sub_raw)
        except Exception:
            return ""

    sub_texts = await asyncio.gather(*(_sub(u) for u in subpages)) if subpages else []
    markdown = _merge_texts([home_text, *sub_texts])
    if len(markdown) < MIN_CONTENT_LENGTH:
        raise ContentUnavailable("direct_empty")
    return UrlReadResult(
        markdown=markdown, source=UrlReadSource.DIRECT_FETCH, success=True, token_estimate=len(markdown) // 4
    )


# ─────────────────────────── Jina Reader ───────────────────────────

# The free/unauthenticated r.jina.ai tier renders noticeably slower than the
# paid tier (observed 2-16s+ per site) since it has no priority queueing.
JINA_TIMEOUT = 25.0
# Jina's published unauthenticated limit; with a user key it's 500 RPM.
JINA_RPM_NO_KEY = 20
JINA_RPM_WITH_KEY = 450


class _RateLimiter:
    """Non-blocking sliding-window limiter: `try_acquire()` says whether a
    call is allowed right now. Callers skip Jina rather than queue behind a
    20-RPM limit while hundreds of companies are in flight."""

    def __init__(self) -> None:
        self._calls: list[float] = []

    def try_acquire(self, per_minute: int) -> bool:
        now = time.monotonic()
        self._calls = [t for t in self._calls if now - t < 60.0]
        if len(self._calls) >= per_minute:
            return False
        self._calls.append(now)
        return True


_jina_limiter = _RateLimiter()


def _cache_key(url: str) -> str:
    return f"url_cache:{hashlib.sha256(url.encode()).hexdigest()}"


def _search_cache_key(company_name: str, location: str | None) -> str:
    raw = f"{company_name.strip().lower()}|{(location or '').strip().lower()}"
    return f"search_cache:{hashlib.sha256(raw.encode()).hexdigest()}"


async def _sync_circuit_status(redis: Redis | None) -> None:
    """Mirrors the in-process circuit breaker state into Redis so the FastAPI
    process (which never calls Jina itself) can show a live indicator of
    what the ARQ worker process is doing."""
    if redis is None:
        return
    try:
        await redis.set(CIRCUIT_STATUS_KEY, json.dumps(jina_breaker.status()), ex=120)
    except Exception:
        logger.warning("Failed to sync circuit breaker status to Redis", exc_info=True)


async def _fetch_jina(
    url: str,
    jina_api_key: str | None = None,
    redis: Redis | None = None,
    key_ref: str | None = None,
) -> UrlReadResult:
    """r.jina.ai works completely unauthenticated - no key, no charge - just
    at lower throughput/priority than the paid tier. There is no shared
    system-level Jina key: `jina_api_key` (when given) is always a specific
    USER's own key from Settings (see llm_router.pick_jina_key). A 402 means
    THAT USER's key balance is exhausted, not that Jina Reader itself is
    unusable - retrying without the Authorization header falls through to
    the free tier instead of abandoning Jina entirely."""
    jina_url = f"https://r.jina.ai/{url}"
    base_headers = {"Accept": "application/json", "X-Return-Format": "markdown"}

    async with httpx.AsyncClient(timeout=JINA_TIMEOUT) as client:
        resp = None
        if jina_api_key:
            headers = {**base_headers, "Authorization": f"Bearer {jina_api_key}"}
            resp = await client.get(jina_url, headers=headers)
            if resp.status_code == 402:
                logger.warning("User's Jina API key has no balance (402) - falling back to free/unauthenticated tier")
                resp = None

        if resp is None:
            resp = await client.get(jina_url, headers=base_headers)

        if redis is not None and key_ref is not None:
            # Jina isn't token-billed, so prompt/completion tokens are always
            # 0 here - record_usage still increments the "requests" counter.
            try:
                await record_usage(redis, f"{key_ref}__reader", "jina", 0, 0, headers=dict(resp.headers))
            except Exception:
                logger.debug("jina usage capture failed", exc_info=True)

        # 400/404/422 mean "Jina looked at THIS url and couldn't render it" -
        # a fact about the company's website, not about Jina's health, so it
        # must not count as a circuit-breaker failure (see ContentUnavailable).
        if resp.status_code in (400, 404, 422):
            raise ContentUnavailable(f"jina_rejected_{resp.status_code}")
        resp.raise_for_status()
        data = resp.json()
        content = data.get("data", {}).get("content", "") or data.get("content", "")

        if len(content) < MIN_CONTENT_LENGTH:
            raise ContentUnavailable("empty_content")

        markdown = content[:MAX_MARKDOWN_CHARS]
        return UrlReadResult(
            markdown=markdown, source=UrlReadSource.JINA_READER, success=True, token_estimate=len(markdown) // 4
        )


# ─────────────────────────── Groq browser_search ───────────────────────────

BROWSER_SEARCH_TIMEOUT = 45.0
# browser_search answers carry inline citation markers like 【5†L8-L12】 that
# are pure noise once the text is pasted into another LLM prompt.
_CITATION_RE = re.compile(r"【[^】]*】")
# Groq's browser_search built-in tool only runs on gpt-oss models.
_SEARCH_MODELS = (("openai/gpt-oss-20b", "gpt-oss-20b"), ("openai/gpt-oss-120b", "gpt-oss-120b"))
# One search (~13K tokens) uses more than a free slot's whole 8K TPM, so
# a slot is rested for a minute after every use; a 429 rests it for as long
# as Groq says (TPD exhaustion: much longer).
_REST_AFTER_USE_S = 60.0
_REST_AFTER_TPD_S = 30 * 60.0
_NOT_FOUND_RE = re.compile(r"\bNOT[_ ]FOUND\b", re.I)


class GroqSearchPool:
    """Round-robins browser_search calls across every (user Groq key, model)
    slot. Built once per job from the user's own keys."""

    def __init__(self, keys: list[tuple[str, str]]):
        self._slots = [
            {"api_key": k, "key_ref": ref, "model": model, "tag": tag, "free_at": 0.0}
            for k, ref in keys
            for model, tag in _SEARCH_MODELS
        ]
        self._next = 0
        self._lock = asyncio.Lock()

    def __bool__(self) -> bool:
        return bool(self._slots)

    async def acquire(self) -> dict | None:
        async with self._lock:
            now = time.monotonic()
            for i in range(len(self._slots)):
                slot = self._slots[(self._next + i) % len(self._slots)]
                if slot["free_at"] <= now:
                    self._next = (self._next + i + 1) % len(self._slots)
                    slot["free_at"] = now + _REST_AFTER_USE_S
                    return slot
            return None

    @staticmethod
    def rest(slot: dict, resp: httpx.Response) -> None:
        retry_after = resp.headers.get("retry-after")
        seconds = _REST_AFTER_USE_S
        if "per day" in resp.text.lower() or "tpd" in resp.text.lower():
            seconds = _REST_AFTER_TPD_S
        elif retry_after:
            try:
                seconds = max(float(retry_after), 5.0)
            except ValueError:
                pass
        slot["free_at"] = time.monotonic() + seconds


async def _browser_search(
    pool: GroqSearchPool, prompt: str, source: UrlReadSource, redis: Redis | None
) -> UrlReadResult | None:
    """One browser_search call on the next free slot. Returns None when no
    slot is free (caller degrades instead of waiting)."""
    slot = await pool.acquire()
    if slot is None:
        return None
    async with httpx.AsyncClient(timeout=BROWSER_SEARCH_TIMEOUT) as client:
        resp = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {slot['api_key']}"},
            json={
                "model": slot["model"],
                "messages": [{"role": "user", "content": prompt}],
                "tools": [{"type": "browser_search"}],
                "tool_choice": "required",
                # Per Groq's docs, "low" stops browser_search from running
                # extended, token-hungry browsing sessions.
                "reasoning_effort": "low",
                "max_completion_tokens": 1500,
            },
        )
    if resp.status_code == 429:
        GroqSearchPool.rest(slot, resp)
        return None
    resp.raise_for_status()
    data = resp.json()
    content = _CITATION_RE.sub("", data["choices"][0]["message"].get("content") or "").strip()

    if redis is not None:
        usage = data.get("usage") or {}
        try:
            await record_usage(
                redis, f"{slot['key_ref']}__{slot['tag']}", "groq",
                usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), headers=dict(resp.headers),
            )
        except Exception:
            logger.debug("browser_search usage capture failed", exc_info=True)

    if len(content) < MIN_CONTENT_LENGTH or _NOT_FOUND_RE.search(content[:300]):
        raise ContentUnavailable("browser_search_empty")
    markdown = content[:MAX_MARKDOWN_CHARS]
    return UrlReadResult(markdown=markdown, source=source, success=True, token_estimate=len(markdown) // 4)


def _visit_prompt(url: str) -> str:
    return (
        f"Visit {url} and summarize: what products/services this company offers, their industry, "
        "target markets, and any specific product names mentioned. Be factual and specific."
    )


def _search_prompt(company_name: str, location: str | None) -> str:
    where = f" located in {location}" if location else ""
    return (
        f'Search the web for the company "{company_name}"{where}. Report: 1) its official website '
        "if one exists, 2) what the company does, 3) the specific products and services it offers. "
        "Only report facts you actually found about THIS company. If you cannot find this specific "
        "company, reply with just NOT_FOUND."
    )


# ─────────────────────────── Entry point ───────────────────────────

async def _cached(redis: Redis | None, key: str) -> UrlReadResult | None:
    if redis is None:
        return None
    try:
        raw = await redis.get(key)
        if raw:
            result = UrlReadResult(**json.loads(raw))
            result.from_cache = True
            return result
    except Exception:
        logger.warning("Redis cache read failed for %s", key, exc_info=True)
    return None


async def _store(redis: Redis | None, key: str, result: UrlReadResult) -> None:
    if redis is not None and result.success:
        try:
            await redis.setex(key, CACHE_TTL_SECONDS, result.model_dump_json())
        except Exception:
            logger.warning("Redis cache write failed for %s", key, exc_info=True)


def normalize_url(url: str | None) -> str | None:
    if not url or not url.strip() or url.strip().lower() in ("n/a", "na", "none", "-", "null"):
        return None
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"
    return url


async def read_company_source(
    company_name: str,
    location: str | None,
    url: str | None,
    redis: Redis | None = None,
    search_pool: GroqSearchPool | None = None,
    jina_api_key: str | None = None,
    jina_key_ref: str | None = None,
) -> UrlReadResult:
    """Main entry point. Never raises - always degrades to name_location."""
    url = normalize_url(url)
    try:
        if url:
            result = await _read_url(url, redis, search_pool, jina_api_key, jina_key_ref)
            if result.success:
                return result
            # Dead or unreadable website: identify the company by name +
            # location instead, exactly like a row with no website at all.
            searched = await _search_company(company_name, location, redis, search_pool)
            if searched.success:
                return searched
            return result.model_copy(update={"error": f"{result.error}; {searched.error}"[:200]})
        return await _search_company(company_name, location, redis, search_pool)
    except Exception as exc:  # noqa: BLE001 - a read failure must never fail the company
        logger.info("Source read failed for %s (%s): %s", company_name, url, exc)
        return UrlReadResult(source=UrlReadSource.NAME_LOCATION, success=False, error=f"read_failed:{exc}"[:200])


async def _read_url(
    url: str,
    redis: Redis | None,
    search_pool: GroqSearchPool | None,
    jina_api_key: str | None,
    jina_key_ref: str | None,
) -> UrlReadResult:
    key = _cache_key(url)
    if (cached := await _cached(redis, key)) is not None:
        return cached

    errors: list[str] = []
    result: UrlReadResult | None = None
    try:
        result = await _fetch_direct(url)
    except ContentUnavailable as exc:
        if str(exc) == "connect_error":
            # The domain doesn't resolve / refuses connections: Jina can't
            # render it either, and asking a browsing model to "visit" a dead
            # URL just invites a made-up summary. Give up on the URL.
            return UrlReadResult(source=UrlReadSource.NAME_LOCATION, success=False, error="website_unreachable")
        errors.append(f"direct:{exc}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"direct:{type(exc).__name__}")

    if result is None and _jina_limiter.try_acquire(JINA_RPM_WITH_KEY if jina_api_key else JINA_RPM_NO_KEY):
        async def _no_fallback(*_args, **_kwargs) -> None:
            return None

        try:
            result = await jina_breaker.call(
                _fetch_jina, url, jina_api_key=jina_api_key, redis=redis, key_ref=jina_key_ref, fallback=_no_fallback
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"jina:{type(exc).__name__}")
        finally:
            await _sync_circuit_status(redis)
        if result is None:
            errors.append("jina:failed")
    elif result is None:
        errors.append("jina:rate_limited")

    if result is None and search_pool:
        try:
            result = await _browser_search(search_pool, _visit_prompt(url), UrlReadSource.COMPOUND_BETA, redis)
            if result is None:
                errors.append("browse:no_free_slot")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"browse:{type(exc).__name__}")

    if result is None:
        return UrlReadResult(source=UrlReadSource.NAME_LOCATION, success=False, error="; ".join(errors)[:200])
    await _store(redis, key, result)
    return result


async def _search_company(
    company_name: str, location: str | None, redis: Redis | None, search_pool: GroqSearchPool | None
) -> UrlReadResult:
    key = _search_cache_key(company_name, location)
    if (cached := await _cached(redis, key)) is not None:
        return cached
    if not search_pool:
        return UrlReadResult(source=UrlReadSource.NAME_LOCATION, success=False, error="no_website_no_search_key")
    try:
        result = await _browser_search(search_pool, _search_prompt(company_name, location), UrlReadSource.WEB_SEARCH, redis)
    except ContentUnavailable:
        return UrlReadResult(source=UrlReadSource.NAME_LOCATION, success=False, error="web_search_not_found")
    if result is None:
        return UrlReadResult(source=UrlReadSource.NAME_LOCATION, success=False, error="web_search_busy")
    await _store(redis, key, result)
    return result
