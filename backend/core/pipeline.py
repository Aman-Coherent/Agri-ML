"""
Per-company pipeline: read source (website, or web search when there's no
website) -> one agriculture analysis call (entities/roles + products).

Every job runs the full analysis - the old classification/generation
`mode` split no longer applies, since roles and products now come from the
same single LLM call.
"""
from __future__ import annotations

import time

from litellm import Router
from redis.asyncio import Redis

from backend.core.agri_analyzer import analyze_company
from backend.core.models import CompanyResult, UrlReadResult
from backend.core.url_reader import GroqSearchPool, read_company_source


async def process_company(
    router: Router,
    redis: Redis | None,
    company_id: str,
    native_id: str | None,
    company_name: str,
    location: str | None,
    url: str | None,
    search_pool: GroqSearchPool | None = None,
    jina_api_key: str | None = None,
    jina_key_ref: str | None = None,
) -> CompanyResult:
    """`search_pool` (every one of the user's Groq keys, see
    llm_router.pick_groq_search_keys) and the Jina key are built ONCE per job
    by JobEngine and shared by every company, so slot rotation/rest state is
    job-wide. Jina has no system-level key at all - None here means "use
    Jina's public unauthenticated tier", not a misconfiguration."""
    start = time.monotonic()
    url_read = UrlReadResult()
    try:
        url_read = await read_company_source(
            company_name,
            location,
            url,
            redis=redis,
            search_pool=search_pool,
            jina_api_key=jina_api_key,
            jina_key_ref=jina_key_ref,
        )
        profile = await analyze_company(router, company_name, location, url, url_read)
        return CompanyResult(
            company_id=company_id,
            native_id=native_id,
            company_name=company_name,
            location=location,
            url=url,
            url_read=url_read,
            profile=profile,
            success=True,
            processing_time_ms=int((time.monotonic() - start) * 1000),
        )
    except Exception as exc:  # noqa: BLE001 - must never crash the whole batch
        return CompanyResult(
            company_id=company_id,
            native_id=native_id,
            company_name=company_name,
            location=location,
            url=url,
            url_read=url_read,
            success=False,
            error=str(exc)[:2000],
            processing_time_ms=int((time.monotonic() - start) * 1000),
        )
