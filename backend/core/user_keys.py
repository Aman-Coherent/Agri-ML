"""Shared helpers for a user's active API keys. There is no system key pool
(see config.py), so these are the ONLY keys a job ever runs with.
Extracted from job_engine.py."""
from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import UserApiKey

# The agriculture analysis runs on one pooled model group (llm_router.AGRI_MODEL)
# that holds every Groq AND Mistral deployment, so a key for either provider is
# enough to run. More keys = more free capacity; a Groq key additionally
# enables web search for companies with no website.
_LLM_PROVIDERS = ("groq", "mistral")


async def ensure_required_keys(session: AsyncSession, user_id: str, mode: str | None = None) -> None:
    """Raises a 400 when the user has no usable LLM key, so they're told
    exactly what to add instead of the job failing later. `mode` is accepted
    for call-site compatibility; every job runs the same analysis now."""
    result = await session.execute(
        select(UserApiKey.provider).where(UserApiKey.user_id == user_id, UserApiKey.is_active.is_(True))
    )
    have = {row[0] for row in result.all()}
    if not have.intersection(_LLM_PROVIDERS):
        raise HTTPException(
            400,
            "Add your own Groq and/or Mistral API key in Settings -> API Keys before running this job.",
        )


async def load_user_keys(session: AsyncSession, user_id: str) -> list[dict]:
    result = await session.execute(
        select(UserApiKey).where(UserApiKey.user_id == user_id, UserApiKey.is_active.is_(True))
    )
    keys = result.scalars().all()
    return [
        {
            "id": k.id,
            "provider": k.provider,
            "api_key": k.api_key,
            "model_name": k.model_name,
            "base_url": k.base_url,
        }
        for k in keys
    ]
