"""
Runs one job to completion: fans out per-company processing with a bounded
worker pool, supports cooperative pause/cancel via Redis-backed control
flags (so the FastAPI process can signal a running ARQ worker process),
writes checkpoints, batches Parquet writes, and publishes SSE progress
events over Redis pub/sub.

IMPORTANT: SQLAlchemy's AsyncSession is NOT safe to share across
concurrently-running coroutines. Every concurrent company task opens its
own short-lived session via `session_factory`; only the job-level
bookkeeping (start/finish) uses a single session held by the caller.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from datetime import datetime, timezone

from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.core.checkpoint import append_checkpoint, load_completed_ids
from backend.core.job_control import (
    acquire_run_lock as _acquire_run_lock,
    get_control as _get_control,
    heartbeat_run_lock as _heartbeat_run_lock,
    release_run_lock as _release_run_lock,
    request_cancel,
    request_pause,
    request_resume,
    run_lock_key as _run_lock_key,
)
from backend.core.llm_router import build_router, pick_groq_search_keys, pick_jina_key
from backend.core.models import CompanyResult, SSEEvent, SSEEventType
from backend.core.pipeline import process_company
from backend.core.url_reader import GroqSearchPool
from backend.core.user_keys import load_user_keys as _load_user_keys
from backend.db.models import CompanyInput, Job
from backend.storage.parquet_writer import ParquetBatchWriter
from backend.workers.sse_publisher import publish_event

logger = logging.getLogger("job_engine")

# request_pause/request_cancel/request_resume and _run_lock_key are
# re-exported here (unused-import ignored) since routers/jobs.py and
# workers/arq_worker.py import them from this module - see job_control.py
# for the actual generic (job-table-agnostic) implementation, now shared
# with EmailJobEngine.


class JobEngine:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession], redis: Redis, concurrency: int = 20):
        self.session_factory = session_factory
        self.redis = redis
        self.concurrency = concurrency
        self._counters_lock = asyncio.Lock()
        # Concurrent unsynchronized appends to the same checkpoint file (one
        # `open(..., "a")` per call, from up to `concurrency` tasks at once)
        # can interleave partial writes and corrupt a line, which is silently
        # dropped on resume — for "failed" companies (the only ones a
        # corrupted line actually affects; "done" ones are already excluded
        # by the DB-level query regardless) this just means an unnecessary
        # re-processing, not data loss, but it's needless waste at scale.
        self._checkpoint_lock = asyncio.Lock()

    async def _publish(self, job_id: str, event_type: SSEEventType, data: dict) -> None:
        event = SSEEvent(event=event_type, job_id=job_id, data=data)
        await publish_event(self.redis, event)

    async def run(self, job_id: str) -> None:
        lock_token = await _acquire_run_lock(self.redis, job_id)
        if lock_token is None:
            logger.warning(
                "Job %s already has an active run in progress elsewhere; skipping this invocation.",
                job_id,
            )
            return

        heartbeat_task = asyncio.create_task(_heartbeat_run_lock(self.redis, job_id, lock_token))
        try:
            await self._run_locked(job_id)
        except Exception as exc:
            # Without this, an unhandled crash (e.g. a DB hiccup, a bug) left
            # the job stuck at status="RUNNING" forever with no active lock
            # and no way to distinguish it from a genuinely healthy run - the
            # UI would show a live job that had actually died. Surface it as
            # a terminal FAILED state so Resume / Retry failed become
            # available again, then re-raise so ARQ's own retry-on-crash
            # (max_tries=3) still kicks in for transient failures.
            logger.exception("Job %s crashed with an unhandled exception", job_id)
            await self._mark_failed(job_id, str(exc) or exc.__class__.__name__)
            raise
        finally:
            heartbeat_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat_task
            await _release_run_lock(self.redis, job_id, lock_token)

    async def _mark_failed(self, job_id: str, error: str) -> None:
        try:
            async with self.session_factory() as session:
                job = await session.get(Job, job_id)
                if job is None or job.status in ("COMPLETED", "CANCELLED"):
                    return
                job.status = "FAILED"
                job.error_message = error[:2000]
                job.finished_at = datetime.now(timezone.utc)
                await session.commit()
                done, failed, total = job.done, job.failed, job.total
            await self._publish(
                job_id,
                SSEEventType.ERROR,
                {"error": error[:2000], "done": done, "failed": failed, "total": total},
            )
        except Exception:
            logger.exception("Failed to persist crash state for job %s", job_id)

    async def _run_locked(self, job_id: str) -> None:
        async with self.session_factory() as session:
            job = await session.get(Job, job_id)
            if job is None:
                logger.error("Job %s not found", job_id)
                return

            # A Stop request against a QUEUED (not yet started) job already
            # wrote status="CANCELLED" directly (see cancel_job in
            # routers/jobs.py, which has no running engine loop to signal
            # yet) before this ARQ-enqueued invocation ever got to run.
            # Without this guard, the unconditional `request_resume` +
            # `job.status = "RUNNING"` below would silently resurrect that
            # cancelled job back to life the moment ARQ picked it up.
            if job.status == "CANCELLED":
                logger.info("Job %s was cancelled before it started running; skipping.", job_id)
                return

            await request_resume(self.redis, job_id)
            job.status = "RUNNING"
            job.started_at = datetime.now(timezone.utc)
            await session.commit()

            # Only the 5 columns processing needs, as lightweight rows - full
            # ORM objects for a 6-lakh-company project would cost ~1 GB.
            companies_result = await session.execute(
                select(
                    CompanyInput.id, CompanyInput.native_id, CompanyInput.company_name,
                    CompanyInput.location, CompanyInput.url,
                )
                .where(CompanyInput.project_id == job.project_id, CompanyInput.status != "done")
                .order_by(CompanyInput.row_index)
            )
            companies = list(companies_result.all())
            user_keys = await _load_user_keys(session, job.user_id)
            concurrency = job.concurrency or self.concurrency
            project_id = job.project_id

        await self._publish(job_id, SSEEventType.STATUS_CHANGE, {"status": "RUNNING"})

        completed_ids = await load_completed_ids(job_id)
        pending = [c for c in companies if c.id not in completed_ids]

        router = build_router(user_keys)
        # Dynamic attribute (not a real litellm.Router field) so
        # structured_output.generate_structured can record per-key usage
        # without threading a redis handle through every layer.
        router._usage_redis = self.redis
        search_pool = GroqSearchPool(pick_groq_search_keys(user_keys))
        jina_api_key, jina_key_ref = pick_jina_key(user_keys)
        writer = ParquetBatchWriter(job_id, project_id, batch_size=100)

        state = {"paused": False, "cancelled": False}
        tasks: list[asyncio.Task] = []

        async def _watch_control() -> None:
            """Polls the Redis control flag while the batch runs so Stop/Pause
            take effect immediately instead of only being checked once per
            company right before it starts (the old behavior). That old
            cooperative-only check had two failure modes: (1) a company
            already mid-flight (doing an LLM/URL call) had no way to be
            interrupted, so Stop had to wait for the entire in-flight wave
            (up to `concurrency`, e.g. 200) to finish naturally, and (2) once
            every remaining company had already passed its one-time check,
            Stop was silently ignored for the rest of the job and it ran to
            COMPLETED anyway. Directly cancelling every task's asyncio.Task
            here interrupts in-flight network calls immediately, regardless
            of where each one currently is.
            """
            while True:
                control = await _get_control(self.redis, job_id)
                if control == "cancel":
                    state["cancelled"] = True
                    for t in tasks:
                        t.cancel()
                    return
                if control == "pause":
                    state["paused"] = True
                    for t in tasks:
                        t.cancel()
                    return
                await asyncio.sleep(1)

        async def _process_one(company) -> None:
            # The try/except wraps the ENTIRE body: _watch_control can
            # t.cancel() a worker anywhere in here, and since
            # asyncio.CancelledError subclasses BaseException (not
            # Exception), letting it escape would blow straight through
            # asyncio.gather() and the generic `except Exception` in
            # JobEngine.run(), acting as an unhandled crash instead of the
            # quiet "not processed this run" outcome intended here.
            try:
                result: CompanyResult = await process_company(
                    router,
                    self.redis,
                    company.id,
                    company.native_id,
                    company.company_name,
                    company.location,
                    company.url,
                    search_pool=search_pool,
                    jina_api_key=jina_api_key,
                    jina_key_ref=jina_key_ref,
                )
                writer.add(result)
                async with self._checkpoint_lock:
                    await append_checkpoint(
                        job_id, company.id, "done" if result.success else "failed", result.error
                    )

                async with self._counters_lock:
                    async with self.session_factory() as write_session:
                        job_row, done, failed, total = await self._apply_result(
                            write_session, job_id, company.id, result
                        )

                await self._publish(
                    job_id,
                    SSEEventType.COMPANY_DONE if result.success else SSEEventType.COMPANY_FAILED,
                    {
                        "company_id": company.id,
                        "company_name": company.company_name,
                        "url_source": result.url_read.source.value,
                        "display_label": result.profile.summary_label() if result.profile else None,
                        "products_count": len(result.profile.products) if result.profile else 0,
                        "error": result.error,
                    },
                )
                pct = round(100 * (done + failed) / total, 2) if total else 0.0
                await self._publish(
                    job_id, SSEEventType.PROGRESS, {"done": done, "failed": failed, "total": total, "percent": pct}
                )
            except asyncio.CancelledError:
                # _watch_control cancelled us mid-flight (Stop/Pause).
                # state["cancelled"/"paused"] is already set by the watcher;
                # swallow here rather than re-raising so asyncio.gather()
                # below doesn't propagate it out of _run_locked as a real
                # crash. If the checkpoint was already written before
                # cancellation landed, this company is correctly treated as
                # done/failed already; otherwise it stays "pending" and is
                # picked up again on the next run.
                return

        # A fixed pool of `concurrency` workers pulling from one shared
        # iterator, instead of one asyncio.Task per company: at 6 lakh
        # companies, creating every task up front costs gigabytes and makes
        # Stop/Pause cancel hundreds of thousands of tasks.
        remaining = iter(pending)

        async def _worker() -> None:
            for company in remaining:
                if state["cancelled"] or state["paused"]:
                    return
                await _process_one(company)

        tasks.extend(asyncio.create_task(_worker()) for _ in range(min(concurrency, len(pending))))
        watcher_task = asyncio.create_task(_watch_control()) if tasks else None
        try:
            if tasks:
                await asyncio.gather(*tasks)
        finally:
            if watcher_task is not None:
                watcher_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await watcher_task

        writer.flush()

        async with self.session_factory() as session:
            job = await session.get(Job, job_id)
            if state["cancelled"]:
                job.status = "CANCELLED"
            elif state["paused"]:
                job.status = "PAUSED"
            else:
                job.status = "COMPLETED"
            job.finished_at = datetime.now(timezone.utc)
            await session.commit()
            final_status, done, failed, total = job.status, job.done, job.failed, job.total

        await self._publish(
            job_id,
            SSEEventType.COMPLETE if final_status == "COMPLETED" else SSEEventType.STATUS_CHANGE,
            {"status": final_status, "done": done, "failed": failed, "total": total},
        )

    @staticmethod
    async def _apply_result(
        session: AsyncSession, job_id: str, company_id: str, result: CompanyResult
    ) -> tuple[Job, int, int, int]:
        """Applies one company's result inside its own short-lived session/transaction."""
        job = await session.get(Job, job_id)
        company = await session.get(CompanyInput, company_id)

        company.status = "done" if result.success else "failed"
        company.url_read_source = result.url_read.source.value if result.url_read else None
        company.url_read_success = result.url_read.success if result.url_read else False
        company.url_markdown_tokens = result.url_read.token_estimate if result.url_read else 0
        company.url_error = (result.url_read.error or "")[:200] or None if result.url_read else None
        company.processing_time_ms = result.processing_time_ms
        company.last_job_id = job_id

        if result.profile:
            profile = result.profile
            company.description = profile.description
            company.is_agriculture = profile.is_agriculture
            company.roles_json = json.dumps([r.model_dump() for r in profile.roles])
            company.has_new_roles = profile.has_new
            company.display_label = profile.summary_label()
            company.classification_confidence = profile.confidence
            company.products_count = len(profile.products)
            company.error_message = None

        if not result.success:
            company.error_message = result.error

        job.done = (job.done or 0) + (1 if result.success else 0)
        job.failed = (job.failed or 0) + (0 if result.success else 1)

        await session.commit()
        return job, job.done, job.failed, job.total
