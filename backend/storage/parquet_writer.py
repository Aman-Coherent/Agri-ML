"""
Batches CompanyResult rows and appends them to per-project Parquet part
files using Polars - two series per project:

  roles_part_NNNNNN.parquet     one row per (company, role) - listed AND new
                                roles. Feeds the entities export and the
                                new/discovered roles export.
  products_part_NNNNNN.parquet  one row per (company, product/service), with
                                the role it belongs to. Feeds the products
                                export and the per-company product drawer.

Batching (every N companies) avoids the overhead of writing a tiny Parquet
file per company while still bounding memory to one batch at a time. Every
row of one company result shares the same `created_at`, which is how
queries pick each company's LATEST result when it has been re-processed
(see duckdb_queries.py).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from backend.config import DATA_DIR
from backend.core.models import CompanyResult

PROJECTS_DIR = DATA_DIR / "projects"
PROJECTS_DIR.mkdir(parents=True, exist_ok=True)

ROLE_PREFIX = "roles_part_"
PRODUCT_PREFIX = "products_part_"

_COMPANY_SCHEMA = {
    "job_id": pl.Utf8,
    "project_id": pl.Utf8,
    "company_id": pl.Utf8,
    "native_id": pl.Utf8,
    "company_name": pl.Utf8,
    "location": pl.Utf8,
    "url": pl.Utf8,
    "url_read_source": pl.Utf8,
    "created_at": pl.Utf8,
}
_ROLE_SCHEMA = {
    "group": pl.Utf8,
    "entity": pl.Utf8,
    "role": pl.Utf8,
    "business_types": pl.Utf8,  # ", "-joined
    "is_new_role": pl.Boolean,
    "is_new_entity": pl.Boolean,
}
ROLE_ROWS_SCHEMA = {
    **_COMPANY_SCHEMA,
    "description": pl.Utf8,
    "is_agriculture": pl.Boolean,
    "confidence": pl.Float64,
    "role_ref": pl.Utf8,
    "role_id": pl.Int64,
    **_ROLE_SCHEMA,
    "note": pl.Utf8,
}
PRODUCT_ROWS_SCHEMA = {
    **_COMPANY_SCHEMA,
    "product_name": pl.Utf8,
    "product_kind": pl.Utf8,
    "product_source": pl.Utf8,
    "role_id": pl.Int64,
    **_ROLE_SCHEMA,
}


def project_dir(project_id: str) -> Path:
    d = PROJECTS_DIR / project_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def role_parquet_glob(project_id: str) -> str:
    """DuckDB/Polars glob pattern matching every role part file for a project."""
    return (project_dir(project_id) / f"{ROLE_PREFIX}*.parquet").as_posix()


def product_parquet_glob(project_id: str) -> str:
    return (project_dir(project_id) / f"{PRODUCT_PREFIX}*.parquet").as_posix()


def clear_project_products(project_id: str) -> int:
    """Deletes every existing result part-file for a project.

    Must be called whenever a project's CompanyInput rows are replaced
    (fresh/re-upload CSV, which assigns brand new company_id values - see
    routers/projects.py upload_csv). Without this, previously-written Parquet
    rows keep referencing company_ids that no longer exist anywhere in the
    DB, permanently inflating every future stats/export query for this
    project with orphaned "ghost" companies and products.
    """
    deleted = 0
    for pattern in (f"{ROLE_PREFIX}*.parquet", f"{PRODUCT_PREFIX}*.parquet"):
        for path in project_dir(project_id).glob(pattern):
            path.unlink(missing_ok=True)
            deleted += 1
    return deleted


def _rows_for_result(job_id: str, project_id: str, result: CompanyResult) -> tuple[list[dict], list[dict]]:
    if not result.success or result.profile is None:
        return [], []
    now = datetime.now(timezone.utc).isoformat()
    base = {
        "job_id": job_id,
        "project_id": project_id,
        "company_id": result.company_id,
        "native_id": result.native_id,
        "company_name": result.company_name,
        "location": result.location,
        "url": result.url,
        "url_read_source": result.url_read.source.value,
        "created_at": now,
    }
    profile = result.profile

    def role_cols(r) -> dict:
        return {
            "role_id": r.role_id,
            "group": r.group,
            "entity": r.entity,
            "role": r.role,
            "business_types": ", ".join(r.business_types),
            "is_new_role": r.is_new_role,
            "is_new_entity": r.is_new_entity,
        }

    role_rows = [
        {
            **base,
            "description": profile.description,
            "is_agriculture": profile.is_agriculture,
            "confidence": profile.confidence,
            "role_ref": r.ref,
            **role_cols(r),
            "note": r.note,
        }
        for r in profile.roles
    ]
    by_ref = profile.role_by_ref()
    product_rows = [
        {
            **base,
            "product_name": p.name,
            "product_kind": p.kind,
            "product_source": p.source,
            **role_cols(by_ref[p.role_ref]),
        }
        for p in profile.products
        if p.role_ref in by_ref
    ]
    return role_rows, product_rows


class ParquetBatchWriter:
    """
    Accumulates CompanyResult rows and flushes NEW part files every
    `batch_size` companies (e.g. roles_part_000042.parquet). This avoids the
    O(n) read-modify-write-whole-file cost a single growing Parquet file
    would incur at 6-lakh-company scale. DuckDB reads all part files via a
    glob pattern with zero extra code.
    """

    def __init__(self, job_id: str, project_id: str, batch_size: int = 100):
        self.job_id = job_id
        self.project_id = project_id
        self.batch_size = batch_size
        self._roles: list[dict] = []
        self._products: list[dict] = []
        self._companies_buffered = 0
        self._part_index = self._next_part_index()

    def _next_part_index(self) -> int:
        existing = [
            *project_dir(self.project_id).glob(f"{ROLE_PREFIX}*.parquet"),
            *project_dir(self.project_id).glob(f"{PRODUCT_PREFIX}*.parquet"),
        ]
        if not existing:
            return 0
        return max(int(p.stem.split("_")[-1]) for p in existing) + 1

    def add(self, result: CompanyResult) -> None:
        roles, products = _rows_for_result(self.job_id, self.project_id, result)
        if not roles:
            return
        self._roles.extend(roles)
        self._products.extend(products)
        self._companies_buffered += 1
        if self._companies_buffered >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self._roles:
            return
        d = project_dir(self.project_id)
        suffix = f"{self._part_index:06d}.parquet"
        pl.DataFrame(self._roles, schema=ROLE_ROWS_SCHEMA).write_parquet(d / f"{ROLE_PREFIX}{suffix}")
        if self._products:
            pl.DataFrame(self._products, schema=PRODUCT_ROWS_SCHEMA).write_parquet(d / f"{PRODUCT_PREFIX}{suffix}")
        self._part_index += 1
        self._roles.clear()
        self._products.clear()
        self._companies_buffered = 0
