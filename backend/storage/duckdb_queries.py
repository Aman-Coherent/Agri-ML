"""
Analytical queries over a project's Parquet part-files using DuckDB.
Used for: per-company product listing, the three exports (entities,
new/discovered roles, products) and aggregate stats. DuckDB pushes
filters/columns down into Parquet, so this stays fast even at 10M+ rows.

A company that was processed more than once (retry, re-run) has rows from
several jobs; every query keeps only its LATEST result (all rows of one
result share one `created_at`), unless scoped to a specific job_id.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import duckdb

from backend.storage.parquet_writer import (
    PRODUCT_PREFIX,
    ROLE_PREFIX,
    product_parquet_glob,
    project_dir,
    role_parquet_glob,
)

ExportKind = Literal["entities", "new_roles", "products"]

_SOURCE_LABEL_SQL = """CASE url_read_source
    WHEN 'web_search' THEN 'Web search'
    WHEN 'name_location' THEN 'Name + location only'
    WHEN 'none' THEN 'Name + location only'
    ELSE 'Website' END"""


def _has_files(project_id: str, prefix: str) -> bool:
    return any(project_dir(project_id).glob(f"{prefix}*.parquet"))


def _rows_cte(glob: str, job_id: str | None) -> tuple[str, list]:
    """`rows` CTE: the latest result per company, or one job's rows."""
    if job_id:
        return f"rows AS (SELECT * FROM read_parquet('{glob}') WHERE job_id = ?)", [job_id]
    return (
        f"""all_rows AS (SELECT * FROM read_parquet('{glob}')),
        latest AS (SELECT company_id, max(created_at) AS created_at FROM all_rows GROUP BY company_id),
        rows AS (SELECT a.* FROM all_rows a JOIN latest l USING (company_id, created_at))""",
        [],
    )


def get_products_for_company(project_id: str, company_id: str) -> list[dict]:
    if not _has_files(project_id, PRODUCT_PREFIX):
        return []
    cte, params = _rows_cte(product_parquet_glob(project_id), None)
    query = f"""
        WITH {cte}
        SELECT product_name, role, product_kind, product_source, entity, business_types, is_new_role
        FROM rows WHERE company_id = ?
        ORDER BY entity, role, product_name
    """
    con = duckdb.connect(":memory:")
    result = con.execute(query, [*params, company_id]).fetchall()
    cols = ["name", "category", "kind", "source", "entity", "business_types", "is_new_role"]
    return [dict(zip(cols, row)) for row in result]


def _export_query(project_id: str, kind: ExportKind, job_id: str | None) -> tuple[str, list] | None:
    if kind == "products":
        if not _has_files(project_id, PRODUCT_PREFIX):
            return None
        cte, params = _rows_cte(product_parquet_glob(project_id), job_id)
        return (
            f"""WITH {cte}
            SELECT native_id AS "Lead ID", company_name AS "Company Name", product_name AS "Product Name",
                   role AS "Role", entity AS "Entity", "group" AS "Group", business_types AS "Business Type",
                   CASE product_kind WHEN 'service' THEN 'Service' ELSE 'Product' END AS "Product / Service",
                   CASE product_source WHEN 'website' THEN 'Website' WHEN 'web_search' THEN 'Web search'
                        ELSE 'Inferred' END AS "Found On",
                   CASE WHEN is_new_role THEN 'NEW' ELSE '' END AS "New Role"
            FROM rows
            ORDER BY company_name, native_id, entity, role, product_name""",
            params,
        )

    if not _has_files(project_id, ROLE_PREFIX):
        return None
    cte, params = _rows_cte(role_parquet_glob(project_id), job_id)
    if kind == "entities":
        # One row per company per entity, listed roles only; business types
        # are the distinct options across that entity's roles.
        return (
            f"""WITH {cte},
            listed AS (SELECT * FROM rows WHERE NOT is_new_role)
            SELECT native_id AS "Lead ID", company_name AS "Company Name", location AS "Location",
                   url AS "Website", "group" AS "Group", entity AS "Entity",
                   string_agg(role, ', ' ORDER BY role_id) AS "Roles",
                   array_to_string(list_filter(list_distinct(flatten(list(string_split(business_types, ', ')
                       ORDER BY role_id))), x -> x <> ''), ', ') AS "Business Types",
                   any_value({_SOURCE_LABEL_SQL}) AS "Data Source",
                   round(any_value(confidence), 2) AS "Confidence"
            FROM listed
            GROUP BY company_id, native_id, company_name, location, url, "group", entity
            ORDER BY company_name, native_id, min(role_id)""",
            params,
        )
    # new_roles: every new entity, and every new role under a listed entity.
    return (
        f"""WITH {cte}
        SELECT native_id AS "Lead ID", company_name AS "Company Name", location AS "Location",
               url AS "Website", "group" AS "Group", entity AS "Entity",
               CASE WHEN is_new_entity THEN 'New entity' ELSE 'Listed entity' END AS "Entity Status",
               role AS "New Role", business_types AS "Business Types", note AS "Reason",
               CASE WHEN is_agriculture THEN 'Yes' ELSE 'No' END AS "Agriculture",
               {_SOURCE_LABEL_SQL} AS "Data Source"
        FROM rows WHERE is_new_role
        ORDER BY is_new_entity DESC, "group", entity, role, company_name""",
        params,
    )


def export_project(
    project_id: str,
    job_id: str | None,
    out_path: str,
    fmt: Literal["csv", "json"] = "csv",
    kind: ExportKind = "products",
) -> Path:
    """Streams one export out to disk using DuckDB's native COPY, which never
    materializes the whole result set in Python memory."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    built = _export_query(project_id, kind, job_id)
    if built is None:
        out.write_text("")
        return out
    query, params = built
    options = "(HEADER, DELIMITER ',')" if fmt == "csv" else "(FORMAT JSON, ARRAY true)"
    con = duckdb.connect(":memory:")
    con.execute(f"COPY ({query}) TO '{out.as_posix()}' {options}", params)
    return out


def get_stats(project_id: str, job_id: str | None = None) -> dict:
    empty = {"total_companies": 0, "total_products": 0, "new_roles": 0, "by_category": {}}
    if not _has_files(project_id, ROLE_PREFIX):
        return empty
    con = duckdb.connect(":memory:")
    cte, params = _rows_cte(role_parquet_glob(project_id), job_id)
    total_companies, new_roles = con.execute(
        f"WITH {cte} SELECT count(DISTINCT company_id), count(*) FILTER (WHERE is_new_role) FROM rows", params
    ).fetchone()
    by_entity = con.execute(
        f"""WITH {cte} SELECT entity, count(DISTINCT company_id) FROM rows
            WHERE NOT is_new_role GROUP BY entity ORDER BY 2 DESC""",
        params,
    ).fetchall()

    total_products = 0
    if _has_files(project_id, PRODUCT_PREFIX):
        pcte, pparams = _rows_cte(product_parquet_glob(project_id), job_id)
        total_products = con.execute(f"WITH {pcte} SELECT count(*) FROM rows", pparams).fetchone()[0]

    return {
        "total_companies": total_companies,
        "total_products": total_products,
        "new_roles": new_roles,
        "by_category": {row[0]: row[1] for row in by_entity},
    }
