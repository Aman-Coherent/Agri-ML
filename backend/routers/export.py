from __future__ import annotations

import re
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from backend.auth.jwt_verify import AuthenticatedUser, get_current_user
from backend.db.database import get_session
from backend.db.models import Project
from backend.storage.duckdb_queries import export_project, get_stats

router = APIRouter(prefix="/api/export", tags=["export"])


_KIND_FILE_LABEL = {"entities": "entities", "new_roles": "new_entities_and_roles", "products": "products"}


@router.get("/{project_id}")
async def export_products(
    project_id: str,
    fmt: str = Query(default="csv", pattern="^(csv|json)$"),
    kind: str = Query(default="products", pattern="^(entities|new_roles|products)$"),
    job_id: str | None = None,
    user: AuthenticatedUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    project = await session.get(Project, project_id)
    if project is None or project.user_id != user.id:
        raise HTTPException(404, "Project not found")

    tmp_dir = Path(tempfile.gettempdir()) / "product_generator_exports"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", project.name).strip("_") or "project"
    out_path = tmp_dir / f"{safe_name}_{_KIND_FILE_LABEL[kind]}_{project_id[:8]}.{fmt}"

    export_project(project_id, job_id, str(out_path), fmt=fmt, kind=kind)  # type: ignore[arg-type]

    media_type = "text/csv" if fmt == "csv" else "application/json"
    return FileResponse(out_path, media_type=media_type, filename=out_path.name)


@router.get("/{project_id}/stats")
async def project_stats(
    project_id: str,
    job_id: str | None = None,
    user: AuthenticatedUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    project = await session.get(Project, project_id)
    if project is None or project.user_id != user.id:
        raise HTTPException(404, "Project not found")
    return get_stats(project_id, job_id)
