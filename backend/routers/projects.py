from __future__ import annotations

import csv
import io
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import delete, insert, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.auth.jwt_verify import AuthenticatedUser, get_current_user
from backend.core.csv_columns import resolve_columns
from backend.core.url_reader import normalize_url
from backend.db.database import get_session
from backend.db.models import CompanyInput, Job, Project
from backend.storage.parquet_writer import clear_project_products

router = APIRouter(prefix="/api/projects", tags=["projects"])

MAX_ROWS = 1_000_000
INSERT_CHUNK = 5_000


class ProjectOut(BaseModel):
    id: str
    name: str
    description: str | None
    mode: str
    total_companies: int
    created_at: datetime
    updated_at: datetime
    latest_job_status: str | None = None

    class Config:
        from_attributes = True


class CreateProjectRequest(BaseModel):
    name: str
    description: str | None = None
    mode: str = "both"  # legacy field; every job now runs the full roles + products analysis


@router.get("", response_model=list[ProjectOut])
async def list_projects(
    user: AuthenticatedUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(
        select(Project).where(Project.user_id == user.id).order_by(Project.updated_at.desc())
    )
    projects = result.scalars().all()

    out = []
    for p in projects:
        job_result = await session.execute(
            select(Job).where(Job.project_id == p.id).order_by(Job.created_at.desc()).limit(1)
        )
        latest_job = job_result.scalars().first()
        item = ProjectOut.model_validate(p)
        item.latest_job_status = latest_job.status if latest_job else None
        out.append(item)
    return out


@router.post("", response_model=ProjectOut)
async def create_project(
    body: CreateProjectRequest,
    user: AuthenticatedUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    if body.mode not in ("classification", "generation", "both"):
        raise HTTPException(400, "mode must be one of: classification, generation, both")

    project = Project(user_id=user.id, name=body.name, description=body.description, mode=body.mode)
    session.add(project)
    await session.commit()
    await session.refresh(project)
    return ProjectOut.model_validate(project)


@router.get("/{project_id}", response_model=ProjectOut)
async def get_project(
    project_id: str,
    user: AuthenticatedUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    project = await _get_owned_project(session, project_id, user.id)
    return ProjectOut.model_validate(project)


@router.delete("/{project_id}")
async def delete_project(
    project_id: str,
    user: AuthenticatedUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    project = await _get_owned_project(session, project_id, user.id)
    await session.execute(delete(Project).where(Project.id == project.id))
    await session.commit()
    return {"status": "deleted"}


class CsvValidationError(BaseModel):
    row: int
    message: str


class UploadCsvResponse(BaseModel):
    total_rows: int
    with_website: int = 0
    without_website: int = 0
    preview: list[dict]
    errors: list[CsvValidationError]
    detected_columns: dict[str, str | None]


@router.post("/{project_id}/upload-csv", response_model=UploadCsvResponse)
async def upload_csv(
    project_id: str,
    file: UploadFile,
    user: AuthenticatedUser = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    project = await _get_owned_project(session, project_id, user.id)

    raw = await file.read()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise HTTPException(400, "CSV must be UTF-8 encoded")

    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        raise HTTPException(400, "CSV file is empty")

    fieldnames = [f for f in reader.fieldnames if f is not None]
    company_col, url_col, location_cols, id_col = resolve_columns(fieldnames)

    if company_col is None:
        raise HTTPException(
            400,
            "Could not find a company name column in your CSV. Found these columns: "
            f"{', '.join(fieldnames)}. Rename one of them to something like 'Company Name', "
            "'Company', or 'Business Name'.",
        )
    if id_col is None:
        raise HTTPException(
            400,
            "Could not find a Lead ID / Native ID column in your CSV - every export needs it to "
            f"match results back to your records. Found these columns: {', '.join(fieldnames)}. "
            "Rename the id column to something like 'Lead ID' or 'Native ID'.",
        )

    detected_columns = {
        "native_id": id_col,
        "company_name": company_col,
        "url": url_col,
        "location": " + ".join(location_cols) if location_cols else None,
    }

    errors: list[CsvValidationError] = []
    rows: list[dict] = []
    for idx, raw_row in enumerate(reader):
        company_name = (raw_row.get(company_col) or "").strip()
        if not company_name:
            errors.append(CsvValidationError(row=idx + 2, message="Missing company name"))
            continue
        native_id = (raw_row.get(id_col) or "").strip()
        if not native_id:
            errors.append(CsvValidationError(row=idx + 2, message=f"Missing {id_col}"))
            continue

        url_val = normalize_url(raw_row.get(url_col)) if url_col else None
        location_val = ", ".join(
            part for c in location_cols if (part := (raw_row.get(c) or "").strip())
        )

        rows.append(
            {
                "native_id": native_id[:255],
                "company_name": company_name[:255],
                "location": (location_val or None) and location_val[:255],
                "url": url_val and url_val[:500],
            }
        )
        if len(rows) > MAX_ROWS:
            raise HTTPException(400, f"CSV exceeds the maximum of {MAX_ROWS:,} rows")

    # Replace existing companies for this project with the new upload. New
    # CompanyInput rows get brand new auto-generated IDs, so any Parquet
    # product data already written for the OLD company_id values becomes
    # permanently unreachable garbage that would otherwise silently inflate
    # every future stats/export query for this project - clear it now.
    await session.execute(delete(CompanyInput).where(CompanyInput.project_id == project.id))
    clear_project_products(project.id)

    # Bulk INSERT in chunks: adding 6 lakh ORM objects one by one to a single
    # session takes minutes and gigabytes of memory; executemany batches are
    # orders of magnitude faster and keep memory flat.
    for start in range(0, len(rows), INSERT_CHUNK):
        chunk = rows[start : start + INSERT_CHUNK]
        await session.execute(
            insert(CompanyInput),
            [{"project_id": project.id, "row_index": start + i, **row} for i, row in enumerate(chunk)],
        )

    project.total_companies = len(rows)
    await session.commit()

    with_website = sum(1 for r in rows if r["url"])
    return UploadCsvResponse(
        total_rows=len(rows),
        with_website=with_website,
        without_website=len(rows) - with_website,
        preview=rows[:5],
        errors=errors[:20],
        detected_columns=detected_columns,
    )


async def _get_owned_project(session: AsyncSession, project_id: str, user_id: str) -> Project:
    project = await session.get(Project, project_id)
    if project is None or project.user_id != user_id:
        raise HTTPException(404, "Project not found")
    return project
