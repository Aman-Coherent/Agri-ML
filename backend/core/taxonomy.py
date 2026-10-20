"""
The agriculture entity/role taxonomy every company is classified against.

Source of truth: backend/taxonomy/entities_and_roles.csv (columns Group,
Entity, Role, Code, Business type allowed). Each row gets a stable numeric
id = its 1-based row position, and the LLM answers with those ids instead of
names - the exact role/entity/group names are then looked up here, so the
output can never contain a misspelled or paraphrased role.

"Business type allowed" is a group of options, e.g. "Exporter / Importer" or
"Distributor / Wholesaler / Trader / Supplier" (or "A + B" for roles that
combine two groups). The LLM must write only the option(s) that actually
apply to the company - e.g. just "Importer" - never the whole group.
"""
from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

TAXONOMY_CSV = Path(__file__).resolve().parent.parent / "taxonomy" / "entities_and_roles.csv"


@dataclass(frozen=True)
class Role:
    id: int
    group: str
    entity: str
    name: str
    code: str
    business_types_allowed: tuple[str, ...]  # individual options, e.g. ("Exporter", "Importer")


def split_business_types(allowed: str) -> tuple[str, ...]:
    """'Exporter / Importer + Distributor / Wholesaler' -> ('Exporter',
    'Importer', 'Distributor', 'Wholesaler'). Empty string -> ()."""
    options: list[str] = []
    for part in allowed.split("+"):
        for option in part.split("/"):
            option = option.strip()
            if option and option not in options:
                options.append(option)
    return tuple(options)


@dataclass(frozen=True)
class Taxonomy:
    roles: tuple[Role, ...]
    version: str  # short content hash - changes whenever the CSV changes

    @property
    def by_id(self) -> dict[int, Role]:
        return {r.id: r for r in self.roles}

    @property
    def groups(self) -> list[str]:
        return list(dict.fromkeys(r.group for r in self.roles))

    @property
    def entities(self) -> list[str]:
        return list(dict.fromkeys(r.entity for r in self.roles))

    def group_of_entity(self, entity: str) -> str | None:
        for r in self.roles:
            if r.entity.lower() == entity.strip().lower():
                return r.group
        return None

    def canonical_entity(self, entity: str) -> str | None:
        """Exact listed spelling of an entity name, matched case-insensitively."""
        for name in self.entities:
            if name.lower() == entity.strip().lower():
                return name
        return None

    def canonical_group(self, group: str) -> str | None:
        for name in self.groups:
            if name.lower() == group.strip().lower():
                return name
        return None

    def code_legend(self) -> dict[str, tuple[str, ...]]:
        legend: dict[str, tuple[str, ...]] = {}
        for r in self.roles:
            legend.setdefault(r.code, r.business_types_allowed)
        return legend

    def render_for_prompt(self) -> str:
        """Compact, byte-stable rendering (~2.6K tokens for 240 roles) used as
        the static prefix of every analysis prompt. Kept identical across
        companies so provider-side prompt caching can reuse it."""
        lines = ["BUSINESS TYPE CODES (the options each code allows):"]
        for code, options in self.code_legend().items():
            if code:
                label = " | ".join(options) if options else "no business type (leave business_types empty)"
                lines.append(f"  {code} = {label}")
        lines.append("  (no code) = no business type (leave business_types empty)")
        lines.append("")
        lines.append("ROLE LIST - format: Entity: <role id> <role name> (<code>); ...")
        current_group = None
        by_entity: dict[str, list[Role]] = {}
        for r in self.roles:
            by_entity.setdefault(r.entity, []).append(r)
        for entity, roles in by_entity.items():
            if roles[0].group != current_group:
                current_group = roles[0].group
                lines.append(f"# GROUP: {current_group}")
            parts = [f"{r.id} {r.name}" + (f" ({r.code})" if r.code else "") for r in roles]
            lines.append(f"{entity}: " + "; ".join(parts))
        return "\n".join(lines)


@lru_cache
def get_taxonomy(path: Path = TAXONOMY_CSV) -> Taxonomy:
    raw = path.read_bytes()
    rows = list(csv.DictReader(raw.decode("utf-8-sig").splitlines()))
    roles = tuple(
        Role(
            id=i,
            group=row["Group"].strip(),
            entity=row["Entity"].strip(),
            name=row["Role"].strip(),
            code=(row.get("Code") or "").strip(),
            business_types_allowed=split_business_types(row.get("Business type allowed") or ""),
        )
        for i, row in enumerate(rows, start=1)
    )
    if not roles:
        raise RuntimeError(f"Taxonomy file {path} has no rows")
    return Taxonomy(roles=roles, version=hashlib.sha256(raw).hexdigest()[:10])
