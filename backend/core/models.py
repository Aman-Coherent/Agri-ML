"""
All Pydantic schemas shared across the pipeline: URL reading results, the
agriculture analysis the LLM returns (entity/role classification against
backend/taxonomy + a 10-50 item product/service catalog, in ONE call), and
the resolved per-company result that gets stored and exported.
"""
from __future__ import annotations

import re
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator

from backend.core.taxonomy import get_taxonomy


# ─────────────────────────── URL reading ───────────────────────────

class UrlReadSource(str, Enum):
    DIRECT_FETCH = "direct_fetch"  # our own homepage(+product pages) fetch - the fast, free default
    JINA_READER = "jina_reader"
    COMPOUND_BETA = "compound_beta"  # now Groq browser_search on the URL; value kept for stored rows
    WEB_SEARCH = "web_search"  # no URL given: Groq browser_search on name + location
    NAME_LOCATION = "name_location"
    NONE = "none"


class UrlReadResult(BaseModel):
    markdown: str = ""
    source: UrlReadSource = UrlReadSource.NONE
    success: bool = False
    token_estimate: int = 0
    error: str | None = None
    from_cache: bool = False


# ─────────────────────── Agriculture analysis (LLM output) ───────────────────────
#
# What the LLM returns. Roles from the taxonomy are referenced by numeric id
# (never by name), roles the taxonomy doesn't cover go in `new_roles` with a
# short key ("N1", "N2", ...), and every product points at one of those via
# `role`. AgriAnalysis's validator enforces all of that - a failure raises,
# which makes structured_output.generate_structured send the exact error back
# to the model for a corrected answer.

MIN_PRODUCTS = 10
MAX_PRODUCTS = 50

# Models sometimes copy the legend's code into a new role name, e.g.
# "Rural Finance Provider (SP)", or hedge with "(hypothetical)".
_ROLE_NAME_NOISE_RE = re.compile(
    r"\s*\((?:M|3PM|EI|D|R|B|F|SP|G|EI\+D|new|hypothetical|inferred|proposed)\)", re.I
)
# Placeholder product names weaker models emit instead of real ones,
# e.g. "Herbicide XYZ", "Insecticide ABC", "Product 1", "Brand X".
_PLACEHOLDER_RE = re.compile(
    r"\b(?:XYZ|ABC|DEF|PQR|Lorem|Ipsum|TBD|N/?A)\b|\b(?:product|brand|item|model|variety)\s+(?:[A-Z]|\d{1,2})$",
    re.I,
)
# Words too generic to identify a company ("Agro", "Industries", ...) - the
# rest of the name is compared against a "new entity" to catch the model
# naming the entity after the company itself.
_GENERIC_NAME_WORDS = {
    "the", "and", "of", "co", "company", "ltd", "limited", "pvt", "private", "inc", "llc", "llp",
    "plc", "corp", "corporation", "group", "india", "international", "industries", "industry",
    "agro", "agri", "agriculture", "agricultural", "farm", "farms", "seeds", "seed", "fertilizers",
    "fertiliser", "fertilizer", "chemicals", "enterprises", "traders", "trading", "agencies",
    "solutions", "services", "products", "global", "sciences", "lifesciences", "foods", "food",
    "gmbh", "srl", "spa", "sarl", "ltda", "kgaa", "imp", "exp", "export", "import", "exports", "imports",
}


def _name_tokens(name: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", name.lower()) if len(t) > 2 and t not in _GENERIC_NAME_WORDS}


class RoleMatch(BaseModel):
    role_id: int = Field(description="Id of a role from the ROLE LIST.")
    business_types: list[str] = Field(
        default_factory=list,
        description="Only the business type option(s) that actually apply, from that role's code (e.g. ['Importer']).",
    )
    evidence: str = Field(default="", description="Short evidence for this role, max ~15 words.")

    @field_validator("role_id", mode="before")
    @classmethod
    def _plain_id(cls, v):
        # "R12" / "12" / 12 -> 12
        if isinstance(v, str):
            v = v.strip().lstrip("Rr#")
        return v


class NewRole(BaseModel):
    key: str = Field(description="Short key 'N1', 'N2', ... that products use to reference this role.")
    group: str = Field(description="A listed GROUP name, or a new group name only if none fits.")
    entity: str = Field(description="A listed entity name if the role fits under one, otherwise a new entity name.")
    role: str = Field(description="New role name in the same style as the ROLE LIST.")
    business_types: list[str] = Field(default_factory=list, description="Applicable business type option(s).")
    reason: str = Field(default="", description="Why no listed role fits, max ~20 words.")


class ProductItem(BaseModel):
    name: str = Field(description="Specific product or service name, in English.")
    kind: Literal["product", "service"]
    role: str = Field(description="role_id (as a string, e.g. '12') or a new role key (e.g. 'N1') it belongs to.")
    seen: bool = Field(description="True only if this exact product/service is named in the provided source text.")

    @field_validator("kind", mode="before")
    @classmethod
    def _normalize_kind(cls, v):
        # "Product", "SERVICE", "services", "solution" ... -> product | service
        return "service" if "serv" in str(v).lower() else "product"

    @field_validator("role", mode="before")
    @classmethod
    def _role_as_str(cls, v):
        return str(v).strip()


class AgriAnalysis(BaseModel):
    """Field order matters: `reasoning` comes first so the model commits to
    its analysis before choosing roles (chain-of-thought ordering)."""

    reasoning: str = Field(description="Brief step-by-step analysis: what the company does, evidence, which roles fit.")
    description: str = Field(description="2-3 sentence plain description of what the company does.")
    is_agriculture: bool = Field(description="True if the company's business is related to agriculture.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the roles chosen, 0 to 1.")
    roles: list[RoleMatch] = Field(default_factory=list)
    new_roles: list[NewRole] = Field(default_factory=list)
    products: list[ProductItem] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _join_text_lists(cls, data):
        # Smaller models sometimes return reasoning/description as a list of
        # sentences - harmless, so join instead of burning a retry on it.
        if isinstance(data, dict):
            for key in ("reasoning", "description"):
                if isinstance(data.get(key), list):
                    data[key] = " ".join(str(x) for x in data[key])
        return data

    @model_validator(mode="after")
    def _check_against_taxonomy(self, info: ValidationInfo) -> "AgriAnalysis":
        """Every corrective retry costs a whole extra call (the full prompt
        plus the bad answer, re-sent), so:

        - Mechanical mistakes are REPAIRED silently: unknown role ids,
          products pointing at a role the company wasn't given, roles with
          no product behind them, placeholder product names, a "new entity"
          that is just the company's own name.
        - Judgment problems are SOFT errors (missing business type, fewer
          than MIN_PRODUCTS products): they trigger a retry, but on the
          final attempt (context {"final_attempt": True}) the answer is kept
          as-is rather than throwing the whole company away.
        - Only an answer with no usable role or no product at all is a HARD
          error that can fail the company.

        Validation context (optional): {"company_name": str, "final_attempt": bool}.
        """
        tax = get_taxonomy()
        context = info.context or {}
        final_attempt = bool(context.get("final_attempt"))
        company_tokens = _name_tokens(context.get("company_name") or "")
        by_id = tax.by_id
        soft: list[str] = []
        hard: list[str] = []

        # Listed roles: ids must exist; keep only allowed business-type options.
        merged: dict[int, RoleMatch] = {}
        for rm in self.roles:
            role = by_id.get(rm.role_id)
            if role is None:
                continue
            allowed = {o.lower(): o for o in role.business_types_allowed}
            chosen = list(dict.fromkeys(
                allowed[b.strip().lower()] for b in rm.business_types if b.strip().lower() in allowed
            ))
            if not chosen and len(role.business_types_allowed) == 1:
                chosen = list(role.business_types_allowed)
            if not chosen:
                # "Seed Distributor" can only be a Distributor - when exactly
                # one allowed option appears in the role's own name, that's
                # the answer; only genuinely ambiguous names ("Fertilizer
                # Distributor / Wholesaler") need the model's judgment.
                in_name = [o for o in role.business_types_allowed if re.search(rf"\b{re.escape(o)}\b", role.name, re.I)]
                if len(in_name) == 1:
                    chosen = in_name
            if not chosen and role.business_types_allowed:
                soft.append(
                    f"role_id {role.id} needs business_types: the option(s) that actually apply, "
                    f"from {list(role.business_types_allowed)}"
                )
            elif len(role.business_types_allowed) >= 3 and len(chosen) == len(role.business_types_allowed):
                soft.append(
                    f"role_id {role.id}: do not list every option {list(role.business_types_allowed)} - "
                    "write only the one(s) this company actually is (usually one)"
                )
            if role.id in merged:
                prev = merged[role.id]
                chosen = list(dict.fromkeys([*prev.business_types, *chosen]))
            merged[role.id] = RoleMatch(role_id=role.id, business_types=chosen, evidence=rm.evidence)
        self.roles = list(merged.values())

        # New roles: a "new" role that is really a listed one is folded back
        # into `roles`.
        all_options = {o.lower(): o for r in tax.roles for o in r.business_types_allowed}
        kept_new: list[NewRole] = []
        aliases: dict[str, str] = {}
        for nr in self.new_roles:
            nr.role = _ROLE_NAME_NOISE_RE.sub("", nr.role).strip()
            nr.entity = _ROLE_NAME_NOISE_RE.sub("", nr.entity).strip()
            # Same role name (under any entity - role names are specific
            # enough), or an "entity" that is actually a listed role's name
            # (e.g. entity "Development Finance Institution" = role 235).
            listed = next(
                (r for r in tax.roles if r.name.lower() in (nr.role.lower(), nr.entity.lower())),
                None,
            )
            if listed is not None:
                aliases[nr.key] = str(listed.id)
                if listed.id not in merged:
                    allowed = {o.lower(): o for o in listed.business_types_allowed}
                    chosen = [allowed[b.strip().lower()] for b in nr.business_types if b.strip().lower() in allowed]
                    if not chosen and len(listed.business_types_allowed) == 1:
                        chosen = list(listed.business_types_allowed)
                    merged[listed.id] = RoleMatch(role_id=listed.id, business_types=chosen, evidence=nr.reason)
                    self.roles.append(merged[listed.id])
                continue
            nr.business_types = [all_options[b.strip().lower()] for b in nr.business_types if b.strip().lower() in all_options]
            entity = tax.canonical_entity(nr.entity)
            if entity is not None:
                nr.entity = entity
                nr.group = tax.group_of_entity(entity) or nr.group
            else:
                nr.group = tax.canonical_group(nr.group) or nr.group.strip()
                # An entity carrying ALL of the company's distinctive name
                # words ("Tata Steel" for Tata Steel, "Rasi Hybrid Seeds" for
                # Rasi Seeds) is the company, not a sector - fall back to the
                # role name. Merely sharing a word ("Steel manufacturing",
                # "Information Technology") is fine.
                if company_tokens and company_tokens <= _name_tokens(nr.entity):
                    nr.entity = nr.role
            kept_new.append(nr)
        self.new_roles = kept_new

        # Products: every one must point at a role this company has.
        valid_refs = {str(rm.role_id) for rm in self.roles} | {nr.key for nr in self.new_roles}
        # Products sometimes name their role ("Seed Distributor") instead of
        # giving its id/key - map those back rather than dropping them.
        ref_by_name = {by_id[rm.role_id].name.lower(): str(rm.role_id) for rm in self.roles}
        ref_by_name.update({nr.role.lower(): nr.key for nr in self.new_roles})
        seen_names: set[str] = set()
        products: list[ProductItem] = []
        for p in self.products:
            ref = aliases.get(p.role, p.role)
            if ref not in valid_refs:
                ref = ref_by_name.get(ref.lower(), ref)
            key = p.name.strip().lower()
            if ref not in valid_refs or not key or key in seen_names or _PLACEHOLDER_RE.search(p.name):
                continue
            seen_names.add(key)
            products.append(ProductItem(name=p.name.strip(), kind=p.kind, role=ref, seen=p.seen))
        self.products = products[:MAX_PRODUCTS]

        # A role with no product/service behind it is an unsupported guess.
        used_refs = {p.role for p in self.products}
        self.roles = [rm for rm in self.roles if str(rm.role_id) in used_refs]
        self.new_roles = [nr for nr in self.new_roles if nr.key in used_refs]

        if not self.products or (not self.roles and not self.new_roles):
            hard.append(
                "no usable answer: give at least one role (from the ROLE LIST, or in new_roles - "
                "non-agriculture businesses go under group 'Non-agriculture') and at least "
                f"{MIN_PRODUCTS} products/services whose `role` is one of those roles"
            )
        elif len(self.products) < MIN_PRODUCTS:
            soft.append(
                f"only {len(self.products)} distinct products/services whose `role` is one of this "
                f"company's roles {sorted(valid_refs)}; at least {MIN_PRODUCTS} are required "
                "(add the most typical offerings for those roles)"
            )

        errors = hard + ([] if final_attempt else soft)
        if errors:
            raise ValueError("; ".join(errors[:6]))
        return self


# ─────────────────────── Resolved result (stored / exported) ───────────────────────

class ResolvedRole(BaseModel):
    ref: str  # role id as string, or new-role key
    role_id: int | None  # None for new (unlisted) roles
    group: str
    entity: str
    role: str
    business_types: list[str]
    is_new_role: bool
    is_new_entity: bool
    note: str = ""  # evidence (listed) / reason (new)


class ResolvedProduct(BaseModel):
    name: str
    kind: str  # product | service
    source: str  # website | web_search | inferred
    role_ref: str


class AgriProfile(BaseModel):
    description: str
    is_agriculture: bool
    confidence: float
    roles: list[ResolvedRole]
    products: list[ResolvedProduct]

    def role_by_ref(self) -> dict[str, ResolvedRole]:
        return {r.ref: r for r in self.roles}

    @property
    def has_new(self) -> bool:
        return any(r.is_new_role for r in self.roles)

    def summary_label(self) -> str:
        """Short one-line label for the companies table, e.g.
        'Seeds & biologicals: Seed Distributor, Seed Importer / Exporter · Fertilizers: ...'."""
        by_entity: dict[str, list[str]] = {}
        for r in self.roles:
            by_entity.setdefault(r.entity, []).append(r.role + (" (NEW)" if r.is_new_role else ""))
        return " · ".join(f"{e}: {', '.join(rs)}" for e, rs in by_entity.items())[:500]


# ─────────────────────────── Pipeline result ───────────────────────────

class CompanyResult(BaseModel):
    company_id: str
    native_id: str | None = None
    company_name: str
    location: str | None = None
    url: str | None = None

    url_read: UrlReadResult
    profile: AgriProfile | None = None

    success: bool = True
    error: str | None = None
    processing_time_ms: int = 0


# ─────────────────────────── SSE events ───────────────────────────

class SSEEventType(str, Enum):
    PROGRESS = "progress"
    COMPANY_DONE = "company_done"
    COMPANY_FAILED = "company_failed"
    STATUS_CHANGE = "status_change"
    LOG = "log"
    COMPLETE = "complete"
    ERROR = "error"


class SSEEvent(BaseModel):
    event: SSEEventType
    job_id: str
    data: dict
    seq: int = 0
