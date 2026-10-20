"""
One LLM call per company: classify it against the agriculture taxonomy
(entities + roles + the business type options that actually apply), discover
roles the taxonomy doesn't cover, and list 10-50 of its products/services -
each tied to one of those roles.

Replaces the old 3-call classify -> research -> generate chain. One call
means the website text and instructions are sent once instead of three
times, which roughly halves tokens/requests per company - the deciding
factor on free-tier rate limits at 6-lakh-company scale.

Prompt layout is deliberate: everything static (rules + the ~2.7K-token
taxonomy + schema) is ONE byte-identical system message, and only the
company-specific part changes in the user message. That's what lets Groq's
automatic prompt caching (cached tokens don't count toward rate limits) and
Mistral's prompt_cache_key reuse the prefix across companies.
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache

from litellm import Router
from pydantic import BaseModel, Field

from backend.core.llm_router import AGRI_MODEL
from backend.core.models import (
    MAX_PRODUCTS,
    MIN_PRODUCTS,
    AgriAnalysis,
    AgriProfile,
    ResolvedProduct,
    ResolvedRole,
    UrlReadResult,
    UrlReadSource,
)
from backend.core.structured_output import generate_structured, schema_hint
from backend.core.taxonomy import get_taxonomy

logger = logging.getLogger("agri_analyzer")

NON_AGRI_GROUP = "Non-agriculture"

SYSTEM_PROMPT = """You are an agriculture industry analyst. For ONE company you must \
(1) find every entity/role it fills from the ROLE LIST below and (2) list its products \
and services.

SOURCE OF TRUTH: Use the provided source text (company website or web search result) \
as the main evidence. If there is no source text, use the company name, location and \
your own knowledge, and lower `confidence` accordingly. Never invent facts about a \
specific company that the evidence doesn't support.

ROLES:
- Pick EVERY listed role the company clearly fills - many companies have several, \
across different entities (e.g. a seed producer that also distributes and gives crop \
advice). Answer with role ids only, in `roles`.
- Only add a role with DIRECT evidence that the company does that activity as a \
business. Partnerships, R&D, mission statements or "could also" are NOT evidence. \
When unsure, leave the role out.
- business_types: write ONLY the option(s) the company actually is, from that role's \
code in the legend - usually exactly ONE. Examples: a seed importer that does not export \
-> ["Importer"]; a fertilizer wholesaler -> ["Wholesaler"]; a dealer selling to \
retailers -> ["Distributor"]; a company that both imports and exports -> ["Exporter", \
"Importer"]. If the evidence doesn't say which, pick the single most likely one. Never \
list every option. Codes with no business type -> [].

NEW ROLES (discovery):
- If the company does something agriculture-related that NO listed role covers, add it \
to `new_roles` (key "N1", "N2", ...) instead of forcing a wrong listed role.
- Put it under a listed entity when one fits (use that entity's exact name and group); \
only create a new entity when no listed entity fits. A new entity is a generic \
industry sector name (e.g. "Mushroom farming", "Steel manufacturing") - NEVER a \
company, subsidiary, division or brand name (wrong: "Astec LifeSciences", "Tata \
Steel"; right: "Agrochemical R&D"). Name roles in the same style as the list (e.g. "Mushroom \
Spawn Producer"), and give a short reason.
- Always prefer a listed role when one genuinely fits - check EVERY group first \
(e.g. banks and development finance are under "Banks & DFIs", farms and hatcheries \
under the Production / Livestock groups). Never put a listed role or entity name in \
new_roles. Write new role names plainly, without codes or notes in brackets.
- The company's MAIN business must always be covered by a role. If the main business \
is not agriculture (e.g. a steel maker, a bank for all sectors, an IT firm), set \
is_agriculture=false and describe that main business as a new role under group \
"Non-agriculture" - even if it also has a small agriculture line, which then gets its \
own (listed) role as well. is_agriculture=true only when agriculture is a core part \
of the business, not a side line.

PRODUCTS AND SERVICES:
- Include EVERY distinct product/service named in the source text (seen=true), up to \
{max_products}. Services count (e.g. "Soil testing service", "Crop loans").
- Only if that gives fewer than {min_products}, add the most typical offerings for the \
company's roles (seen=false) - just enough to reach {min_products}, no more. Never pad \
the list with generic filler (events, podcasts, loyalty programs, hotlines, training \
programs) unless the source names them.
- Do not invent model numbers, variety codes or brand names that are not in the source. \
One clear name per item - no "e.g." lists inside a name.
- Each product's `role` must be one of THIS company's roles: a role id from `roles` \
written as a string (e.g. "4"), or a new role key (e.g. "N1"). Tie each product to \
the role it actually comes from (a poultry breed belongs to a hatchery/breeding role, \
not a vaccine role). If a real product doesn't fit any role you chose, add the \
matching role instead of forcing it under a wrong one. Every role must have at least \
one product/service tied to it - a role with nothing to show for it should not be \
listed.
- Names must be in English (keep brand names, model numbers and units as they are). \
No near-duplicates.

{taxonomy}

Respond with ONE minified JSON object on a single line - no line breaks or \
indentation, no prose, no markdown fences - matching exactly this shape:
{schema}"""


@lru_cache
def _system_prompt() -> str:
    return SYSTEM_PROMPT.format(
        min_products=MIN_PRODUCTS,
        max_products=MAX_PRODUCTS,
        taxonomy=get_taxonomy().render_for_prompt(),
        schema=schema_hint(AgriAnalysis),
    )


_SOURCE_LABEL = {
    UrlReadSource.DIRECT_FETCH: "company website",
    UrlReadSource.JINA_READER: "company website",
    UrlReadSource.COMPOUND_BETA: "company website (read by a browsing assistant)",
    UrlReadSource.WEB_SEARCH: "web search result for this company",
}


def _user_prompt(company_name: str, location: str | None, url: str | None, url_read: UrlReadResult) -> str:
    parts = [
        f"Company Name: {company_name}",
        f"Location: {location or 'Unknown'}",
        f"Website: {url or 'none given'}",
    ]
    if url_read.success and url_read.markdown:
        label = _SOURCE_LABEL.get(url_read.source, "source text")
        parts.append(f"\nSOURCE TEXT ({label}):\n---\n{url_read.markdown}\n---")
    else:
        parts.append(
            "\nNo source text is available "
            f"(reason: {url_read.error or 'no website given'}). "
            "Use the company name and location only, and be conservative: unless you "
            "genuinely know this company, a small local business whose name only names "
            "goods (e.g. 'X Fertilizers and Pesticides', 'Y Seeds', 'Z Agro Agencies') is "
            "usually a dealer/distributor/retailer, NOT a manufacturer - choose manufacturer "
            "roles only if the name says so (e.g. 'Industries', 'Manufacturing', "
            "'Chemicals Ltd', 'Mills'). Keep confidence at 0.5 or below."
        )
    parts.append("\nAnalyze this company now.")
    return "\n".join(parts)


def _resolve(analysis: AgriAnalysis, url_read: UrlReadResult) -> AgriProfile:
    tax = get_taxonomy()
    by_id = tax.by_id
    roles: list[ResolvedRole] = []
    for rm in analysis.roles:
        r = by_id[rm.role_id]
        roles.append(
            ResolvedRole(
                ref=str(r.id), role_id=r.id, group=r.group, entity=r.entity, role=r.name,
                business_types=rm.business_types, is_new_role=False, is_new_entity=False, note=rm.evidence,
            )
        )
    for nr in analysis.new_roles:
        is_new_entity = tax.canonical_entity(nr.entity) is None
        group = nr.group
        if not analysis.is_agriculture and is_new_entity and tax.canonical_group(group) is None:
            group = NON_AGRI_GROUP
        roles.append(
            ResolvedRole(
                ref=nr.key, role_id=None, group=group, entity=nr.entity.strip(), role=nr.role.strip(),
                business_types=nr.business_types, is_new_role=True, is_new_entity=is_new_entity, note=nr.reason,
            )
        )

    has_source = url_read.success and bool(url_read.markdown)
    seen_label = "web_search" if url_read.source == UrlReadSource.WEB_SEARCH else "website"
    products = [
        ResolvedProduct(
            name=p.name, kind=p.kind, role_ref=p.role,
            source=seen_label if (p.seen and has_source) else "inferred",
        )
        for p in analysis.products
    ]
    return AgriProfile(
        description=analysis.description.strip(),
        is_agriculture=analysis.is_agriculture,
        confidence=analysis.confidence,
        roles=roles,
        products=products,
    )


async def analyze_company(
    router: Router,
    company_name: str,
    location: str | None,
    url: str | None,
    url_read: UrlReadResult,
) -> AgriProfile:
    analysis = await generate_structured(
        router,
        AGRI_MODEL,
        messages=[
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": _user_prompt(company_name, location, url, url_read)},
        ],
        schema_cls=AgriAnalysis,
        validation_context={"company_name": company_name},
        temperature=0.2,
        # ~2K tokens of JSON for 50 products + roles, plus headroom for
        # reasoning models (gpt-oss/qwen count hidden reasoning here too).
        max_tokens=4000,
    )
    profile = _resolve(analysis, url_read)
    profile.products = await _translate_non_english_names(router, profile.products)
    return profile


# ─────────── Non-English product name safety net (rare; only runs when needed) ───────────

_NON_ASCII_RE = re.compile(r"[^\x00-\x7F]")
# Typographic punctuation models love to emit (non-breaking hyphens, curly
# quotes, en/em dashes) - normalized first so they don't trigger a needless
# translation call for an already-English name.
_PUNCT_MAP = str.maketrans({
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
    "‘": "'", "’": "'", "“": '"', "”": '"', " ": " ",
    "™": "", "®": "", "©": "",
})

_TRANSLATE_SYSTEM_PROMPT = """You are an agricultural/industrial translator. Translate \
each numbered product or service name into clear, natural, industry-standard English. \
Keep brand names, model numbers, and units unchanged. Return exactly one translation per \
input line, in the same order, with nothing else.

Respond with a single JSON object ONLY (no prose, no markdown fences), matching exactly \
this shape:
{schema}"""


class _TranslationBatch(BaseModel):
    translations: list[str] = Field(
        description="English translation for each input name, one-to-one, in the exact same order as given."
    )


async def _translate_non_english_names(router: Router, products: list[ResolvedProduct]) -> list[ResolvedProduct]:
    """The model is told to answer in English, but occasionally copies a
    source-language name verbatim. One extra call fixes just those names;
    the common all-English case never pays for it."""
    products = [p.model_copy(update={"name": p.name.translate(_PUNCT_MAP).strip()}) for p in products]
    flagged = [(i, p.name) for i, p in enumerate(products) if _NON_ASCII_RE.search(p.name)]
    if not flagged:
        return products
    try:
        batch = await generate_structured(
            router,
            AGRI_MODEL,
            messages=[
                {"role": "system", "content": _TRANSLATE_SYSTEM_PROMPT.format(schema=schema_hint(_TranslationBatch))},
                {"role": "user", "content": "\n".join(f"{n}. {name}" for n, (_, name) in enumerate(flagged, start=1))},
            ],
            schema_cls=_TranslationBatch,
            temperature=0.0,
            max_tokens=1500,
        )
        if len(batch.translations) != len(flagged):
            logger.warning("Translation size mismatch (%d vs %d) - keeping originals", len(flagged), len(batch.translations))
            return products
        for (idx, _), translated in zip(flagged, batch.translations):
            if translated.strip():
                products[idx] = products[idx].model_copy(update={"name": translated.strip()})
    except Exception:
        logger.warning("Product name translation failed - keeping originals", exc_info=True)

    seen: set[str] = set()
    unique: list[ResolvedProduct] = []
    for p in products:
        if p.name.lower() not in seen:
            seen.add(p.name.lower())
            unique.append(p)
    return unique
