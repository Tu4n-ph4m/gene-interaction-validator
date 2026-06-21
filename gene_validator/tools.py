"""Tool implementations that hit StringDB and BioGRID directly (no LLM involved).

Single-pair functions are decorated with @beta_tool so the Anthropic tool
runner can generate its schema from the signature + docstring, for the
interactive agentic loop (gene_validator.agent). They return JSON strings
rather than raising on a "not found" result, so the model can reason about a
negative result instead of the loop crashing.

The bulk_* functions at the bottom are plain functions (no LLM involved) used
by the deterministic network-validation path (gene_validator.batch) for
large gene lists, where calling the single-pair tools once per pair would be
O(N^2) API calls -- StringDB and BioGRID both support querying a whole gene
list at once and getting back only the edges that exist among them.
"""

from __future__ import annotations

import json
import os

import requests
from anthropic import beta_tool

STRING_API_BASE = "https://string-db.org/api"
BIOGRID_API_BASE = "https://webservice.thebiogrid.org"
HPA_API_BASE = "https://www.proteinatlas.org/api"
HTTP_TIMEOUT = 15
BIOGRID_MAX_PER_PAGE = 10000


def _gene_matches_biogrid_side(gene: str, official_symbol: str | None, synonyms: str | None) -> bool:
    """True if `gene` is the official symbol or a listed synonym for one side of a BioGRID record."""
    gene_upper = gene.upper()
    if gene_upper == (official_symbol or "").upper():
        return True
    return gene_upper in (synonyms or "").upper().split("|")


def _biogrid_record_matches_pair(rec: dict, gene1: str, gene2: str) -> bool:
    """True if a BioGRID record is genuinely an edge between gene1 and gene2 --
    not a self-interaction or a record only involving one of them. Matches
    against BioGRID's official symbol AND its listed synonyms on each side,
    since `searchNames=true` can match an input alias that isn't the
    official symbol BioGRID reports back."""
    a_official, a_syn = rec.get("OFFICIAL_SYMBOL_A"), rec.get("SYNONYMS_A")
    b_official, b_syn = rec.get("OFFICIAL_SYMBOL_B"), rec.get("SYNONYMS_B")
    return (
        _gene_matches_biogrid_side(gene1, a_official, a_syn)
        and _gene_matches_biogrid_side(gene2, b_official, b_syn)
    ) or (
        _gene_matches_biogrid_side(gene1, b_official, b_syn)
        and _gene_matches_biogrid_side(gene2, a_official, a_syn)
    )


def _biogrid_key() -> str:
    key = os.environ.get("BIOGRID_ACCESS_KEY")
    if not key:
        raise RuntimeError(
            "BIOGRID_ACCESS_KEY is not set. Get a free key at "
            "https://webservice.thebiogrid.org/ and put it in your .env file."
        )
    return key


@beta_tool
def resolve_gene(gene_symbol: str, species_tax_id: int = 9606) -> str:
    """Resolve and normalize a gene symbol/name against the STRING database.

    Use this first to confirm a gene identifier is valid and to get its
    canonical STRING ID before checking interactions. Catches typos and
    unrecognized gene names before they reach the interaction-lookup tools.

    Args:
        gene_symbol: Gene symbol, alias, or name to resolve (e.g. "TP53").
        species_tax_id: NCBI taxonomy ID for the species (default 9606 = human).
    """
    resp = requests.get(
        f"{STRING_API_BASE}/json/get_string_ids",
        params={
            "identifiers": gene_symbol,
            "species": species_tax_id,
            "limit": 1,
        },
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    matches = resp.json()

    if not matches:
        return json.dumps(
            {
                "input": gene_symbol,
                "resolved": False,
                "reason": "No match found in STRING for this identifier/species.",
            }
        )

    match = matches[0]
    return json.dumps(
        {
            "input": gene_symbol,
            "resolved": True,
            "string_id": match.get("stringId"),
            "preferred_name": match.get("preferredName"),
            "taxon_id": match.get("ncbiTaxonId"),
            "annotation": match.get("annotation"),
        }
    )


@beta_tool
def check_string_interaction(
    gene1: str, gene2: str, species_tax_id: int = 9606
) -> str:
    """Check StringDB for a known protein-protein interaction between two genes.

    Returns the STRING combined confidence score (0-1) and the individual
    evidence channel scores (experimental, database, textmining, etc.) if an
    interaction edge exists between the two genes.

    Args:
        gene1: First gene symbol (e.g. "BRCA1").
        gene2: Second gene symbol (e.g. "BRCA2").
        species_tax_id: NCBI taxonomy ID for the species (default 9606 = human).
    """
    resp = requests.get(
        f"{STRING_API_BASE}/json/network",
        params={
            "identifiers": f"{gene1}\r{gene2}",
            "species": species_tax_id,
        },
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    edges = resp.json()

    matching = [
        e
        for e in edges
        if {e.get("preferredName_A"), e.get("preferredName_B")} == {gene1, gene2}
    ]

    if not matching:
        return json.dumps(
            {
                "source": "StringDB",
                "gene1": gene1,
                "gene2": gene2,
                "interaction_found": False,
            }
        )

    edge = matching[0]
    return json.dumps(
        {
            "source": "StringDB",
            "gene1": gene1,
            "gene2": gene2,
            "interaction_found": True,
            "combined_score": edge.get("score"),
            "evidence_scores": {
                "neighborhood": edge.get("nscore"),
                "fusion": edge.get("fscore"),
                "phylogenetic_cooccurrence": edge.get("pscore"),
                "coexpression": edge.get("ascore"),
                "experimental": edge.get("escore"),
                "database": edge.get("dscore"),
                "textmining": edge.get("tscore"),
            },
        }
    )


@beta_tool
def check_biogrid_interaction(
    gene1: str, gene2: str, species_tax_id: int = 9606
) -> str:
    """Check BioGRID for genetic/physical interaction evidence between two genes.

    Returns the experimental system(s) used (e.g. "Two-hybrid",
    "Affinity Capture-MS") and supporting PubMed IDs for each reported
    interaction record between the two genes.

    Args:
        gene1: First gene symbol (e.g. "BRCA1").
        gene2: Second gene symbol (e.g. "BRCA2").
        species_tax_id: NCBI taxonomy ID for the species (default 9606 = human).
    """
    resp = requests.get(
        f"{BIOGRID_API_BASE}/interactions/",
        params={
            "searchNames": "true",
            "geneList": f"{gene1}|{gene2}",
            "taxId": species_tax_id,
            "includeInteractors": "false",
            "format": "json",
            "accesskey": _biogrid_key(),
        },
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    records = resp.json()

    # BioGRID's geneList+includeInteractors=false can still return records
    # involving only ONE of the two genes (e.g. a gene1-gene1 self-interaction,
    # or gene1 paired with something else matched by name search). Filter to
    # records that are genuinely an edge between gene1 and gene2.
    true_pairs = [
        rec for rec in records.values() if _biogrid_record_matches_pair(rec, gene1, gene2)
    ]

    if not true_pairs:
        return json.dumps(
            {
                "source": "BioGRID",
                "gene1": gene1,
                "gene2": gene2,
                "interaction_found": False,
            }
        )

    evidence = [
        {
            "experimental_system": rec.get("EXPERIMENTAL_SYSTEM"),
            "experimental_system_type": rec.get("EXPERIMENTAL_SYSTEM_TYPE"),
            "pubmed_id": rec.get("PUBMED_ID"),
            "throughput": rec.get("THROUGHPUT"),
        }
        for rec in true_pairs
    ]

    return json.dumps(
        {
            "source": "BioGRID",
            "gene1": gene1,
            "gene2": gene2,
            "interaction_found": True,
            "evidence_count": len(evidence),
            "evidence": evidence,
        }
    )


@beta_tool
def check_tissue_expression(gene: str, tissue: str) -> str:
    """Check whether a gene's RNA is expressed in a specific tissue/cell type, via the Human Protein Atlas (HPA).

    Use this to add cell-type/tissue context to an interaction result: two
    genes can have strong StringDB/BioGRID evidence yet be biologically
    irrelevant in a given tissue if one of them isn't expressed there.

    Caveat: HPA's lightweight lookup only reports tissues where a gene's RNA
    is *tissue-enriched* (elevated relative to other tissues). A broadly
    expressed gene (HPA category "Detected in all/many") may still be
    expressed in the queried tissue at a normal level even if that tissue
    doesn't show up with an explicit value here. Treat a "Not detected"
    result as a confident negative; treat everything else as informative
    but not exhaustive.

    Args:
        gene: Gene symbol (e.g. "KLK3").
        tissue: Tissue or cell-type name as used by HPA (e.g. "prostate",
            "liver", "brain", "skin"). Lowercase, singular tissue names work
            best.
    """
    resp = requests.get(
        f"{HPA_API_BASE}/search_download.php",
        params={
            "search": gene,
            "format": "json",
            "columns": "g,eg,rnatd,rnatsm",
            "compress": "no",
        },
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    records = resp.json()

    match = next(
        (r for r in records if r.get("Gene", "").upper() == gene.upper()), None
    )
    if not match:
        return json.dumps(
            {
                "source": "Human Protein Atlas",
                "gene": gene,
                "tissue": tissue,
                "found_in_hpa": False,
            }
        )

    distribution = match.get("RNA tissue distribution")
    enriched = match.get("RNA tissue specific nTPM") or {}
    tissue_key = next(
        (k for k in enriched if k.lower() == tissue.lower()), None
    )

    if distribution == "Not detected":
        likely_expressed = False
        note = "Gene RNA not detected in any tissue per HPA consensus data."
    elif tissue_key:
        likely_expressed = True
        note = f"Tissue-enriched expression confirmed (nTPM={enriched[tissue_key]})."
    elif distribution == "Detected in all":
        likely_expressed = True
        note = (
            "Broadly expressed across all tissues per HPA category; exact nTPM "
            "for this specific tissue isn't returned by this lookup, but "
            "expression here is expected."
        )
    else:
        likely_expressed = None
        note = (
            f"Gene RNA distribution is '{distribution}' and this tissue is not "
            "in its list of tissue-enriched tissues. It may still be expressed "
            "at a baseline level not captured by this lookup -- inconclusive."
        )

    return json.dumps(
        {
            "source": "Human Protein Atlas",
            "gene": gene,
            "tissue": tissue,
            "found_in_hpa": True,
            "rna_tissue_distribution": distribution,
            "tissue_enriched_nTPM": enriched,
            "likely_expressed_in_tissue": likely_expressed,
            "note": note,
        }
    )


ALL_TOOLS = [
    resolve_gene,
    check_string_interaction,
    check_biogrid_interaction,
    check_tissue_expression,
]


def bulk_resolve_genes(genes: list[str], species_tax_id: int = 9606) -> dict[str, dict]:
    """Resolve many gene symbols against STRING in a single request.

    Returns a dict mapping each *input* gene string to its resolution info
    (stringId, preferredName, ...), or omits the key entirely if STRING
    couldn't resolve it -- check `gene in result` rather than truthiness.
    """
    if not genes:
        return {}

    resp = requests.post(
        f"{STRING_API_BASE}/json/get_string_ids",
        data={
            "identifiers": "\r".join(genes),
            "species": species_tax_id,
        },
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    matches = resp.json()

    resolved: dict[str, dict] = {}
    for match in matches:
        query_item = match.get("queryItem")
        if query_item is not None:
            resolved[query_item] = match
    return resolved


def bulk_string_network(genes: list[str], species_tax_id: int = 9606) -> list[dict]:
    """Query StringDB once for the full network among `genes`.

    Returns only the edges that actually exist between members of the list
    -- StringDB does the filtering server-side, so this is one request
    regardless of list size (within StringDB's own limits; intended for
    up to a few thousand genes per call).
    """
    if len(genes) < 2:
        return []

    resp = requests.post(
        f"{STRING_API_BASE}/json/network",
        data={
            "identifiers": "\r".join(genes),
            "species": species_tax_id,
        },
        timeout=HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def _chunk(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def bulk_biogrid_network(
    genes: list[str], species_tax_id: int = 9606, chunk_size: int = 250
) -> list[dict]:
    """Query BioGRID for the full network among `genes`.

    BioGRID's geneList parameter has practical URL-length limits, so genes
    are split into chunks; to still catch interactions *between* two
    different chunks (not just within one), every pair of chunks (including
    a chunk with itself) is queried with includeInteractors=false, which
    restricts results to interactions where both genes are in the queried
    set. For N genes and chunk size C this is ~ (N/C)*(N/C+1)/2 requests --
    e.g. 10 requests for 1000 genes at chunk_size=250, instead of ~500,000
    pairwise requests.
    """
    if len(genes) < 2:
        return []

    key = _biogrid_key()
    chunks = _chunk(genes, chunk_size)
    seen_interaction_ids: set[str] = set()
    records: list[dict] = []

    for i in range(len(chunks)):
        for j in range(i, len(chunks)):
            gene_set = chunks[i] if i == j else chunks[i] + chunks[j]
            start = 0
            while True:
                resp = requests.get(
                    f"{BIOGRID_API_BASE}/interactions/",
                    params={
                        "searchNames": "true",
                        "geneList": "|".join(gene_set),
                        "taxId": species_tax_id,
                        "includeInteractors": "false",
                        "format": "json",
                        "accesskey": key,
                        "max": BIOGRID_MAX_PER_PAGE,
                        "start": start,
                    },
                    timeout=HTTP_TIMEOUT,
                )
                resp.raise_for_status()
                page = resp.json()
                if not page:
                    break
                for interaction_id, rec in page.items():
                    if interaction_id in seen_interaction_ids:
                        continue
                    seen_interaction_ids.add(interaction_id)
                    records.append(rec)
                if len(page) < BIOGRID_MAX_PER_PAGE:
                    break
                start += BIOGRID_MAX_PER_PAGE

    return records
