"""Batch validation of gene-gene pairs from a CSV file.

Unlike agent.validate_gene_pair (which runs an LLM agentic loop with prose
output for one pair), this calls the StringDB/BioGRID tool functions
directly and computes a deterministic verdict per pair -- appropriate for
many pairs where you want a structured table, not narration.

Input CSV must have columns: gene1,gene2  (optional: species_tax_id,tissue)
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import asdict, dataclass

import requests

from gene_validator.species import resolve_species
from gene_validator.tools import (
    HTTP_TIMEOUT,
    bulk_biogrid_network,
    bulk_resolve_genes,
    bulk_string_network,
    check_biogrid_interaction,
    check_string_interaction,
    check_tissue_expression,
    resolve_gene,
)


def read_source(path_or_url: str) -> str:
    """Read text content from a local file path or an http(s) URL.

    Lets --batch/--gene-file point at a hosted file (Google Sheets export
    link, GitHub raw URL, S3, etc.) instead of requiring a local download.
    """
    if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
        resp = requests.get(path_or_url, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        return resp.text
    with open(path_or_url) as f:
        return f.read()


MAX_PUBMED_IDS_SHOWN = 15

STRING_EVIDENCE_LABELS = {
    "neighborhood": "neighborhood",
    "fusion": "gene_fusion",
    "phylogenetic_cooccurrence": "phylogenetic_cooccurrence",
    "coexpression": "coexpression",
    "experimental": "experimental",
    "database": "curated_database",
    "textmining": "textmining",
}

# Raw STRING network-endpoint field name -> our internal evidence_scores key
STRING_RAW_EVIDENCE_FIELDS = {
    "nscore": "neighborhood",
    "fscore": "fusion",
    "pscore": "phylogenetic_cooccurrence",
    "ascore": "coexpression",
    "escore": "experimental",
    "dscore": "database",
    "tscore": "textmining",
}


@dataclass
class PairResult:
    gene1: str
    gene2: str
    gene1_valid: bool
    gene2_valid: bool
    string_interaction_found: bool | None
    string_combined_score: float | None
    string_evidence_breakdown: str
    string_source_url: str
    string_curated_overlap_risk: str
    biogrid_interaction_found: bool | None
    biogrid_evidence_count: int | None
    biogrid_experimental_systems: str
    biogrid_pubmed_ids: str
    biogrid_source_url: str
    tissue: str
    gene1_tissue_expression: str
    gene2_tissue_expression: str
    verdict: str
    notes: str


def _resolve(gene: str, species_tax_id: int) -> bool:
    result = json.loads(resolve_gene.func(gene, species_tax_id))
    return bool(result.get("resolved"))


def _string_network_url(gene1: str, gene2: str, species_tax_id: int) -> str:
    # %0d = carriage return, STRING's separator for multiple identifiers
    return (
        f"https://string-db.org/cgi/network?identifiers={gene1}%0d{gene2}"
        f"&species={species_tax_id}"
    )


def _biogrid_search_url(gene1: str, gene2: str) -> str:
    return f"https://thebiogrid.org/search.php?search={gene1}%20{gene2}"


def _format_string_evidence(evidence_scores: dict) -> str:
    # Only show channels that actually contributed (score > 0), highest first,
    # so the breakdown tells you *which kind* of evidence drove the combined score.
    contributing = [
        (STRING_EVIDENCE_LABELS.get(key, key), score)
        for key, score in evidence_scores.items()
        if score
    ]
    contributing.sort(key=lambda kv: kv[1], reverse=True)
    if not contributing:
        return "none"
    return "; ".join(f"{label}={score:.3f}" for label, score in contributing)


def _curated_overlap_risk(evidence_scores: dict) -> str:
    """Flag how much of STRING's combined score comes from its "database"
    channel -- curated pathway/complex databases (KEGG, Reactome, BioCyc,
    GO Complexes) that STRING aggregates *alongside* the same pairwise
    interaction databases (BioGRID, IntAct, MINT) feeding its "experiments"
    channel. A high database-channel score means STRING and BioGRID
    agreement here is not necessarily two independent lines of evidence --
    STRING may be partly reflecting curated records that originated
    elsewhere, including BioGRID itself.
    """
    database_score = evidence_scores.get("database", 0) or 0
    if database_score >= 0.5:
        return "high"
    if database_score > 0:
        return "low"
    return "none"


def _format_biogrid_evidence(evidence: list[dict]) -> tuple[str, str]:
    systems = sorted({e.get("experimental_system") for e in evidence if e.get("experimental_system")})
    pubmed_ids = sorted({str(e.get("pubmed_id")) for e in evidence if e.get("pubmed_id")})

    systems_str = "; ".join(systems) if systems else "none"

    if len(pubmed_ids) > MAX_PUBMED_IDS_SHOWN:
        shown = pubmed_ids[:MAX_PUBMED_IDS_SHOWN]
        pubmed_str = "; ".join(shown) + f"; +{len(pubmed_ids) - MAX_PUBMED_IDS_SHOWN} more"
    else:
        pubmed_str = "; ".join(pubmed_ids) if pubmed_ids else "none"

    return systems_str, pubmed_str


def _format_tissue_expression(gene: str, tissue: str) -> tuple[str, bool | None]:
    result = json.loads(check_tissue_expression.func(gene, tissue))
    if not result.get("found_in_hpa"):
        return "not_found_in_hpa", None

    likely = result.get("likely_expressed_in_tissue")
    enriched = result.get("tissue_enriched_nTPM") or {}
    tissue_key = next((k for k in enriched if k.lower() == tissue.lower()), None)

    if likely is True and tissue_key:
        return f"expressed (nTPM={enriched[tissue_key]})", True
    if likely is True:
        return "expressed (broadly_detected)", True
    if likely is False:
        return "not_detected", False
    return f"inconclusive ({result.get('rna_tissue_distribution')})", None


def validate_pair(
    gene1: str,
    gene2: str,
    species_tax_id: int = 9606,
    tissue: str | None = None,
) -> PairResult:
    gene1_valid = _resolve(gene1, species_tax_id)
    gene2_valid = _resolve(gene2, species_tax_id)

    if not (gene1_valid and gene2_valid):
        invalid = [g for g, ok in [(gene1, gene1_valid), (gene2, gene2_valid)] if not ok]
        return PairResult(
            gene1=gene1,
            gene2=gene2,
            gene1_valid=gene1_valid,
            gene2_valid=gene2_valid,
            string_interaction_found=None,
            string_combined_score=None,
            string_evidence_breakdown="not_checked",
            string_source_url="",
            string_curated_overlap_risk="not_checked",
            biogrid_interaction_found=None,
            biogrid_evidence_count=None,
            biogrid_experimental_systems="not_checked",
            biogrid_pubmed_ids="not_checked",
            biogrid_source_url="",
            tissue=tissue or "",
            gene1_tissue_expression="not_checked",
            gene2_tissue_expression="not_checked",
            verdict="invalid_gene",
            notes=f"Could not resolve: {', '.join(invalid)}. Skipped interaction checks.",
        )

    string_result = json.loads(check_string_interaction.func(gene1, gene2, species_tax_id))
    biogrid_result = json.loads(check_biogrid_interaction.func(gene1, gene2, species_tax_id))

    string_found = string_result.get("interaction_found", False)
    biogrid_found = biogrid_result.get("interaction_found", False)

    string_evidence_scores = string_result.get("evidence_scores", {})
    string_evidence_breakdown = (
        _format_string_evidence(string_evidence_scores) if string_found else "no_edge_in_string"
    )
    string_curated_overlap_risk = (
        _curated_overlap_risk(string_evidence_scores) if string_found else "not_applicable"
    )
    biogrid_systems, biogrid_pubmed_ids = (
        _format_biogrid_evidence(biogrid_result.get("evidence", []))
        if biogrid_found
        else ("no_records_in_biogrid", "no_records_in_biogrid")
    )

    if string_found and biogrid_found:
        verdict = "concordant_positive"
        notes = "Both StringDB and BioGRID report evidence of interaction."
        if string_curated_overlap_risk == "high":
            notes += (
                " CAVEAT: StringDB's score here is driven mainly by its curated-database "
                "channel, which can itself be sourced from BioGRID/IntAct/KEGG/etc. -- this "
                "agreement may not be fully independent corroboration."
            )
    elif (not string_found) and (not biogrid_found):
        verdict = "concordant_negative"
        notes = "Neither database reports evidence. Absence of evidence is not proof of no interaction."
    else:
        verdict = "discordant"
        source = "StringDB" if string_found else "BioGRID"
        notes = f"Only {source} reports evidence; the other source does not."

    gene1_tissue_expression = "not_checked"
    gene2_tissue_expression = "not_checked"
    if tissue:
        gene1_tissue_expression, gene1_expressed = _format_tissue_expression(gene1, tissue)
        gene2_tissue_expression, gene2_expressed = _format_tissue_expression(gene2, tissue)
        if (string_found or biogrid_found) and (gene1_expressed is False or gene2_expressed is False):
            not_expressed = [
                g for g, e in [(gene1, gene1_expressed), (gene2, gene2_expressed)] if e is False
            ]
            notes += (
                f" CAVEAT: interaction evidence exists, but {', '.join(not_expressed)} "
                f"is not detected in '{tissue}' per HPA -- biological relevance in this "
                "tissue is questionable."
            )

    return PairResult(
        gene1=gene1,
        gene2=gene2,
        gene1_valid=gene1_valid,
        gene2_valid=gene2_valid,
        string_interaction_found=string_found,
        string_combined_score=string_result.get("combined_score"),
        string_evidence_breakdown=string_evidence_breakdown,
        string_source_url=_string_network_url(gene1, gene2, species_tax_id),
        string_curated_overlap_risk=string_curated_overlap_risk,
        biogrid_interaction_found=biogrid_found,
        biogrid_evidence_count=biogrid_result.get("evidence_count"),
        biogrid_experimental_systems=biogrid_systems,
        biogrid_pubmed_ids=biogrid_pubmed_ids,
        biogrid_source_url=_biogrid_search_url(gene1, gene2),
        tissue=tissue or "",
        gene1_tissue_expression=gene1_tissue_expression,
        gene2_tissue_expression=gene2_tissue_expression,
        verdict=verdict,
        notes=notes,
    )


def validate_pairs_from_csv(
    input_path_or_url: str, default_tissue: str | None = None
) -> list[PairResult]:
    """Validate every gene1,gene2 row in `input_path_or_url`.

    Accepts a local file path or an http(s) URL (see `read_source`).
    Each row may include an optional `tissue` column that overrides
    `default_tissue` for that row, and an optional `species_tax_id` column
    that accepts either a common name ("mouse") or a raw NCBI taxonomy ID.
    """
    results = []
    content = read_source(input_path_or_url)
    for row in csv.DictReader(io.StringIO(content)):
        gene1, gene2 = row["gene1"].strip(), row["gene2"].strip()
        species_tax_id = resolve_species(row.get("species_tax_id"))
        tissue = (row.get("tissue") or "").strip() or default_tissue
        results.append(validate_pair(gene1, gene2, species_tax_id, tissue))
    return results


def _string_evidence_from_edge(edge: dict) -> dict:
    return {
        internal_key: edge.get(raw_key, 0)
        for raw_key, internal_key in STRING_RAW_EVIDENCE_FIELDS.items()
    }


def _biogrid_evidence_from_records(records: list[dict]) -> list[dict]:
    return [
        {
            "experimental_system": rec.get("EXPERIMENTAL_SYSTEM"),
            "pubmed_id": rec.get("PUBMED_ID"),
        }
        for rec in records
    ]


def validate_gene_network(
    genes: list[str],
    species_tax_id: int = 9606,
    tissue: str | None = None,
) -> tuple[list[PairResult], list[str]]:
    """Find all existing StringDB/BioGRID interactions within a gene list.

    Unlike validate_pair (one pair at a time), this bulk-queries both
    databases for the whole list at once -- O(1) StringDB calls and
    O((N/chunk)^2) BioGRID calls, not O(N^2) -- so it scales to hundreds or
    thousands of genes. Only pairs where at least one database reports
    evidence are returned (a full matrix including non-interacting pairs
    would be mostly-empty noise at this scale: e.g. 1000 genes is ~500,000
    possible pairs).

    Tissue expression (if `tissue` is given) is only looked up for genes
    that actually appear in a returned pair, not the whole input list --
    cheaper, and irrelevant for genes with no interaction to caveat.

    Returns (results, invalid_genes) where invalid_genes are inputs that
    STRING could not resolve at all.
    """
    deduped = list(dict.fromkeys(g.strip() for g in genes if g.strip()))
    if len(deduped) < 2:
        return [], []

    resolved = bulk_resolve_genes(deduped, species_tax_id)
    valid_genes = [g for g in deduped if g in resolved]
    invalid_genes = [g for g in deduped if g not in resolved]

    if len(valid_genes) < 2:
        return [], invalid_genes

    valid_upper_to_original = {g.upper(): g for g in valid_genes}

    string_pairs: dict[frozenset, dict] = {}
    for edge in bulk_string_network(valid_genes, species_tax_id):
        a, b = edge.get("preferredName_A"), edge.get("preferredName_B")
        if not a or not b or a.upper() == b.upper():
            continue
        string_pairs[frozenset({a.upper(), b.upper()})] = edge

    biogrid_pairs: dict[frozenset, list[dict]] = {}
    for rec in bulk_biogrid_network(valid_genes, species_tax_id):
        a, b = rec.get("OFFICIAL_SYMBOL_A"), rec.get("OFFICIAL_SYMBOL_B")
        if not a or not b or a.upper() == b.upper():
            continue
        biogrid_pairs.setdefault(frozenset({a.upper(), b.upper()}), []).append(rec)

    tissue_cache: dict[str, tuple[str, bool | None]] = {}

    def cached_tissue(gene: str) -> tuple[str, bool | None]:
        if gene not in tissue_cache:
            tissue_cache[gene] = _format_tissue_expression(gene, tissue)
        return tissue_cache[gene]

    results = []
    for pair_key in set(string_pairs) | set(biogrid_pairs):
        a_upper, b_upper = sorted(pair_key)
        gene1 = valid_upper_to_original.get(a_upper, a_upper)
        gene2 = valid_upper_to_original.get(b_upper, b_upper)

        string_edge = string_pairs.get(pair_key)
        biogrid_records = biogrid_pairs.get(pair_key, [])
        string_found = string_edge is not None
        biogrid_found = bool(biogrid_records)

        string_evidence_scores = (
            _string_evidence_from_edge(string_edge) if string_found else {}
        )
        string_evidence_breakdown = (
            _format_string_evidence(string_evidence_scores) if string_found else "no_edge_in_string"
        )
        string_curated_overlap_risk = (
            _curated_overlap_risk(string_evidence_scores) if string_found else "not_applicable"
        )
        biogrid_evidence = _biogrid_evidence_from_records(biogrid_records)
        biogrid_systems, biogrid_pubmed_ids = (
            _format_biogrid_evidence(biogrid_evidence)
            if biogrid_found
            else ("no_records_in_biogrid", "no_records_in_biogrid")
        )

        verdict = "concordant_positive" if (string_found and biogrid_found) else "discordant"
        if verdict == "concordant_positive":
            notes = "Both StringDB and BioGRID report evidence of interaction."
            if string_curated_overlap_risk == "high":
                notes += (
                    " CAVEAT: StringDB's score here is driven mainly by its curated-database "
                    "channel, which can itself be sourced from BioGRID/IntAct/KEGG/etc. -- this "
                    "agreement may not be fully independent corroboration."
                )
        else:
            notes = f"Only {'StringDB' if string_found else 'BioGRID'} reports evidence; the other source does not."

        gene1_tissue_expression = "not_checked"
        gene2_tissue_expression = "not_checked"
        if tissue:
            gene1_tissue_expression, gene1_expressed = cached_tissue(gene1)
            gene2_tissue_expression, gene2_expressed = cached_tissue(gene2)
            if gene1_expressed is False or gene2_expressed is False:
                not_expressed = [
                    g for g, e in [(gene1, gene1_expressed), (gene2, gene2_expressed)] if e is False
                ]
                notes += (
                    f" CAVEAT: interaction evidence exists, but {', '.join(not_expressed)} "
                    f"is not detected in '{tissue}' per HPA -- biological relevance in this "
                    "tissue is questionable."
                )

        results.append(
            PairResult(
                gene1=gene1,
                gene2=gene2,
                gene1_valid=True,
                gene2_valid=True,
                string_interaction_found=string_found,
                string_combined_score=string_edge.get("score") if string_edge else None,
                string_evidence_breakdown=string_evidence_breakdown,
                string_source_url=_string_network_url(gene1, gene2, species_tax_id),
                string_curated_overlap_risk=string_curated_overlap_risk,
                biogrid_interaction_found=biogrid_found,
                biogrid_evidence_count=len(biogrid_records) if biogrid_found else None,
                biogrid_experimental_systems=biogrid_systems,
                biogrid_pubmed_ids=biogrid_pubmed_ids,
                biogrid_source_url=_biogrid_search_url(gene1, gene2),
                tissue=tissue or "",
                gene1_tissue_expression=gene1_tissue_expression,
                gene2_tissue_expression=gene2_tissue_expression,
                verdict=verdict,
                notes=notes,
            )
        )

    return results, invalid_genes


def write_results_csv(results: list[PairResult], output_path: str) -> None:
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()))
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))


def write_results_json(results: list[PairResult], output_path: str) -> None:
    with open(output_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
