"""Conversational interface to the gene interaction network finder.

A small agentic loop: Claude has one tool, find_gene_interactions, and
extracts the gene list / tissue / species from whatever the user typed in
plain English, then writes a short conversational reply. The full
StringDB/BioGRID result table (which can be large) never goes into the
model's context -- only a capped summary does. The caller gets the full
table back separately for rendering in the UI.

History is kept as plain {role, content} text turns (no tool_use/
tool_result blocks persisted across turns) -- each turn's tool round-trip
is internal to that turn. This is a deliberate simplification: enough for
natural follow-ups ("now check those in brain tissue") since gene names
typically appear in the assistant's own prior reply, without the
complexity of replaying full tool-call history every turn.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple

import anthropic
from dotenv import load_dotenv

from gene_validator.batch import validate_gene_network
from gene_validator.species import resolve_species

load_dotenv()

MODEL = os.environ.get("GENE_VALIDATOR_MODEL", "claude-opus-4-8")
MAX_AGENT_ITERATIONS = 6
MAX_TOP_PAIRS_IN_SUMMARY = 12

SYSTEM_PROMPT = """\
You are a friendly research assistant that helps people explore gene-gene \
interaction networks via StringDB and BioGRID, optionally scoped to a \
tissue/cell type via the Human Protein Atlas.

You have one tool, find_gene_interactions(genes, tissue, species). Call it \
whenever the user names a set of genes (2 or more) they want checked for \
interactions -- extract the gene symbols, tissue/cell type (if mentioned), \
and species (if mentioned, otherwise default to human) from their message, \
even if phrased casually ("do BRCA1, BRCA2 and TP53 interact in liver?").

If the user names fewer than 2 genes, or hasn't named any genes yet, ask a \
short clarifying question instead of calling the tool.

After the tool returns, write a brief, conversational summary (a few \
sentences, not a report) -- mention the most notable pairs, whether sources \
agree, and any caveats (genes that didn't resolve, tissue-expression flags, \
or curated-database overlap risk) only if they're notable. The full results \
table is shown separately in the UI, so don't try to enumerate every pair \
in your reply -- highlight what's interesting and let the table carry the \
detail.
"""

FIND_INTERACTIONS_TOOL = {
    "name": "find_gene_interactions",
    "description": (
        "Find all existing interactions (StringDB + BioGRID) among a list of "
        "genes, optionally scoped to a tissue/cell type."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "genes": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Gene symbols to check, e.g. ['BRCA1', 'BRCA2', 'TP53']",
            },
            "tissue": {
                "type": "string",
                "description": "Tissue or cell type to check expression context for, if mentioned.",
            },
            "species": {
                "type": "string",
                "description": "Species name (human, mouse, etc.) or NCBI taxonomy ID. Default human.",
            },
        },
        "required": ["genes"],
    },
}


def _run_find_interactions(
    genes: List[str], tissue: Optional[str] = None, species: str = "human"
) -> Tuple[list, list]:
    species_tax_id = resolve_species(species)
    return validate_gene_network(genes, species_tax_id, tissue)


def _summarize_for_model(results: list, invalid_genes: list) -> str:
    top_pairs = sorted(
        results, key=lambda r: r.string_combined_score or 0, reverse=True
    )[:MAX_TOP_PAIRS_IN_SUMMARY]

    summary = {
        "total_pairs_found": len(results),
        "concordant_positive_count": sum(1 for r in results if r.verdict == "concordant_positive"),
        "discordant_count": sum(1 for r in results if r.verdict == "discordant"),
        "pairs_with_caveat": [
            {"gene1": r.gene1, "gene2": r.gene2, "note": r.notes}
            for r in results
            if "CAVEAT" in r.notes
        ][:MAX_TOP_PAIRS_IN_SUMMARY],
        "invalid_genes": invalid_genes,
        "top_pairs_by_string_score": [
            {
                "gene1": r.gene1,
                "gene2": r.gene2,
                "verdict": r.verdict,
                "string_score": r.string_combined_score,
                "biogrid_evidence_count": r.biogrid_evidence_count,
            }
            for r in top_pairs
        ],
    }
    return json.dumps(summary)


def chat_turn(
    message: str, history: Iterable[Dict[str, str]]
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Run one chat turn.

    `history` is the prior turns as plain {"role": "user"|"assistant",
    "content": str} dicts (display text only).

    Returns (reply_text, results_payload). results_payload is None if no
    tool call happened this turn, else {"results": [...], "invalid_genes": [...]}.
    """
    client = anthropic.Anthropic()
    messages: List[Dict[str, Any]] = [
        {"role": h["role"], "content": h["content"]} for h in history
    ]
    messages.append({"role": "user", "content": message})

    results_payload: Optional[Dict[str, Any]] = None

    for _ in range(MAX_AGENT_ITERATIONS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=[FIND_INTERACTIONS_TOOL],
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            text = "".join(b.text for b in response.content if b.type == "text")
            return text, results_payload

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            if block.name == "find_gene_interactions":
                results, invalid_genes = _run_find_interactions(**block.input)
                results_payload = {
                    "results": [asdict(r) for r in results],
                    "invalid_genes": invalid_genes,
                }
                tool_result_str = _summarize_for_model(results, invalid_genes)
            else:
                tool_result_str = json.dumps({"error": f"unknown tool {block.name}"})
            tool_results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": tool_result_str}
            )
        messages.append({"role": "user", "content": tool_results})

    return (
        "I'm having trouble completing that request -- try rephrasing or narrowing the gene list.",
        results_payload,
    )
