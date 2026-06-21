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

Follow-up requests (e.g. "now check those in liver", "just the StringDB-only \
ones", "what about EGFR too") refer back to the genes from the conversation \
so far. If you can identify the exact gene list from earlier in this \
conversation, call the tool again with that list (plus whatever changed -- \
new tissue, added gene, etc.) rather than answering from memory of the \
earlier summary, since that summary may not have included every gene or \
pair. If you cannot identify the exact gene list (e.g. it was never fully \
stated, or this is a fresh conversation with no prior gene-list turn), ask \
the user to restate it -- do not guess and do not say you'll do something \
without doing it.

CRITICAL: never produce a reply like "let me check that" or "I'll run the \
full panel" as your final answer. Either call the tool in this same turn, \
or ask a clarifying question. A reply that promises an action but takes \
none leaves the user with nothing -- that is worse than asking a question.

After the tool returns, write a brief, conversational summary (a few \
sentences, not a report) -- mention the most notable pairs, whether sources \
agree, and any caveats (genes that didn't resolve, tissue-expression flags, \
or curated-database overlap risk) only if they're notable. The full results \
table is shown separately in the UI, so don't try to enumerate every pair \
in your reply -- highlight what's interesting and let the table carry the \
detail. If the user asked for a specific subset (e.g. "StringDB-only \
pairs" -- meaning string_interaction_found is true and \
biogrid_interaction_found is false), filter to exactly that subset in your \
summary; don't substitute a different ranking (like "sorted by score") for \
what was actually asked.
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
) -> Tuple[str, Optional[Dict[str, Any]], str]:
    """Run one chat turn.

    `history` is the prior turns as plain {"role": "user"|"assistant",
    "content": str} dicts. Each assistant entry should be the *history_content*
    returned by a previous call (not necessarily the same as what was shown
    to the user) -- see the third return value below.

    Returns (reply_text, results_payload, history_content).
      - reply_text: what to show the user in the chat UI.
      - results_payload: None if no tool call happened this turn, else
        {"results": [...], "invalid_genes": [...]}.
      - history_content: what the caller should store as this turn's
        assistant content for future chat_turn() calls. When a tool was
        called, this is reply_text plus a hidden note recording the exact
        genes/tissue/species used, so follow-ups ("just the StringDB-only
        ones") can be answered by re-querying the same gene set instead of
        the model trying to recall it from its own prior summary (which may
        not have enumerated every gene, especially for large panels).
    """
    client = anthropic.Anthropic()
    messages: List[Dict[str, Any]] = [
        {"role": h["role"], "content": h["content"]} for h in history
    ]
    messages.append({"role": "user", "content": message})

    results_payload: Optional[Dict[str, Any]] = None
    last_query: Optional[Dict[str, Any]] = None

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
            history_content = text
            if last_query is not None:
                history_content += (
                    f"\n\n[context (not shown to user): last queried genes = "
                    f"{', '.join(last_query['genes'])}; tissue="
                    f"{last_query.get('tissue') or 'none'}; species="
                    f"{last_query.get('species') or 'human'}]"
                )
            return text, results_payload, history_content

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            is_error = False
            if block.name == "find_gene_interactions":
                try:
                    results, invalid_genes = _run_find_interactions(**block.input)
                except Exception as exc:  # upstream API hiccup -- let the model recover, don't crash the turn
                    is_error = True
                    tool_result_str = (
                        f"Lookup failed due to an upstream error: {exc}. "
                        "Tell the user briefly and suggest trying again."
                    )
                else:
                    results_payload = {
                        "results": [asdict(r) for r in results],
                        "invalid_genes": invalid_genes,
                    }
                    last_query = dict(block.input)
                    tool_result_str = _summarize_for_model(results, invalid_genes)
            else:
                is_error = True
                tool_result_str = json.dumps({"error": f"unknown tool {block.name}"})
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": tool_result_str,
                    "is_error": is_error,
                }
            )
        messages.append({"role": "user", "content": tool_results})

    fallback = "I'm having trouble completing that request -- try rephrasing or narrowing the gene list."
    return fallback, results_payload, fallback
