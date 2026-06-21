"""Conversational interface to the gene interaction network finder.

Two explicit agents:

  Planner agent  -- talks to the user, asks clarifying questions, never
                    touches data directly. When it has enough information,
                    it delegates the lookup to the executor via the
                    delegate_lookup tool, then explains the executor's
                    factual report back to the user conversationally.

  Executor agent -- has the only access to find_gene_interactions. Given a
                    delegated task (genes/tissue/species), it calls the
                    tool and reports back factually (not conversationally)
                    -- its report is consumed by the planner, not the user.

The full StringDB/BioGRID result table (which can be large) never goes
into either model's context -- only a capped summary does, computed in
Python. The caller gets the full table back separately for rendering in
the UI.

History is kept as plain {role, content} text turns for the planner only
(no tool_use/tool_result blocks persisted across turns -- each turn's
planner<->executor round-trip is internal to that turn). This is a
deliberate simplification: enough for natural follow-ups ("now check those
in brain tissue") via a hidden context note (see chat_turn's docstring),
without the complexity of replaying full tool-call history every turn.
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
MAX_EXECUTOR_ITERATIONS = 3
MAX_TOP_PAIRS_IN_SUMMARY = 12
# Generous: a tool_use call for a large gene list (e.g. 1000 genes) needs
# several thousand output tokens just to write out the JSON array argument.
# Too low a limit truncates the call mid-argument (stop_reason="max_tokens"),
# silently producing an empty/partial input instead of an error.
MAX_TOKENS = 8192

TOO_LARGE_MESSAGE = (
    "That gene list is too large for me to process in one request -- "
    "try splitting it into smaller batches."
)

# --- Planner agent -----------------------------------------------------

PLANNER_SYSTEM_PROMPT = """\
You are a friendly research assistant that helps people explore gene-gene \
interaction networks via StringDB and BioGRID, optionally scoped to a \
tissue/cell type via the Human Protein Atlas.

You don't have direct access to any data yourself. A separate execution \
agent does the actual lookups. Your job is purely conversational:

1. Understand what the user wants. If they've named 2+ genes, extract the \
gene symbols, tissue/cell type (if mentioned), and species (if mentioned, \
otherwise default to human) -- even if phrased casually ("do BRCA1, BRCA2 \
and TP53 interact in liver?").
2. If they've named fewer than 2 genes, or it's not clear what they want \
yet, ask a short clarifying question instead of delegating.
3. Once you have enough information, call delegate_lookup(genes, tissue, \
species) to hand the task to the execution agent.
4. When the execution agent reports back, write a brief, conversational \
reply (a few sentences, not a report) for the user based on that report -- \
mention the most notable pairs, whether sources agree, and any caveats \
only if they're notable. The full results table is shown separately in \
the UI, so don't enumerate every pair -- highlight what's interesting and \
let the table carry the detail. If the user asked for a specific subset \
(e.g. "StringDB-only pairs"), make sure your reply reflects exactly that \
subset, using the report's data -- don't substitute a different ranking.

Follow-up requests (e.g. "now check those in liver", "just the StringDB-only \
ones", "what about EGFR too") refer back to the genes from the conversation \
so far. If you can identify the exact gene list from earlier in this \
conversation, delegate again with that list (plus whatever changed) rather \
than answering from memory of an earlier summary, which may not have \
enumerated every gene. If you cannot identify the exact gene list, ask the \
user to restate it -- do not guess.

CRITICAL: never produce a reply like "let me check that" or "I'll run the \
full panel" as your final answer. Either call delegate_lookup in this same \
turn, or ask a clarifying question. A reply that promises an action but \
takes none leaves the user with nothing -- that is worse than asking a \
question.
"""

DELEGATE_TOOL = {
    "name": "delegate_lookup",
    "description": (
        "Hand off a gene-interaction lookup to the execution agent, which "
        "will query StringDB/BioGRID and report back factually."
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

# --- Executor agent ------------------------------------------------------

EXECUTOR_SYSTEM_PROMPT = """\
You are an execution agent for gene-interaction lookups. You have one tool, \
find_gene_interactions(genes, tissue, species). You will be given a task \
naming the exact genes/tissue/species to look up -- call the tool with \
those exact parameters, then report the results back factually: pair \
counts, notable high/low-confidence pairs, and any caveats (unresolved \
genes, tissue-expression flags, curated-database overlap risk). Your report \
goes to another agent, not the end user, so be precise and data-dense \
rather than conversational. If the tool call fails, report the failure \
plainly so the planner agent can relay it.
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


def _run_executor_agent(
    genes: List[str], tissue: Optional[str], species: Optional[str]
) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Agent 2: the only one that actually calls find_gene_interactions.

    Returns (factual_report, results_payload). results_payload is None if
    the tool was never successfully called (e.g. hit the iteration limit).
    """
    client = anthropic.Anthropic()
    task = (
        f"Look up interactions for genes={genes!r}, tissue={tissue!r}, "
        f"species={species!r}."
    )
    messages: List[Dict[str, Any]] = [{"role": "user", "content": task}]
    results_payload: Optional[Dict[str, Any]] = None

    for _ in range(MAX_EXECUTOR_ITERATIONS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=EXECUTOR_SYSTEM_PROMPT,
            tools=[FIND_INTERACTIONS_TOOL],
            messages=messages,
        )

        if response.stop_reason == "max_tokens":
            return TOO_LARGE_MESSAGE, results_payload

        if response.stop_reason != "tool_use":
            text = "".join(b.text for b in response.content if b.type == "text")
            return text, results_payload

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            is_error = False
            if block.name == "find_gene_interactions":
                try:
                    results, invalid_genes = _run_find_interactions(**block.input)
                except Exception as exc:  # upstream API hiccup -- recover, don't crash the turn
                    is_error = True
                    tool_result_str = (
                        f"Lookup failed due to an upstream error: {exc}. "
                        "Report this failure plainly."
                    )
                else:
                    results_payload = {
                        "results": [asdict(r) for r in results],
                        "invalid_genes": invalid_genes,
                    }
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

    return "Execution agent could not complete the lookup in time.", results_payload


def chat_turn(
    message: str, history: Iterable[Dict[str, str]]
) -> Tuple[str, Optional[Dict[str, Any]], str]:
    """Run one chat turn through the planner agent, which may delegate to
    the executor agent.

    `history` is the prior turns as plain {"role": "user"|"assistant",
    "content": str} dicts -- planner-facing only. Each assistant entry
    should be the *history_content* returned by a previous call (not
    necessarily the same as what was shown to the user) -- see the third
    return value below.

    Returns (reply_text, results_payload, history_content).
      - reply_text: what to show the user in the chat UI.
      - results_payload: None if no lookup happened this turn, else
        {"results": [...], "invalid_genes": [...]}.
      - history_content: what the caller should store as this turn's
        assistant content for future chat_turn() calls. When a lookup was
        delegated, this is reply_text plus a hidden note recording the
        exact genes/tissue/species used, so follow-ups ("just the
        StringDB-only ones") can be answered by re-delegating with the same
        gene set instead of the planner guessing from prose memory.
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
            max_tokens=MAX_TOKENS,
            system=PLANNER_SYSTEM_PROMPT,
            tools=[DELEGATE_TOOL],
            messages=messages,
        )

        if response.stop_reason == "max_tokens":
            return TOO_LARGE_MESSAGE, results_payload, (
                "(a request was truncated for being too large; ask the user to split the gene list)"
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
            if block.name == "delegate_lookup":
                genes = block.input.get("genes", [])
                tissue = block.input.get("tissue")
                species = block.input.get("species")
                executor_report, executor_results = _run_executor_agent(genes, tissue, species)
                if executor_results is not None:
                    results_payload = executor_results
                    last_query = {"genes": genes, "tissue": tissue, "species": species}
                tool_result_str = executor_report
                is_error = executor_results is None
            else:
                tool_result_str = json.dumps({"error": f"unknown tool {block.name}"})
                is_error = True
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
