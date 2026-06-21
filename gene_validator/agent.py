"""Agentic loop that validates a gene-gene pair against StringDB and BioGRID.

validate_gene_pair() uses the Anthropic Python SDK's beta tool runner: the
model decides which tool to call, the SDK executes our Python functions and
feeds results back, and the loop ends when the model produces a final
answer. It only exposes the final text -- good enough for the CLI.

validate_gene_pair_stream() is a manual agentic loop (same tools, same
model) that yields a structured event per step -- tool_call, tool_result,
text -- so a UI (e.g. the Streamlit app) can show live progress instead of
just the final answer. Built with a manual loop instead of the tool runner
because the runner doesn't surface intermediate tool *results* to the
caller, only the assistant's messages.
"""

from __future__ import annotations

import os
from typing import Any, Iterator

import anthropic
from dotenv import load_dotenv

from gene_validator.tools import ALL_TOOLS

load_dotenv()

MODEL = os.environ.get("GENE_VALIDATOR_MODEL", "claude-opus-4-8")
MAX_AGENT_ITERATIONS = 10

SYSTEM_PROMPT = """\
You are a gene-gene interaction validation agent. Given a pair of gene \
symbols, you must:

1. Resolve each gene symbol against STRING (resolve_gene) to confirm it is a \
valid, recognized gene identifier for the given species. If a gene fails to \
resolve, stop and report it as invalid — do not guess a correction yourself.
2. Check StringDB for a known interaction (check_string_interaction) and \
report the combined confidence score and evidence channels if found.
3. Check BioGRID for interaction evidence (check_biogrid_interaction) and \
report the experimental system(s) and supporting PubMed IDs if found.
4. Cross-validate: state clearly whether StringDB and BioGRID agree, \
disagree, or only one source has evidence. Note that a "no evidence" result \
from one database does not necessarily mean there is no interaction — \
explain this caveat when relevant.
5. If the user specifies a tissue or cell type, call check_tissue_expression \
for each gene against that tissue. A strong interaction score means little \
biologically if one of the genes isn't expressed in the relevant cell type — \
flag this explicitly rather than only reporting the StringDB/BioGRID result. \
Be clear about the tool's limitation: it confidently flags non-expression, \
but a missing tissue-specific value does not prove non-expression.

Always call all relevant tools (gene validation x2, both interaction checks, \
and tissue expression checks if a tissue/cell type was given) before giving \
your final answer. Finish with a concise structured verdict: gene validity, \
StringDB result, BioGRID result, tissue expression context (if requested), \
and overall agreement/confidence assessment.
"""


def _build_user_request(gene1: str, gene2: str, species_tax_id: int, tissue: str | None) -> str:
    request = (
        f"Validate the interaction between gene '{gene1}' and "
        f"gene '{gene2}' (species taxonomy ID {species_tax_id})."
    )
    if tissue:
        request += (
            f" Also check whether each gene is expressed in '{tissue}' tissue "
            "and factor that into your assessment."
        )
    return request


def validate_gene_pair_stream(
    gene1: str, gene2: str, species_tax_id: int = 9606, tissue: str | None = None
) -> Iterator[dict[str, Any]]:
    """Run the agentic validation loop, yielding one event per step.

    Event shapes:
      {"type": "tool_call", "name": str, "input": dict}
      {"type": "tool_result", "name": str, "output": str}
      {"type": "text", "text": str}

    Manual loop (not the tool runner) so each tool's actual result is
    visible to the caller as it happens, not just the assistant's messages.
    """
    client = anthropic.Anthropic()
    tools_by_name = {t.name: t for t in ALL_TOOLS}
    raw_tools = [t.to_dict() for t in ALL_TOOLS]

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": _build_user_request(gene1, gene2, species_tax_id, tissue)}
    ]

    for _ in range(MAX_AGENT_ITERATIONS):
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=raw_tools,
            messages=messages,
        )

        for block in response.content:
            if block.type == "text" and block.text:
                yield {"type": "text", "text": block.text}

        if response.stop_reason != "tool_use":
            return

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            yield {"type": "tool_call", "name": block.name, "input": block.input}
            result = tools_by_name[block.name].func(**block.input)
            yield {"type": "tool_result", "name": block.name, "output": result}
            tool_results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": result}
            )
        messages.append({"role": "user", "content": tool_results})

    yield {"type": "text", "text": "(Stopped after reaching the max agent iteration limit.)"}


def validate_gene_pair(
    gene1: str, gene2: str, species_tax_id: int = 9606, tissue: str | None = None
) -> str:
    """Run the agentic validation loop for a single gene pair and return the
    model's final text response. Pass `tissue` to add cell-type/tissue
    expression context (e.g. "prostate", "liver")."""
    final_text = ""
    for event in validate_gene_pair_stream(gene1, gene2, species_tax_id, tissue):
        if event["type"] == "text":
            final_text = event["text"]
    return final_text
