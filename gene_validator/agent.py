"""Agentic loop that validates a gene-gene pair against StringDB and BioGRID.

Uses the Anthropic Python SDK's beta tool runner: the model decides which
tool to call (resolve_gene / check_string_interaction / check_biogrid_interaction),
the SDK executes our Python functions and feeds results back, and the loop
ends when the model produces a final answer.
"""

from __future__ import annotations

import os

import anthropic
from dotenv import load_dotenv

from gene_validator.tools import ALL_TOOLS

load_dotenv()

MODEL = os.environ.get("GENE_VALIDATOR_MODEL", "claude-opus-4-8")

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


def validate_gene_pair(
    gene1: str, gene2: str, species_tax_id: int = 9606, tissue: str | None = None
) -> str:
    """Run the agentic validation loop for a single gene pair and return the
    model's final text response. Pass `tissue` to add cell-type/tissue
    expression context (e.g. "prostate", "liver")."""
    client = anthropic.Anthropic()

    user_request = (
        f"Validate the interaction between gene '{gene1}' and "
        f"gene '{gene2}' (species taxonomy ID {species_tax_id})."
    )
    if tissue:
        user_request += (
            f" Also check whether each gene is expressed in '{tissue}' tissue "
            "and factor that into your assessment."
        )

    runner = client.beta.messages.tool_runner(
        model=MODEL,
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        tools=ALL_TOOLS,
        messages=[{"role": "user", "content": user_request}],
    )

    final_text = ""
    for message in runner:
        for block in message.content:
            if block.type == "text":
                final_text = block.text
    return final_text
