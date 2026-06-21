"""Common-name <-> NCBI taxonomy ID resolution, so user-facing inputs (CLI,
web UI, CSV) can take "human"/"mouse" instead of requiring "9606"/"10090".

Internal code (gene_validator.tools, gene_validator.batch) keeps using the
raw NCBI taxonomy ID int everywhere -- this module is only consulted at the
edges where a human types a species.
"""

from __future__ import annotations

# name/synonym (lowercase) -> NCBI taxonomy ID
COMMON_SPECIES = {
    "human": 9606,
    "homo sapiens": 9606,
    "mouse": 10090,
    "mus musculus": 10090,
    "rat": 10116,
    "rattus norvegicus": 10116,
    "zebrafish": 7955,
    "danio rerio": 7955,
    "fruit fly": 7227,
    "fly": 7227,
    "drosophila": 7227,
    "drosophila melanogaster": 7227,
    "yeast": 4932,
    "saccharomyces cerevisiae": 4932,
    "c. elegans": 6239,
    "c elegans": 6239,
    "celegans": 6239,
    "caenorhabditis elegans": 6239,
    "chicken": 9031,
    "gallus gallus": 9031,
    "pig": 9823,
    "sus scrofa": 9823,
    "cow": 9913,
    "bos taurus": 9913,
    "dog": 9615,
    "canis lupus familiaris": 9615,
    "rabbit": 9986,
    "oryctolagus cuniculus": 9986,
}

# Display options for dropdowns, in a sensible default order.
SPECIES_DISPLAY_OPTIONS = [
    "Human",
    "Mouse",
    "Rat",
    "Zebrafish",
    "Fruit fly",
    "Yeast",
    "C. elegans",
    "Chicken",
    "Pig",
    "Cow",
    "Dog",
    "Rabbit",
]


def resolve_species(value) -> int:
    """Resolve a species name or NCBI taxonomy ID to an int taxonomy ID.

    Accepts: an int, a numeric string ("9606"), a common name ("human",
    case-insensitive), or a scientific name ("Homo sapiens"). None defaults
    to human (9606, the species this project is built around).
    """
    if value is None or value == "":
        return 9606
    if isinstance(value, int):
        return value

    text = str(value).strip()
    if text.isdigit():
        return int(text)

    key = text.lower()
    if key in COMMON_SPECIES:
        return COMMON_SPECIES[key]

    raise ValueError(
        f"Unknown species '{value}'. Use a common name (e.g. human, mouse, "
        f"zebrafish) or an NCBI taxonomy ID (e.g. 9606)."
    )
