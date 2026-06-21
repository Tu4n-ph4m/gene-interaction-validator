"""CLI entry point: validate gene-gene pairs against StringDB and BioGRID.

Single pair (agentic, prose report):
    python main.py GENE1 GENE2
    python main.py GENE1 GENE2 --species mouse --tissue liver

--species accepts a common name (human, mouse, rat, zebrafish, fruit fly,
yeast, c. elegans, chicken, pig, cow, dog, rabbit) or a raw NCBI taxonomy ID
(e.g. 9606). Defaults to human.

Explicit pairs from a CSV (deterministic, structured table):
    python main.py --batch pairs.csv --out results.csv
    python main.py --batch pairs.csv --out results.json --format json

Existing interactions within a gene list (any size, even ~1000 genes),
optionally scoped to a tissue/cell type (deterministic, structured table):
    python main.py --genes BRCA1,BRCA2,TP53,EGFR --tissue breast --out results.csv
    python main.py --gene-file genes.txt --tissue liver --out results.csv

--batch and --gene-file also accept an http(s) URL instead of a local path
(e.g. a Google Sheets export link or a raw GitHub file URL) -- the file is
fetched directly, no local download needed:
    python main.py --gene-file https://example.com/genes.txt --tissue liver --out results.csv
    python main.py --batch https://example.com/pairs.csv --out results.csv

Only pairs with at least one database reporting evidence are returned --
a full matrix including the (huge majority of) non-interacting pairs would
be mostly-empty noise once the list reaches hundreds of genes.

pairs.csv must have columns: gene1,gene2  (optional: species_tax_id,tissue --
species_tax_id may also be a common name like "mouse")
A per-row "tissue" column overrides --tissue for that row.
"""

import argparse
import sys

from gene_validator.agent import validate_gene_pair
from gene_validator.batch import (
    read_source,
    validate_gene_network,
    validate_pairs_from_csv,
    write_results_csv,
    write_results_json,
)
from gene_validator.species import resolve_species


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--batch",
        help="CSV file (or URL) of gene1,gene2 pairs to validate",
    )
    parser.add_argument(
        "--genes",
        help="Comma-separated gene list; finds existing interactions within it",
    )
    parser.add_argument(
        "--gene-file",
        help="File (or URL) with one gene symbol per line (or comma-separated); for large lists",
    )
    parser.add_argument("--out", help="Output file path for batch/genes results")
    parser.add_argument("--format", choices=["csv", "json"], default="csv")
    parser.add_argument(
        "--tissue",
        help="Tissue/cell type to check expression context for (e.g. 'liver', 'prostate')",
    )
    parser.add_argument(
        "--species",
        default="human",
        help="Species name (human, mouse, rat, zebrafish, ...) or NCBI taxonomy ID. Default: human",
    )
    parser.add_argument("positional", nargs="*")
    args = parser.parse_args()

    try:
        species_tax_id = resolve_species(args.species)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    if args.batch or args.genes or args.gene_file:
        if not args.out:
            print("Error: --batch/--genes/--gene-file requires --out FILE")
            sys.exit(1)

        invalid_genes: list[str] = []
        if args.genes or args.gene_file:
            if args.gene_file:
                raw = read_source(args.gene_file)
                genes = [g.strip() for g in raw.replace(",", "\n").splitlines() if g.strip()]
            else:
                genes = [g.strip() for g in args.genes.split(",") if g.strip()]
            if len(genes) < 2:
                print("Error: need at least 2 gene symbols")
                sys.exit(1)
            results, invalid_genes = validate_gene_network(
                genes, species_tax_id, tissue=args.tissue
            )
        else:
            results = validate_pairs_from_csv(args.batch, default_tissue=args.tissue)

        if not results:
            print("No interactions found among the given genes (or all genes were invalid).")
        elif args.format == "json":
            write_results_json(results, args.out)
        else:
            write_results_csv(results, args.out)

        if results:
            print(f"Found {len(results)} interacting pairs -> {args.out}")
        if invalid_genes:
            print(f"Could not resolve {len(invalid_genes)} gene(s): {', '.join(invalid_genes)}")
        return

    if len(args.positional) < 2:
        print(__doc__)
        sys.exit(1)

    gene1, gene2 = args.positional[0], args.positional[1]
    if len(args.positional) > 2:
        try:
            species_tax_id = resolve_species(args.positional[2])
        except ValueError as exc:
            print(f"Error: {exc}")
            sys.exit(1)

    result = validate_gene_pair(gene1, gene2, species_tax_id, tissue=args.tissue)
    print(result)


if __name__ == "__main__":
    main()
