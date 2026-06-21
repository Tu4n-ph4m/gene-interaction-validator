"""Interactive frontend for the gene-gene interaction validation agent.

Input: a list of genes. Output: every interacting pair found among them,
with clickable links to the StringDB/BioGRID source records.

Run locally:
    streamlit run streamlit_app.py

Deployed on Streamlit Community Cloud, secrets (ANTHROPIC_API_KEY,
BIOGRID_ACCESS_KEY) are set in the app's Settings -> Secrets, and bridged
into os.environ below so the existing gene_validator code (which reads
os.environ directly) works unchanged in both environments.
"""

import os
import re
from dataclasses import asdict

import pandas as pd
import streamlit as st

for _key in ("ANTHROPIC_API_KEY", "BIOGRID_ACCESS_KEY"):
    if _key in st.secrets:
        os.environ[_key] = st.secrets[_key]

from gene_validator.batch import validate_gene_network  # noqa: E402

st.set_page_config(page_title="Gene Interaction Network", page_icon="🧬", layout="wide")

st.title("🧬 Gene Interaction Network Finder")
st.caption(
    "Paste a list of genes. Finds every interacting pair among them via "
    "StringDB and BioGRID, with links to the source records. Optionally "
    "scope to a tissue/cell type via the Human Protein Atlas."
)

with st.form("network_form"):
    genes_text = st.text_area(
        "Genes (comma, space, or newline separated)",
        placeholder="BRCA1, BRCA2, TP53, EGFR, MYC, PTEN",
        height=140,
    )
    col1, col2 = st.columns(2)
    tissue = col1.text_input(
        "Tissue / cell type (optional)", placeholder="liver, prostate, bone marrow, ..."
    ).strip()
    species_tax_id = col2.number_input(
        "Species NCBI taxonomy ID", value=9606, step=1, help="9606 = human"
    )
    submitted = st.form_submit_button("Find interactions", use_container_width=True)

if submitted:
    genes = [g for g in re.split(r"[,\s]+", genes_text.strip()) if g]
    if len(genes) < 2:
        st.error("Enter at least 2 gene symbols.")
    elif not os.environ.get("BIOGRID_ACCESS_KEY"):
        st.error("BIOGRID_ACCESS_KEY is not configured for this app.")
    else:
        with st.spinner(f"Querying StringDB + BioGRID for {len(genes)} genes..."):
            try:
                results, invalid_genes = validate_gene_network(
                    genes, int(species_tax_id), tissue or None
                )
            except Exception as exc:
                st.error(f"Lookup failed: {exc}")
                results, invalid_genes = [], []

        if invalid_genes:
            st.warning(f"Could not resolve {len(invalid_genes)} gene(s): {', '.join(invalid_genes)}")

        if not results:
            st.info("No interactions found among the resolved genes.")
        else:
            df = pd.DataFrame([asdict(r) for r in results])
            df = df.sort_values("string_combined_score", ascending=False, na_position="last")

            display_df = df[
                [
                    "gene1",
                    "gene2",
                    "verdict",
                    "string_combined_score",
                    "string_curated_overlap_risk",
                    "biogrid_evidence_count",
                    "string_source_url",
                    "biogrid_source_url",
                ]
            ].rename(
                columns={
                    "gene1": "Gene 1",
                    "gene2": "Gene 2",
                    "verdict": "Verdict",
                    "string_combined_score": "StringDB Score",
                    "string_curated_overlap_risk": "Curated-Overlap Risk",
                    "biogrid_evidence_count": "BioGRID Evidence #",
                    "string_source_url": "StringDB Link",
                    "biogrid_source_url": "BioGRID Link",
                }
            )

            st.success(f"Found {len(results)} interacting pairs among {len(genes) - len(invalid_genes)} genes.")
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "StringDB Link": st.column_config.LinkColumn("StringDB", display_text="View ↗"),
                    "BioGRID Link": st.column_config.LinkColumn("BioGRID", display_text="View ↗"),
                },
            )

            if tissue:
                caveat_rows = df[df["notes"].str.contains("CAVEAT", na=False)]
                if not caveat_rows.empty:
                    with st.expander(f"⚠️ {len(caveat_rows)} pair(s) flagged with a tissue/independence caveat"):
                        for _, row in caveat_rows.iterrows():
                            st.write(f"**{row['gene1']} ↔ {row['gene2']}**: {row['notes']}")

            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download full results (CSV)",
                data=csv_bytes,
                file_name="gene_interaction_network.csv",
                mime="text/csv",
            )
