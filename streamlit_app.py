"""Interactive frontend for the gene-gene interaction validation agent.

Run locally:
    streamlit run streamlit_app.py

Deployed on Streamlit Community Cloud, secrets (ANTHROPIC_API_KEY,
BIOGRID_ACCESS_KEY) are set in the app's Settings -> Secrets, and bridged
into os.environ below so the existing gene_validator code (which reads
os.environ directly) works unchanged in both environments.
"""

import os

import streamlit as st

for _key in ("ANTHROPIC_API_KEY", "BIOGRID_ACCESS_KEY"):
    if _key in st.secrets:
        os.environ[_key] = st.secrets[_key]

from gene_validator.agent import validate_gene_pair_stream  # noqa: E402

st.set_page_config(page_title="Gene Interaction Validator", page_icon="🧬", layout="centered")

st.title("🧬 Gene-Gene Interaction Validator")
st.caption(
    "An agentic loop cross-validates a gene pair against StringDB, BioGRID, "
    "and (optionally) tissue expression via the Human Protein Atlas."
)

with st.form("validate_form"):
    col1, col2 = st.columns(2)
    gene1 = col1.text_input("Gene 1", placeholder="BRCA1").strip()
    gene2 = col2.text_input("Gene 2", placeholder="BRCA2").strip()
    tissue = st.text_input(
        "Tissue / cell type (optional)",
        placeholder="liver, prostate, bone marrow, ...",
    ).strip()
    species_tax_id = st.number_input(
        "Species NCBI taxonomy ID", value=9606, step=1, help="9606 = human"
    )
    submitted = st.form_submit_button("Validate", use_container_width=True)

TOOL_LABELS = {
    "resolve_gene": "🔎 Resolving gene identifier",
    "check_string_interaction": "🧪 Checking StringDB",
    "check_biogrid_interaction": "🧫 Checking BioGRID",
    "check_tissue_expression": "🧬 Checking tissue expression (HPA)",
}

if submitted:
    if not gene1 or not gene2:
        st.error("Please enter both gene symbols.")
    elif not os.environ.get("ANTHROPIC_API_KEY"):
        st.error("ANTHROPIC_API_KEY is not configured for this app.")
    else:
        final_text = ""
        with st.status(f"Validating {gene1} ↔ {gene2}...", expanded=True) as status:
            try:
                for event in validate_gene_pair_stream(
                    gene1, gene2, int(species_tax_id), tissue or None
                ):
                    if event["type"] == "tool_call":
                        label = TOOL_LABELS.get(event["name"], f"🔧 Calling {event['name']}")
                        args = ", ".join(f"{k}={v!r}" for k, v in event["input"].items())
                        st.write(f"{label} — `{args}`")
                    elif event["type"] == "tool_result":
                        with st.expander(f"Raw result from `{event['name']}`"):
                            st.code(event["output"], language="json")
                    elif event["type"] == "text":
                        final_text += event["text"]
                        st.write(event["text"])
            except Exception as exc:  # surface API/network errors in the UI, not a stack trace
                status.update(label="Failed", state="error")
                st.error(f"Validation failed: {exc}")
                final_text = ""
            else:
                status.update(label="Done", state="complete")

        if final_text:
            st.divider()
            st.markdown("## Verdict")
            st.markdown(final_text)
