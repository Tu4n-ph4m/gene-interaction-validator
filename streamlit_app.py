"""Conversational frontend for the gene interaction network finder.

Chat with the agent in plain English -- it extracts genes/tissue/species
from your message, finds real StringDB/BioGRID interactions, and writes a
short conversational reply plus a results table.

Run locally:
    streamlit run streamlit_app.py

Deployed on Streamlit Community Cloud, secrets (ANTHROPIC_API_KEY,
BIOGRID_ACCESS_KEY) are set in the app's Settings -> Secrets, and bridged
into os.environ below so the existing gene_validator code (which reads
os.environ directly) works unchanged in both environments.
"""

import os
import re

import pandas as pd
import streamlit as st

for _key in ("ANTHROPIC_API_KEY", "BIOGRID_ACCESS_KEY"):
    if _key in st.secrets:
        os.environ[_key] = st.secrets[_key]

from gene_validator.chat import chat_turn  # noqa: E402

st.set_page_config(page_title="Gene Interaction Chat", page_icon="🧬", layout="centered")

st.title("🧬 Gene Interaction Chat")
st.caption(
    'Ask in plain English -- e.g. "do BRCA1, BRCA2, and TP53 interact in liver?" -- '
    "or attach a gene list file below."
)

if "messages" not in st.session_state:
    st.session_state.messages = []  # [{"role", "content", "results": optional dict}]
if "last_file_name" not in st.session_state:
    st.session_state.last_file_name = None


def render_results_table(results_payload: dict) -> None:
    invalid_genes = results_payload.get("invalid_genes") or []
    if invalid_genes:
        st.warning(f"Could not resolve: {', '.join(invalid_genes)}")

    rows = results_payload.get("results") or []
    if not rows:
        st.caption("No interactions found among the resolved genes.")
        return

    df = pd.DataFrame(rows)
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
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "StringDB Link": st.column_config.LinkColumn("StringDB", display_text="View ↗"),
            "BioGRID Link": st.column_config.LinkColumn("BioGRID", display_text="View ↗"),
        },
    )

    caveat_rows = df[df["notes"].str.contains("CAVEAT", na=False)]
    if not caveat_rows.empty:
        with st.expander(f"⚠️ {len(caveat_rows)} pair(s) flagged with a tissue/independence caveat"):
            for _, row in caveat_rows.iterrows():
                st.write(f"**{row['gene1']} ↔ {row['gene2']}**: {row['notes']}")

    st.download_button(
        "Download full results (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="gene_interaction_network.csv",
        mime="text/csv",
        key=f"download-{id(results_payload)}",
    )


def run_turn(message: str) -> None:
    if not os.environ.get("BIOGRID_ACCESS_KEY"):
        st.session_state.messages.append(
            {"role": "assistant", "content": "BIOGRID_ACCESS_KEY is not configured for this app."}
        )
        return

    text_history = [
        {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
    ]
    st.session_state.messages.append({"role": "user", "content": message})

    with st.spinner("Thinking..."):
        try:
            reply, results_payload = chat_turn(message, text_history)
        except Exception as exc:
            reply, results_payload = f"Something went wrong: {exc}", None

    st.session_state.messages.append(
        {"role": "assistant", "content": reply, "results": results_payload}
    )


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("results"):
            render_results_table(msg["results"])

gene_file = st.file_uploader(
    "Attach a gene list file (.txt or .csv)", type=["txt", "csv"], label_visibility="collapsed"
)
if gene_file and gene_file.name != st.session_state.last_file_name:
    st.session_state.last_file_name = gene_file.name
    genes = [g for g in re.split(r"[,\s]+", gene_file.read().decode("utf-8").strip()) if g]
    if genes:
        run_turn(f"Find interactions among these genes: {', '.join(genes)}.")
        st.rerun()

if user_message := st.chat_input("Type a message..."):
    run_turn(user_message)
    st.rerun()
