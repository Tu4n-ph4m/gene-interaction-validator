"""No-Streamlit web frontend: FastAPI backend + plain HTML/CSS/JS frontend.

Input: a list of genes. Output: every interacting pair found among them,
with links to the StringDB/BioGRID source records. Same underlying logic
as the CLI's --genes mode and the Streamlit app (gene_validator.batch).

Run:
    uvicorn webapp:app --reload
    -> open http://127.0.0.1:8000
"""

import os
from dataclasses import asdict
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from gene_validator.batch import validate_gene_network
from gene_validator.chat import chat_turn
from gene_validator.species import SPECIES_DISPLAY_OPTIONS, resolve_species

load_dotenv()

app = FastAPI(title="Gene Interaction Network Finder")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


class NetworkRequest(BaseModel):
    genes: List[str] = Field(..., min_length=2)
    tissue: Optional[str] = None
    species: Optional[str] = "human"


class ChatTurn(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    history: List[ChatTurn] = Field(default_factory=list)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/species")
def species_options() -> dict:
    return {"options": SPECIES_DISPLAY_OPTIONS}


@app.post("/api/network")
def network(req: NetworkRequest) -> dict:
    if not os.environ.get("ANTHROPIC_API_KEY") or not os.environ.get("BIOGRID_ACCESS_KEY"):
        raise HTTPException(
            status_code=500,
            detail="Server is missing ANTHROPIC_API_KEY/BIOGRID_ACCESS_KEY -- check .env.",
        )

    genes = [g.strip() for g in req.genes if g.strip()]
    if len(genes) < 2:
        raise HTTPException(status_code=400, detail="Provide at least 2 gene symbols.")

    try:
        species_tax_id = resolve_species(req.species)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    try:
        results, invalid_genes = validate_gene_network(genes, species_tax_id, req.tissue)
    except Exception as exc:  # surface upstream API errors as a clean 502, not a stack trace
        raise HTTPException(status_code=502, detail=f"Lookup failed: {exc}") from exc

    return {
        "results": [asdict(r) for r in results],
        "invalid_genes": invalid_genes,
    }


@app.post("/api/chat")
def chat(req: ChatRequest) -> dict:
    if not os.environ.get("ANTHROPIC_API_KEY") or not os.environ.get("BIOGRID_ACCESS_KEY"):
        raise HTTPException(
            status_code=500,
            detail="Server is missing ANTHROPIC_API_KEY/BIOGRID_ACCESS_KEY -- check .env.",
        )

    try:
        reply, results_payload = chat_turn(
            req.message, [turn.model_dump() for turn in req.history]
        )
    except ValueError as exc:  # e.g. an unresolvable species name extracted from chat
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Chat turn failed: {exc}") from exc

    return {"reply": reply, "results": results_payload}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
