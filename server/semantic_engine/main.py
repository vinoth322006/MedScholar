import os
import time
import logging
import xml.etree.ElementTree as ET
from contextlib import asynccontextmanager
from typing import Optional, List, Dict

import faiss
import numpy as np
import requests
import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
from pydantic import BaseModel
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

load_dotenv()
logger = logging.getLogger("uvicorn.error")

# ---- Config, loaded from .env — never hardcode secrets in the script itself ----
HF_REPO = os.getenv("HF_REPO", "dharandhamo/pubmed-medcpt-ivfpq")
HF_TOKEN = os.getenv("HF_TOKEN") or None  # only needed if the repo is private
NCBI_EMAIL = os.getenv("NCBI_EMAIL")
NCBI_API_KEY = os.getenv("NCBI_API_KEY")
NCBI_TOOL = "medcpt-semantic-search-local"
NPROBE = int(os.getenv("NPROBE", "24"))
RERANK_POOL_MAX = int(os.getenv("RERANK_POOL_MAX", "50"))  # hard ceiling on reranked candidates, regardless of k

state: Dict = {}


def encode_query(query: str) -> np.ndarray:
    with torch.no_grad():
        inputs = state["tokenizer"](
            [query], truncation=True, padding=True, return_tensors="pt", max_length=64
        ).to(state["device"])
        embedding = state["model"](**inputs).last_hidden_state[:, 0, :]
    return embedding.cpu().numpy().astype("float32")


def rerank(query: str, candidates: List[dict]) -> List[dict]:
    """Cross-encoder reranking over a candidate shortlist. Returns candidates
    sorted by rerank score, each with a 'rerank_score' key added.
    Never raises — on any model failure, logs and returns candidates
    unchanged (original FAISS order) so a reranker issue never breaks the
    whole request."""
    if not candidates:
        return candidates
    try:
        pairs = [[query, f"{c.get('title', '')} {c.get('abstract', '')}".strip()] for c in candidates]
        inputs = state["rerank_tokenizer"](
            pairs, truncation=True, padding=True, return_tensors="pt", max_length=512
        ).to(state["device"])
        with torch.no_grad():
            scores = state["rerank_model"](**inputs).logits.squeeze(-1).cpu().tolist()
        # squeeze(-1) on a single-candidate batch returns a 0-d tensor; .tolist()
        # then gives a float, not a list — normalize that edge case explicitly
        if isinstance(scores, float):
            scores = [scores]
        for c, s in zip(candidates, scores):
            c["rerank_score"] = float(s)
        return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    except Exception as e:
        logger.warning(f"Reranking failed, falling back to FAISS order: {e}")
        for c in candidates:
            c.setdefault("rerank_score", c.get("score", 0.0))
        return candidates


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ---------------------------------------------------------
    # STARTUP — everything here runs ONCE, not per-request
    # ---------------------------------------------------------
    print("Downloading merged FAISS index + PMID map from Hugging Face Hub...")
    print("(first run will take a while — files are cached locally after that)")

    index_path = hf_hub_download(
        repo_id=HF_REPO, repo_type="dataset",
        filename="merged/medcpt_full_ivfpq.index", token=HF_TOKEN,
    )
    pmids_path = hf_hub_download(
        repo_id=HF_REPO, repo_type="dataset",
        filename="merged/id_to_pmid.npy", token=HF_TOKEN,
    )

    index = faiss.read_index(index_path)
    index.nprobe = NPROBE
    state["index"] = index
    state["id_to_pmid"] = np.load(pmids_path)
    print(f"✅ FAISS index loaded — {index.ntotal:,} vectors, nprobe={NPROBE}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state["device"] = device

    print(f"Loading MedCPT Query Encoder on device: {device} ...")
    state["tokenizer"] = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
    state["model"] = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to(device).eval()
    _ = encode_query("warmup")
    print("✅ Query encoder ready")

    print(f"Loading MedCPT Cross-Encoder on device: {device} ...")
    state["rerank_tokenizer"] = AutoTokenizer.from_pretrained("ncbi/MedCPT-Cross-Encoder")
    state["rerank_model"] = AutoModelForSequenceClassification.from_pretrained(
        "ncbi/MedCPT-Cross-Encoder"
    ).to(device).eval()
    # warm up the reranker too, so the first real request isn't paying init cost
    _ = rerank("warmup", [{"title": "warmup", "abstract": "warmup text"}])
    print("✅ Cross-Encoder ready — server is fully warmed up")

    yield  # ---- the app serves requests here ----

    print("Shutting down.")


app = FastAPI(title="PubMed MedCPT Semantic Search (Local)", lifespan=lifespan)

# Loosen this before any real deployment beyond your own machine
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def fetch_pubmed_metadata(pmids: List[int], max_retries: int = 4) -> Dict[int, dict]:
    """One batched EFetch call for all requested PMIDs — not one call per PMID.
    Retries with exponential backoff on transient 502/503/429 responses."""
    if not pmids:
        return {}

    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    payload = {
        "db": "pubmed",
        "id": ",".join(str(p) for p in pmids),
        "rettype": "abstract",
        "retmode": "xml",
        "tool": NCBI_TOOL,
        "email": NCBI_EMAIL,
        "api_key": NCBI_API_KEY,
    }

    resp = None
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, data=payload, timeout=30)
            resp.raise_for_status()
            break
        except requests.exceptions.HTTPError:
            status = resp.status_code if resp is not None else None
            if status in (502, 503, 429) and attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise

    root = ET.fromstring(resp.content)
    metadata = {}
    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        if pmid_el is None:
            continue
        pmid = int(pmid_el.text)

        title_el = article.find(".//ArticleTitle")
        title = "".join(title_el.itertext()).strip() if title_el is not None else "(no title)"

        abstract_parts = article.findall(".//Abstract/AbstractText")
        abstract = (
            " ".join("".join(p.itertext()).strip() for p in abstract_parts)
            if abstract_parts else "(no abstract available)"
        )

        doi = None
        for aid in article.findall(".//ArticleIdList/ArticleId"):
            if aid.get("IdType") == "doi":
                doi = aid.text
                break

        year_el = article.find(".//PubDate/Year")
        pubdate = year_el.text if year_el is not None else "(unknown date)"

        metadata[pmid] = {"title": title, "abstract": abstract, "doi": doi, "pubdate": pubdate}
    return metadata


class SearchResult(BaseModel):
    pmid: int
    score: float
    rerank_score: Optional[float] = None
    title: str
    abstract: str
    doi: Optional[str] = None
    pubdate: str


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    timings_ms: Dict[str, float]


@app.get("/search", response_model=SearchResponse)
def search(q: str = Query(..., min_length=1, description="Search query"),
           k: int = Query(10, ge=1, le=50, description="Number of final results to return")):
    timings: Dict[str, float] = {}

    # ---- 1. Encode query ----
    t0 = time.time()
    q_vec = encode_query(q)
    timings["encode_ms"] = round((time.time() - t0) * 1000, 1)

    # ---- 2. FAISS search — pull a WIDER pool than k, so reranking has real
    #         signal to work with, capped so latency stays predictable
    #         regardless of what k the caller asks for ----
    candidate_pool_size = min(max(k * 3, 20), RERANK_POOL_MAX)
    t0 = time.time()
    scores, indices = state["index"].search(q_vec, candidate_pool_size)
    timings["faiss_search_ms"] = round((time.time() - t0) * 1000, 1)

    pmid_score_pairs = [
        (int(state["id_to_pmid"][i]), float(s))
        for i, s in zip(indices[0], scores[0]) if i != -1
    ]
    if not pmid_score_pairs:
        raise HTTPException(status_code=404, detail="No results found")

    # ---- 3. Metadata for the whole candidate pool (needed for reranking text) ----
    t0 = time.time()
    metadata = fetch_pubmed_metadata([pmid for pmid, _ in pmid_score_pairs])
    timings["ncbi_fetch_ms"] = round((time.time() - t0) * 1000, 1)

    candidates = []
    for pmid, score in pmid_score_pairs:
        meta = metadata.get(pmid) or {
            "title": "(metadata unavailable)", "abstract": "(metadata unavailable)",
            "doi": None, "pubdate": "(unknown)",
        }
        candidates.append({"pmid": pmid, "score": score, **meta})

    # ---- 4. Cross-encoder rerank the pool, then trim to what the caller asked for ----
    t0 = time.time()
    reranked = rerank(q, candidates)[:k]
    timings["rerank_ms"] = round((time.time() - t0) * 1000, 1)

    timings["total_ms"] = round(sum(timings.values()), 1)

    results = [
        SearchResult(
            pmid=c["pmid"], score=round(c["score"], 2),
            rerank_score=round(c["rerank_score"], 4) if "rerank_score" in c else None,
            title=c["title"], abstract=c["abstract"], doi=c["doi"], pubdate=c["pubdate"],
        )
        for c in reranked
    ]

    return SearchResponse(query=q, results=results, timings_ms=timings)


@app.get("/health")
def health():
    ready = all(k in state for k in ("index", "model", "rerank_model"))
    return {
        "status": "healthy" if ready else "starting",
        "index_size": state["index"].ntotal if "index" in state else 0,
        "device": state.get("device", "unknown"),
    }


@app.get("/")
def root():
    return {"status": "running", "docs": "/docs", "search": "/search?q=your+query&k=10"}


@app.get("/api/search")
def litsense_compat_search(query: str = Query(...), rerank_param: str = Query("false", alias="rerank")):
    """Legacy LitSense compatibility endpoint. 
    The Express backend expects an array of {text, score, pmid, pmcid, section}."""
    k = 20
    do_rerank = rerank_param.lower() == "true"
    
    q_vec = encode_query(query)
    candidate_pool_size = min(max(k * 3, 20), RERANK_POOL_MAX)
    scores, indices = state["index"].search(q_vec, candidate_pool_size)
    pmid_score_pairs = [
        (int(state["id_to_pmid"][i]), float(s))
        for i, s in zip(indices[0], scores[0]) if i != -1
    ]
    if not pmid_score_pairs:
        return []

    metadata = fetch_pubmed_metadata([pmid for pmid, _ in pmid_score_pairs])

    candidates = []
    for pmid, score in pmid_score_pairs:
        meta = metadata.get(pmid) or {
            "title": "", "abstract": "",
            "doi": None, "pubdate": "",
        }
        candidates.append({"pmid": pmid, "score": score, **meta})
        
    if do_rerank:
        final_candidates = rerank(query, candidates)[:k]
    else:
        final_candidates = candidates[:k]
        
    out = []
    for c in final_candidates:
        text = c["abstract"] if c.get("abstract") and c["abstract"] != "(metadata unavailable)" else c["title"]
        out.append({
            "text": text,
            "score": float(c.get("rerank_score", c["score"])),
            "pmid": int(c["pmid"]),
            "pmcid": "",
            "section": "abstract"
        })
    return out