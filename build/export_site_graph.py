"""Export a knowledge graph for the in-browser demo: site/data/graph.json.

    python build/export_site_graph.py --curated build/curated_graph.json --out site/data/graph.json
    python build/export_site_graph.py --lightrag build/graph_work     --out site/data/graph.json

Entity and relation vectors are computed with sentence-transformers/all-MiniLM-L6-v2, the same model the browser
runs through transformers.js, so query and graph vectors live in one space. They are stored as int8 (x127),
which keeps a few-thousand-node graph well under a few MB.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import json
from pathlib import Path

import numpy as np

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def from_curated(path: Path) -> tuple[list[dict], list[dict]]:
    d = json.loads(path.read_text())
    nodes = {e["name"]: {"id": e["name"], "type": e["type"], "desc": e["description"], "docs": set()}
             for e in d["entities"]}
    edges = []
    for r in d["relations"]:
        edges.append({"s": r["source"], "t": r["target"], "desc": r["description"], "docs": {r["document"]}})
        nodes[r["source"]]["docs"].add(r["document"])
        nodes[r["target"]]["docs"].add(r["document"])
    return list(nodes.values()), edges


def from_lightrag(workdir: Path) -> tuple[list[dict], list[dict]]:
    from graphwalk_rag.store import GraphStore

    gs = GraphStore(workdir)
    nodes = [{"id": n, "type": gs.node_type(n) or "Entity", "desc": gs.node_desc(n)[:320],
              "docs": set(gs.node_docs(n))} for n in gs.G.nodes if gs.degree(n) > 0]
    edges = [{"s": u, "t": v, "desc": gs.edge(u, v)["desc"][:260],
              "docs": {c["doc"] for c in gs.edge_chunks(u, v)}} for u, v in gs.G.edges]
    return nodes, edges


def q8(mat: np.ndarray) -> str:
    return base64.b64encode(np.clip(np.round(mat * 127), -127, 127).astype(np.int8).tobytes()).decode()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--curated", type=Path)
    src.add_argument("--lightrag", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source", default=None, help="label shown on the site")
    ap.add_argument("--hub-degree", type=int, default=0, help="0 = automatic (max(15, 98th percentile))")
    a = ap.parse_args()

    nodes, edges = from_curated(a.curated) if a.curated else from_lightrag(a.lightrag)
    idx = {n["id"]: i for i, n in enumerate(nodes)}
    edges = [e for e in edges if e["s"] in idx and e["t"] in idx and e["s"] != e["t"]]
    docs = sorted({d for n in nodes for d in n["docs"]} | {d for e in edges for d in e["docs"]})
    didx = {d: i for i, d in enumerate(docs)}
    deg = np.zeros(len(nodes), int)
    for e in edges:
        deg[idx[e["s"]]] += 1
        deg[idx[e["t"]]] += 1
    hub = a.hub_degree or int(max(15, np.percentile(deg, 98) if len(deg) else 15))

    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(MODEL, device="cpu")
    ent = st.encode([f"{n['id']}. {n['desc']}" for n in nodes], normalize_embeddings=True, batch_size=64)
    rel = st.encode([f"{e['s']} {e['desc']} {e['t']}" for e in edges], normalize_embeddings=True, batch_size=64)

    out = {
        "meta": {"source": a.source or ("Curated starter graph" if a.curated else "LightRAG graph"),
                 "built": dt.date.today().isoformat(), "nodes": len(nodes), "edges": len(edges),
                 "dim": int(ent.shape[1]), "model": MODEL, "hub_degree": hub},
        "docs": docs,
        "nodes": [[n["id"], n["type"], n["desc"], sorted(didx[d] for d in n["docs"])] for n in nodes],
        "edges": [[idx[e["s"]], idx[e["t"]], e["desc"], sorted(didx[d] for d in e["docs"])] for e in edges],
        "ent_vec": q8(ent), "rel_vec": q8(rel),
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, separators=(",", ":"), ensure_ascii=False))
    print(f"{len(nodes)} nodes, {len(edges)} edges, hub degree > {hub}, "
          f"{a.out.stat().st_size / 1024:.0f} KB -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
