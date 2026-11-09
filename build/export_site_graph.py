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


TYPES = {t.lower(): t for t in ("Person", "Organization", "Role", "Education", "Achievement", "Project", "Algorithm",
                                 "Method", "Technology", "PhysicsConcept", "Process", "Result", "Metric", "Dataset")}
ALIASES = {"gopal": "Gopal Sharma", "gopalsharma": "Gopal Sharma", "gopalsharma1615ai": "Gopal Sharma",
           "httpsgithubcomgopalsharma1615ai": "Gopal Sharma"}
EXCLUDE = {"claudedesktop", "claudecode", "claude",   # MCP-client examples from setup docs, not portfolio content
           # skill-list headings: the LLM turns them into hubs that connect everything to everything
           "generativeaiandllms", "machinelearningandnlp", "retrievalanddata", "engineering",
           "physicsandmathematics", "skills", "education", "achievements",
           # one-word generic nouns the LLM promotes to entities; they only add distractor hubs
           "graph", "data", "model", "models", "system", "code", "tool", "tools", "method", "result", "results",
           "paper", "project", "document", "documents", "question", "questions", "answer", "query"}


# LLM spellings of curated concepts that differ by more than case/punctuation (key -> curated key)
SYNONYMS = {"multihopquestionanswering": "multihopqa", "βfunction": "betafunction", "betafunctions": "betafunction",
            "bidirectionalmeetinthemiddlesearch": "meetinthemiddlesearch", "meetinthemiddle": "meetinthemiddlesearch",
            "wardidentity": "wardtakahashiidentity", "modelcontextprotocolmcp": "modelcontextprotocol",
            "mcp": "modelcontextprotocol", "qwen314b": "qwen3", "renormalizationofquantumfieldtheory": "renormalizationofqft"}


def key(name: str) -> str:
    return "".join(c for c in name.lower() if c.isalnum())


def from_lightrag(workdir: Path) -> tuple[list[dict], list[dict]]:
    """Read a LightRAG graph and resolve entities: names equal up to case and punctuation are merged (the most
    connected spelling wins), owner aliases fold into one node, and LLM type labels map onto the site's types."""
    from graphwalk_rag.store import GraphStore

    gs = GraphStore(workdir)
    groups: dict[str, list[str]] = {}
    for n in gs.G.nodes:
        if n.startswith("project_") or n.endswith((".md", ".txt")):     # a corpus file name, not an entity
            continue
        k = key(ALIASES.get(key(n), n))
        if k and k not in EXCLUDE:
            groups.setdefault(k, []).append(n)
    canon = {}
    nodes = {}
    for k, names in groups.items():
        best = ALIASES.get(key(names[0])) or max(names, key=lambda x: (gs.degree(x), x != x.lower(), " " in x))
        types = [gs.node_type(x).lower() for x in names]
        t = next((TYPES[x] for x in types if x in TYPES), "Concept")
        desc = max((gs.node_desc(x) for x in names), key=len)[:320]
        nodes[best] = {"id": best, "type": t, "desc": desc, "docs": set().union(*(gs.node_docs(x) for x in names))}
        for x in names:
            canon[x] = best
    edges: dict[tuple, dict] = {}
    for u, v in gs.G.edges:
        a, b = canon.get(u), canon.get(v)
        if a is None or b is None or a == b:
            continue
        pair = tuple(sorted((a, b)))
        desc = gs.edge(u, v)["desc"][:260]
        docs = {c["doc"] for c in gs.edge_chunks(u, v)}
        if pair in edges:
            edges[pair]["docs"] |= docs
            if len(desc) > len(edges[pair]["desc"]):
                edges[pair]["desc"] = desc
        else:
            edges[pair] = {"s": pair[0], "t": pair[1], "desc": desc, "docs": docs}
    linked = {e["s"] for e in edges.values()} | {e["t"] for e in edges.values()}
    print(f"entity resolution: {gs.G.number_of_nodes()} raw -> {len(nodes)} merged -> {len(linked)} connected")
    return [n for n in nodes.values() if n["id"] in linked], list(edges.values())


def union(base: tuple[list[dict], list[dict]], extra: tuple[list[dict], list[dict]]) -> tuple[list[dict], list[dict]]:
    """Curated backbone + LLM-extracted breadth. Entities matching a curated name (up to case/punctuation) merge into
    the curated node, which keeps its name, type and description; everything else is added."""
    bn, be = base
    en, ee = extra
    by_key = {key(n["id"]): n for n in bn}
    rename = {}
    for n in en:
        k = SYNONYMS.get(key(n["id"]), key(n["id"]))
        if k.startswith("project") and k[7:] in by_key:      # "ProjectGraphwalkRag": a README name echoed back
            k = k[7:]
        if k in by_key:
            by_key[k]["docs"] |= n["docs"]
            rename[n["id"]] = by_key[k]["id"]
        else:
            by_key[k] = n
            rename[n["id"]] = n["id"]
    edges = {tuple(sorted((e["s"], e["t"]))): e for e in be}
    for e in ee:
        a, b = rename[e["s"]], rename[e["t"]]
        if a == b:
            continue
        pair = tuple(sorted((a, b)))
        if pair in edges:
            edges[pair]["docs"] |= e["docs"]
        else:
            edges[pair] = {"s": a, "t": b, "desc": e["desc"], "docs": e["docs"]}
    print(f"union: {len(bn)} curated + {len(en)} extracted -> {len(by_key)} entities, {len(edges)} relations")
    return list(by_key.values()), list(edges.values())


def q8(mat: np.ndarray) -> str:
    return base64.b64encode(np.clip(np.round(mat * 127), -127, 127).astype(np.int8).tobytes()).decode()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--curated", type=Path, help="curated triples (alone, or as the backbone of a union)")
    ap.add_argument("--lightrag", type=Path, help="LightRAG working dir (alone, or merged onto --curated)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--source", default=None, help="label shown on the site")
    ap.add_argument("--hub-degree", type=int, default=0, help="0 = automatic (max(15, 98th percentile))")
    a = ap.parse_args()
    if not (a.curated or a.lightrag):
        ap.error("give --curated, --lightrag, or both")

    if a.curated and a.lightrag:
        nodes, edges = union(from_curated(a.curated), from_lightrag(a.lightrag))
    else:
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
