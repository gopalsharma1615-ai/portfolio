import { GROUP_LABELS, OWNER, edgeIndex, lexicalScorer, loadGraph, norm, prepareQuery, search, vectorScorer,
  verbalize } from "./graph.js";

const $ = (s) => document.querySelector(s);
const sleep = (ms) => new Promise((r) => setTimeout(r, matchMedia("(prefers-reduced-motion: reduce)").matches ? 0 : ms));
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

const EXAMPLES = [
  "How did Gopal improve multi-hop question answering?",
  "What did Gopal build with Neo4j?",
  "Where does physics show up in his AI work?",
  "How is the beta function discovered from data?",
  "What experience does he have with Qwen and vLLM?",
  "How does the Ward identity connect to his code?",
];

const PROJECTS = [
  ["graphwalk-rag", "Graph RAG that walks the graph", "Provenance-preserving LightRAG build, NetworkX / Neo4j storage, bidirectional meet-in-the-middle retrieval, a document-routed walk and a single-first router.", "Gold-path recall 100% vs 88.6% with ≈200× fewer candidates"],
  ["layout-rag", "Layout-aware RAG, CPU-only", "Typography-derived heading levels, header-carrying table rows, chunks budgeted in the embedder's tokens, hybrid dense + BM25 search and a local reranker.", "OHR-Bench page hit@5 85.5% from raw PDFs (ceiling 95.5%)"],
  ["qft-renorm-ai", "QED renormalization, in code", "SymPy derives the Z-factors, Ward identity and β-function from Feynman rules; sparse regression rediscovers β₀ from coupling data.", "Asymptotic freedom lost at n_f ≈ 16.3 (theory 16.5)"],
  ["rag-mcp-server", "Retrieval as MCP tools", "Document search and graph walks exposed over the Model Context Protocol to any AI assistant, over stdio or HTTP.", "Hybrid search 0.11 s p50 on CPU; entity tools < 1 ms"],
  ["openai-budget-guard", "Token & cost budgets for LLM keys", "Per-key, per-task budgets with atomic preflight reservations, streaming metering, a SQLite ledger, proxy and CLI.", "Concurrent calls cannot overshoot; openai SDK 1.x–3.x"],
  ["research-agent", "Citation-verifying agent (design)", "An MCP-client agent that plans, retrieves, verifies every claim against evidence and answers with citations.", "Design stage: roadmap in the README"],
];

// ---------------------------------------------------------------------------------------------- theme
const root = document.documentElement;
try { const t = localStorage.getItem("theme"); if (t) root.dataset.theme = t; } catch {}
$("#theme").addEventListener("click", () => {
  const dark = getComputedStyle(root).colorScheme === "dark";
  root.dataset.theme = dark ? "light" : "dark";
  try { localStorage.setItem("theme", root.dataset.theme); } catch {}
  if (cy) cy.style(cyStyle());
});
const css = (v) => getComputedStyle(root).getPropertyValue(v).trim();

// ---------------------------------------------------------------------------------------------- static content
$("#chips").innerHTML = EXAMPLES.map((q) => `<button type="button" class="chip">${esc(q)}</button>`).join("");
$("#cards").innerHTML = PROJECTS.map(([name, tag, desc, metric]) => `<article class="card">
  <span class="tag">${esc(tag)}</span><h3>${esc(name)}</h3><p>${esc(desc)}</p><div class="metric">${esc(metric)}</div>
  <a href="https://github.com/gopalsharma1615-ai/${name}" rel="noopener">View on GitHub →</a></article>`).join("");

// ---------------------------------------------------------------------------------------------- graph + model
let G, cy, embedder = null, modelState = "loading", lastResult = null, lastQuery = "";
const status = (t) => { $("#status").textContent = t; };

async function loadModel() {
  try {
    const { pipeline, env } = await import("https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2");
    env.allowLocalModels = false;
    embedder = await pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2", { dtype: "q8" });
    modelState = "ready";
  } catch (e) {
    console.warn("embedding model unavailable, using lexical scoring", e);
    modelState = "fallback";
  }
  if (!lastResult) status(modelState === "ready" ? "Embedding model ready: the search runs on your device."
    : "Embedding model couldn't load here; using keyword scoring instead.");
}

async function scorer(q) {
  if (modelState === "loading") { status("Loading the in-browser embedding model (≈23 MB, once)…"); await modelReady; }
  if (embedder) {
    const out = await embedder(q, { pooling: "mean", normalize: true });
    return vectorScorer(G, out.data);
  }
  return lexicalScorer(G)(q);
}

function color(group) { return css(`--g-${group}`); }

function cyStyle() {
  return [
    { selector: "node", style: { "background-color": "data(color)", width: "data(size)", height: "data(size)",
      label: "data(label)", "font-size": 10, "font-family": "Inter, sans-serif", color: css("--ink"),
      "text-valign": "bottom", "text-margin-y": 3, "text-outline-color": css("--surface"), "text-outline-width": 2,
      "min-zoomed-font-size": 7, "border-width": 2, "border-color": css("--surface") } },
    { selector: "edge", style: { width: 1.2, "line-color": css("--line-2"), "curve-style": "haystack", opacity: 0.9 } },
    { selector: ".faded", style: { opacity: 0.12, "text-opacity": 0 } },
    { selector: "node.anchor", style: { "border-width": 4, "border-color": css("--ink"), "z-index": 9 } },
    { selector: "node.answer", style: { "border-width": 3, "border-style": "dashed", "border-color": css("--ink-2") } },
    { selector: "node.mid", style: { "border-width": 3, "border-color": css("--accent") } },
    { selector: "node.onpath", style: { opacity: 1, "font-weight": 600, "font-size": 12, "z-index": 10 } },
    { selector: "edge.onpath", style: { width: 4, "line-color": css("--path"), opacity: 1, "z-index": 10,
      "curve-style": "bezier", label: "data(short)", "font-size": 9, color: css("--ink-2"),
      "text-background-color": css("--surface"), "text-background-opacity": 1, "text-background-padding": 2,
      "text-rotation": "autorotate" } },
    { selector: "edge.other", style: { width: 2.2, "line-color": css("--ink-2"), opacity: 0.55, "curve-style": "bezier" } },
  ];
}

function initCy() {
  const big = G.nodes.length > 350;
  const deg = G.adj.map((a) => a.length);
  const keep = new Set(big ? [...G.nodes.keys()].sort((a, b) => deg[b] - deg[a]).slice(0, 300) : G.nodes.keys());
  const els = [];
  for (const n of G.nodes) {
    els.push({ group: "nodes", data: { id: `n${n.i}`, label: deg[n.i] >= 3 || !big ? n.id : "",
      full: n.id, color: color(n.group), size: 10 + Math.min(18, deg[n.i] * 1.6) }, classes: keep.has(n.i) ? "" : "hidden" });
  }
  for (const e of G.edges) {
    els.push({ group: "edges", data: { id: `e${e.i}`, source: `n${e.s}`, target: `n${e.t}`,
      short: e.desc.length > 38 ? e.desc.slice(0, 36) + "…" : e.desc } });
  }
  // maxZoom caps fit(): a 2-node path must not fill the canvas with giant labels
  cy = cytoscape({ container: $("#cy"), elements: els, style: cyStyle(), minZoom: 0.2, maxZoom: 1.6,
    boxSelectionEnabled: false });
  cy.$(".hidden").style("display", "none");
  layout(cy.elements(":visible"));
  cy.on("tap", "node", (ev) => {
    const n = G.nodes[+ev.target.id().slice(1)];
    status(`${n.id} · ${n.type}: ${n.desc}`);
  });
}

function layout(eles) {
  eles.layout({ name: "cose", animate: false, nodeRepulsion: () => 9000, idealEdgeLength: () => 70,
    gravity: 0.25, numIter: 1500, randomize: true, fit: true, padding: 30 }).run();
}

function renderLegend() {
  const used = new Set(G.nodes.map((n) => n.group));
  $("#legend").innerHTML = Object.entries(GROUP_LABELS).filter(([g]) => used.has(g))
    .map(([g, l]) => `<li><i style="background:var(--g-${g})"></i>${l}</li>`).join("");
}

function renderList() {
  $("#nodelist").innerHTML = `<table><thead><tr><th>Entity</th><th>Type</th><th>Description</th><th>Links</th></tr></thead><tbody>${
    G.nodes.map((n) => `<tr><td>${esc(n.id)}</td><td>${esc(n.type)}</td><td>${esc(n.desc)}</td><td>${G.adj[n.i].length}</td></tr>`).join("")}</tbody></table>`;
}

// ---------------------------------------------------------------------------------------------- the walk
function setStep(k, text) {
  document.querySelectorAll("#steps li").forEach((li) => {
    const s = +li.dataset.step;
    li.classList.toggle("on", s <= k);
    li.classList.toggle("now", s === k);
  });
  if (text !== undefined) $(`#n${k}`).textContent = text;
}

function focusOn(nodeIdx) {
  if (G.nodes.length <= 350) return;                       // small graphs stay whole
  const show = new Set(nodeIdx);
  for (const n of nodeIdx) for (const v of G.adj[n].slice(0, 6)) show.add(v);
  cy.nodes().forEach((el) => el.style("display", show.has(+el.id().slice(1)) ? "element" : "none"));
  layout(cy.elements(":visible"));
}

function highlight(rank) {
  if (!lastResult) return;
  const top = lastResult.top;
  cy.elements().removeClass("onpath other").addClass("faded");
  top.forEach((t, j) => {
    const cls = j === rank ? "onpath" : "other";
    t.path.forEach((n, i) => {
      cy.$(`#n${n}`).removeClass("faded").addClass(j === rank ? "onpath" : "");
      if (i) cy.$(`#e${edgeIndex(G, t.path[i - 1], n)}`).removeClass("faded").addClass(cls);
    });
  });
  document.querySelectorAll(".path").forEach((el, j) => el.classList.toggle("sel", j === rank));
  const ids = top[rank].path.map((n) => `#n${n}`).join(",");
  cy.animate({ fit: { eles: cy.$(ids), padding: 70 } }, { duration: 450 });
}

function renderPaths(r) {
  if (!r.top.length) {
    $("#paths").innerHTML = `<p class="muted">No path within 4 hops connects these anchors to a relevant answer. Try naming a project, a method or a physics concept.</p>`;
    return;
  }
  $("#paths").innerHTML = r.top.map((t, j) => {
    const hops = verbalize(G, t.path);
    const pill = (name) => { const n = G.nodes[G.byName.get(norm(name))];
      return `<span class="pill"><i style="background:var(--g-${n.group})"></i>${esc(name)}</span>`; };
    return `<button type="button" class="path" data-rank="${j}">
      <div class="rank"><span>Path ${j + 1} · ${t.path.length - 1} hop${t.path.length > 2 ? "s" : ""}</span><span>score ${t.score.toFixed(3)}</span></div>
      ${hops.map((h, i) => `<p class="hop">${i === 0 ? pill(h.from) : ""} → ${pill(h.to)}<br>${esc(h.desc)}
        <br><span class="src">source: ${esc(h.docs.join(", "))}</span></p>`).join("")}
    </button>`;
  }).join("");
  document.querySelectorAll(".path").forEach((el) => el.addEventListener("click", () => highlight(+el.dataset.rank)));
}

async function run(query) {
  query = query.trim();
  if (!query || !G) return;
  lastQuery = query;
  $("#go").disabled = true;
  $("#replay").disabled = true;
  const prepared = prepareQuery(query);
  const sc = await scorer(prepared);
  const r = search(G, prepared, sc);
  lastResult = r;
  await animate(r);
  $("#go").disabled = false;
  $("#replay").disabled = false;
}

async function animate(r) {
  status(`Searched ${G.nodes.length} entities in ${r.ms} ms${modelState === "fallback" ? " (keyword scoring)" : ""}.`);
  $("#paths").innerHTML = "";
  for (let k = 1; k <= 4; k++) $(`#n${k}`).textContent = "";
  cy.elements().removeClass("anchor answer mid onpath other faded");
  focusOn([...r.anchors, ...r.answers.map((a) => a.n), ...r.top.flatMap((t) => t.path)]);

  setStep(1, `${r.anchors.length} anchor${r.anchors.length === 1 ? "" : "s"}: ${r.anchors.map((a) => G.nodes[a].id).join(", ")}`);
  cy.elements().addClass("faded");
  r.anchors.forEach((a) => cy.$(`#n${a}`).removeClass("faded").addClass("anchor"));
  await sleep(900);

  setStep(2, `${r.reachable - r.anchors.length} reachable → ${r.answers.length} candidates`);
  for (const a of r.answers) { cy.$(`#n${a.n}`).removeClass("faded").addClass("answer"); await sleep(40); }
  await sleep(600);

  const mids = new Set(r.perDepth.flatMap((d) => d.midpoints));
  const fwd = r.perDepth.reduce((s, d) => s + d.forward, 0), bwd = r.perDepth.reduce((s, d) => s + d.backward, 0);
  setStep(3, `${fwd} forward + ${bwd} backward halves → ${r.candidates} joined paths`);
  mids.forEach((m) => { if (!r.anchors.includes(m)) cy.$(`#n${m}`).removeClass("faded").addClass("mid"); });
  await sleep(900);

  setStep(4, r.top.length ? `top ${r.top.length} of ${r.candidates}` : "no connecting path");
  renderPaths(r);
  if (r.top.length) highlight(0);
}

// ---------------------------------------------------------------------------------------------- wiring
$("#ask").addEventListener("submit", (e) => { e.preventDefault(); run($("#q").value); });
$("#chips").addEventListener("click", (e) => {
  if (e.target.classList.contains("chip")) { $("#q").value = e.target.textContent; run(e.target.textContent); }
});
$("#replay").addEventListener("click", () => lastResult && animate(lastResult));
$("#fit").addEventListener("click", () => cy && cy.animate({ fit: { eles: cy.elements(":visible"), padding: 30 } }));
$("#listview").addEventListener("click", (e) => {
  const box = $("#nodelist"), open = box.hidden;
  box.hidden = !open;
  e.target.setAttribute("aria-expanded", String(open));
  e.target.textContent = open ? "Hide list" : "List view";
});

let modelReady;
(async function main() {
  try {
    const res = await fetch("data/graph.json");
    G = loadGraph(await res.json());
  } catch (e) {
    $("#graph-meta").textContent = "The graph failed to load. Please refresh.";
    return;
  }
  const m = G.meta;
  $("#graph-meta").textContent = `${m.source} · ${m.nodes} entities · ${m.edges} relations · built ${m.built}. ` +
    `Ask anything about ${OWNER.node.split(" ")[0]}'s projects, work or physics.`;
  renderLegend();
  renderList();
  initCy();
  modelReady = loadModel();
  status("Loading the in-browser embedding model (≈23 MB, once)… you can ask right away.");
})();
