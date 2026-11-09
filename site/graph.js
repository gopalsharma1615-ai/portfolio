// GraphWalk in the browser: the same algorithm as github.com/gopalsharma1615-ai/graphwalk-rag, sized for a
// portfolio graph. Pure functions, no DOM - the UI animates the `trace` this returns.
//
//   anchors  -> named things in the question (string match) + nearest entities (embedding)
//   answers  -> the anchors' hub-guarded reachable set, ranked by similarity to the question, per distance shell
//   meet     -> forward half-paths from anchors ⋈ backward half-paths from answers, joined at a shared midpoint
//   rank     -> mean relation relevance + answer relevance - length penalty; top 5 with distinct endpoints

export const GROUPS = {
  Person: "career", Organization: "career", Role: "career", Education: "career", Achievement: "career",
  Project: "project",
  Algorithm: "method", Method: "method",
  Technology: "technology",
  PhysicsConcept: "physics", Process: "physics",
  Result: "result", Metric: "result", Dataset: "result",
};
export const GROUP_LABELS = { career: "Career", project: "Project", method: "Method", technology: "Technology",
  physics: "Physics", result: "Result & data" };

function b64ToInt8(b64) {
  const bin = atob(b64);
  const out = new Int8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = (bin.charCodeAt(i) << 24) >> 24;
  return out;
}

function unitRows(i8, dim) {
  const n = i8.length / dim, out = new Float32Array(i8.length);
  for (let r = 0; r < n; r++) {
    let s = 0;
    for (let k = 0; k < dim; k++) { const v = i8[r * dim + k] / 127; out[r * dim + k] = v; s += v * v; }
    s = Math.sqrt(s) || 1;
    for (let k = 0; k < dim; k++) out[r * dim + k] /= s;
  }
  return out;
}

export const norm = (s) => s.toLowerCase().normalize("NFKD").replace(/[^a-z0-9+$%.]+/g, " ").trim();

export function loadGraph(json) {
  const { meta, docs } = json;
  const dim = meta.dim;
  const nodes = json.nodes.map(([id, type, desc, d], i) =>
    ({ i, id, type, group: GROUPS[type] || "technology", desc, docs: d.map((x) => docs[x]) }));
  const edges = json.edges.map(([s, t, desc, d], i) => ({ i, s, t, desc, docs: d.map((x) => docs[x]) }));
  const adj = nodes.map(() => []);
  const edgeOf = new Map();
  for (const e of edges) {
    adj[e.s].push(e.t); adj[e.t].push(e.s);
    edgeOf.set(e.s < e.t ? `${e.s}|${e.t}` : `${e.t}|${e.s}`, e.i);
  }
  for (const a of adj) a.sort((x, y) => x - y);
  const ent = unitRows(b64ToInt8(json.ent_vec), dim);
  const rel = unitRows(b64ToInt8(json.rel_vec), dim);
  const byName = new Map(nodes.map((n) => [norm(n.id), n.i]));
  return { meta, nodes, edges, adj, edgeOf, ent, rel, dim, byName, hub: meta.hub_degree };
}

const dot = (M, row, q, dim) => { let s = 0; for (let k = 0; k < dim; k++) s += M[row * dim + k] * q[k]; return s; };
export const edgeIndex = (G, u, v) => G.edgeOf.get(u < v ? `${u}|${v}` : `${v}|${u}`);
const isHub = (G, n) => G.adj[n].length > G.hub || (G.blocked !== undefined && G.blocked.has(n));

// Lexical fallback when the embedding model cannot load (offline, blocked CDN): bag-of-words cosine.
export function lexicalScorer(G) {
  const toks = (s) => new Set(norm(s).split(" ").filter((w) => w.length > 2));
  const ents = G.nodes.map((n) => toks(`${n.id} ${n.desc}`));
  const rels = G.edges.map((e) => toks(`${G.nodes[e.s].id} ${e.desc} ${G.nodes[e.t].id}`));
  const cos = (a, b) => { let c = 0; for (const w of a) if (b.has(w)) c++; return c / Math.sqrt((a.size || 1) * (b.size || 1)); };
  return (query) => { const q = toks(query); return { ent: (i) => cos(q, ents[i]), rel: (i) => cos(q, rels[i]) }; };
}

export function vectorScorer(G, qvec) {
  return { ent: (i) => dot(G.ent, i, qvec, G.dim), rel: (i) => dot(G.rel, i, qvec, G.dim) };
}

export function findAnchors(G, query, score, maxAnchors = 3) {
  const q = ` ${norm(query)} `;
  const hits = [];
  for (const [name, i] of G.byName) {
    if (name.length >= 3 && q.includes(` ${name} `)) hits.push([name.length, i]);
    else if (G.nodes[i].type === "Person") {             // "Gopal" alone names "Gopal Sharma"
      const first = name.split(" ")[0];
      if (first.length >= 4 && q.includes(` ${first} `)) hits.push([first.length, i]);
    }
  }
  hits.sort((a, b) => b[0] - a[0]);
  const out = [];
  for (const [, i] of hits) if (!out.includes(i)) out.push(i);
  const dense = G.nodes.map((n) => [score.ent(n.i), n.i]).sort((a, b) => b[0] - a[0]);
  for (const [s, i] of dense) {
    if (out.length >= maxAnchors) break;
    if (!out.includes(i) && (s >= 0.35 || out.length === 0)) out.push(i);
  }
  return { anchors: out.slice(0, maxAnchors), named: hits.length };
}

export function reach(G, sources, k) {
  const dist = new Map(sources.map((s) => [s, 0]));
  const queue = [...sources];
  while (queue.length) {
    const u = queue.shift();
    if (dist.get(u) >= k) continue;
    for (const v of G.adj[u]) if (!dist.has(v) && !isHub(G, v)) { dist.set(v, dist.get(u) + 1); queue.push(v); }
  }
  return dist;
}

function halves(G, src, k, allowed, cap = 20000) {
  const out = [], path = [src], on = new Set([src]);
  const dfs = (depth) => {
    if (out.length >= cap) return;
    if (depth === k) { out.push(path.slice()); return; }
    for (const v of G.adj[path[path.length - 1]]) {
      if (on.has(v) || isHub(G, v) || (allowed && !allowed(v, k - depth - 1))) continue;
      path.push(v); on.add(v); dfs(depth + 1); path.pop(); on.delete(v);
    }
  };
  dfs(0);
  return out;
}

// Meet-in-the-middle: simple paths of exactly k edges from any anchor to any answer, distance-pruned both ways.
export function meet(G, anchors, answers, k) {
  const kf = Math.ceil(k / 2), kb = Math.floor(k / 2);
  const zone = reach(G, answers, kb);
  const toZone = reach(G, [...zone.keys()], kf);
  const fwd = [];
  for (const a of anchors) for (const p of halves(G, a, kf, (v, left) => (toZone.get(v) ?? 1e9) <= left))
    if ((zone.get(p[p.length - 1]) ?? 1e9) <= kb) fwd.push(p);
  const mids = new Set(fwd.map((p) => p[p.length - 1]));
  const toMid = reach(G, [...mids], kb);
  const back = new Map();
  for (const d of answers) for (const p of halves(G, d, kb, (v, left) => (toMid.get(v) ?? 1e9) <= left)) {
    const m = p[p.length - 1];
    if (mids.has(m)) { if (!back.has(m)) back.set(m, []); back.get(m).push(p); }
  }
  const out = [], seen = new Set();
  for (const pf of fwd) {
    const fset = new Set(pf);
    for (const pb of back.get(pf[pf.length - 1]) || []) {
      const tail = pb.slice(0, -1).reverse();
      if (tail.some((x) => fset.has(x))) continue;
      const full = pf.concat(tail), key = full.join(",");
      if (!seen.has(key) && new Set(full).size === full.length) { seen.add(key); out.push(full); }
    }
  }
  return { paths: out, forward: fwd.length, midpoints: [...mids], backward: [...back.values()].flat().length };
}

// Every question on the site is about the owner, so the owner's name carries no information: strip it (and
// pronouns) before anchoring and embedding. The person node anchors only when nothing else does.
export const OWNER = { node: "Gopal Sharma", aliases: ["gopal sharma", "gopal", "sharma", "his", "him", "he", "gopal's"] };

export function prepareQuery(query, owner = OWNER) {
  let q = ` ${query.toLowerCase()} `;
  for (const a of [...owner.aliases].sort((x, y) => y.length - x.length))
    q = q.replace(new RegExp(`(^|[^a-z])${a.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}(?=[^a-z]|$)`, "g"), "$1 ");
  q = q.replace(/\s+/g, " ").trim();
  return q.length >= 3 ? q : query;
}

export function search(G, query, score, opts = {}) {
  const t0 = typeof performance !== "undefined" ? performance.now() : Date.now();
  const maxHops = opts.maxHops ?? 4, perShell = opts.perShell ?? 6, topAnswers = opts.answers ?? 12;
  const ownerIdx = G.byName.get(norm(opts.owner ?? OWNER.node));
  let { anchors, named } = findAnchors(G, query, score);
  if (ownerIdx !== undefined && anchors.length > 1) anchors = anchors.filter((a) => a !== ownerIdx);
  // The owner links to almost everything, so like a hub it is never walked through or offered as an answer
  // (it can still START a walk when it is the only anchor).
  G.blocked = new Set(ownerIdx !== undefined ? [ownerIdx] : []);
  const dist = reach(G, anchors, maxHops);
  const pool = [...dist.entries()].filter(([n, d]) => d >= 1 && !anchors.includes(n))
    .map(([n, d]) => ({ n, d, s: score.ent(n) })).sort((a, b) => b.s - a.s);
  const chosen = new Set(pool.slice(0, topAnswers).map((x) => x.n));
  for (let d = 1; d <= maxHops; d++) pool.filter((x) => x.d === d).slice(0, perShell).forEach((x) => chosen.add(x.n));
  const answers = pool.filter((x) => chosen.has(x.n));
  const perDepth = [];
  let paths = [];
  for (let k = 1; k <= maxHops; k++) {
    const r = meet(G, anchors, answers.filter((x) => x.d <= k).map((x) => x.n), k);
    perDepth.push({ k, forward: r.forward, backward: r.backward, joined: r.paths.length, midpoints: r.midpoints });
    paths = paths.concat(r.paths);
  }
  const ans = new Map(answers.map((x) => [x.n, x.s]));
  const scored = paths.map((p) => {
    let rel = 0;
    for (let j = 0; j + 1 < p.length; j++) rel += score.rel(edgeIndex(G, p[j], p[j + 1]));
    rel /= p.length - 1;
    return { path: p, score: 0.6 * rel + 0.4 * (ans.get(p[p.length - 1]) ?? 0) - 0.04 * (p.length - 2) };
  }).sort((a, b) => b.score - a.score);
  const top = [], perEnd = new Map();
  for (const s of scored) {
    const end = s.path[s.path.length - 1];
    if ((perEnd.get(end) || 0) >= 1) continue;
    perEnd.set(end, 1); top.push(s);
    if (top.length === (opts.top ?? 5)) break;
  }
  const t1 = typeof performance !== "undefined" ? performance.now() : Date.now();
  return { anchors, named, reachable: dist.size, answers, perDepth, candidates: paths.length, top,
           ms: Math.round((t1 - t0) * 10) / 10 };
}

export function verbalize(G, path) {
  return path.slice(0, -1).map((u, j) => {
    const e = G.edges[edgeIndex(G, u, path[j + 1])];
    return { from: G.nodes[u].id, to: G.nodes[path[j + 1]].id, desc: e.desc, docs: e.docs };
  });
}
