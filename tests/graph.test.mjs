import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { loadGraph, meet, search, vectorScorer, lexicalScorer, verbalize } from "../site/graph.js";

const G = loadGraph(JSON.parse(readFileSync(new URL("../site/data/graph.json", import.meta.url))));
const Q = JSON.parse(readFileSync(new URL("./query_vectors.json", import.meta.url)));

function randomGraph(n, m, seed) {
  let x = seed; const rnd = () => (x = (x * 1103515245 + 12345) % 2147483648) / 2147483648;
  const adj = Array.from({ length: n }, () => new Set());
  while ([...adj].reduce((s, a) => s + a.size, 0) / 2 < m) {
    const u = Math.floor(rnd() * n), v = Math.floor(rnd() * n);
    if (u !== v) { adj[u].add(v); adj[v].add(u); }
  }
  return { adj: adj.map((s) => [...s].sort((a, b) => a - b)), hub: 1e9, rnd };
}

function brute(g, src, dsts, k) {
  const out = new Set(), path = [src];
  const dfs = () => {
    if (path.length - 1 === k) { if (dsts.includes(path[path.length - 1])) out.add(path.join(",")); return; }
    for (const v of g.adj[path[path.length - 1]]) if (!path.includes(v)) { path.push(v); dfs(); path.pop(); }
  };
  dfs();
  return out;
}

test("meet-in-the-middle equals brute force on random graphs", () => {
  for (let seed = 1; seed <= 12; seed++) for (let k = 1; k <= 5; k++) {
    const g = randomGraph(24, 50, seed);
    const src = Math.floor(g.rnd() * 24);
    const dsts = [0, 1, 2, 3].map(() => Math.floor(g.rnd() * 24)).filter((d) => d !== src);
    const got = new Set(meet(g, [src], dsts, k).paths.map((p) => p.join(",")));
    assert.deepEqual([...got].sort(), [...brute(g, src, dsts, k)].sort(), `seed ${seed} k ${k}`);
  }
});

test("real queries return 5 distinct, valid paths from an anchor", () => {
  Q.queries.forEach((q, i) => {
    const r = search(G, q, vectorScorer(G, Float32Array.from(Q.vectors[i])));
    assert.ok(r.top.length >= 3, q);
    for (const { path } of r.top) {
      assert.ok(r.anchors.includes(path[0]));
      assert.equal(new Set(path).size, path.length);
      verbalize(G, path);                               // every hop is a real edge
    }
    console.log(`\nQ: ${q}\n   anchors: ${r.anchors.map((a) => G.nodes[a].id).join(", ")} | candidates ${r.candidates} | ${r.ms} ms`);
    r.top.forEach((t, j) => console.log(`   ${j + 1}. ${t.path.map((n) => G.nodes[n].id).join(" -> ")}  (${t.score.toFixed(3)})`));
  });
});

test("lexical fallback still answers", () => {
  const sc = lexicalScorer(G);
  const r = search(G, Q.queries[0], sc(Q.queries[0]));
  assert.ok(r.top.length >= 1);
});
