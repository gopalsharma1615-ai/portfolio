# Gopal Sharma: portfolio with a live knowledge-graph demo

A static site (no build step) whose centrepiece is an interactive knowledge graph of my work. Visitors ask a
question and watch the same **bidirectional meet-in-the-middle search** as
[graphwalk-rag](https://github.com/gopalsharma1615-ai/graphwalk-rag) find the top-5 evidence paths:

1. **Anchor:** find the entities the question names.
2. **Answer candidates:** rank the anchors' reachable set.
3. **Meet in the middle:** grow half-paths from both ends and join them.
4. **Rank:** score the joined paths and keep the top 5.

Everything runs in the visitor's browser. A 23 MB embedding model (`Xenova/all-MiniLM-L6-v2` via transformers.js)
embeds the question, and the search itself is plain JavaScript in `site/graph.js`, typically 1–3 ms. There is no
backend to keep alive, and nothing a visitor types leaves their browser.

```
corpus/   what the graph is built from: thesis, project READMEs, experience, skills (no client names)
build/    build_graph.sh (Qwen3 via graphwalk-rag), export_site_graph.py, curated starter graph
site/     the website: index.html, styles.css, app.js, graph.js, data/graph.json, assets/
tests/    node --test: meet-in-the-middle == brute force; real queries give 5 valid paths
```

## Deploy

- **Netlify:** "Add new site → Import from Git", pick this repo. `netlify.toml` publishes `site/`, with no build
  command.
- **Vercel:** "Add New → Project", import this repo. `vercel.json` serves `site/`, with no build.
- Either redeploys automatically on every push.

## The graph: Qwen3-14B extraction on a curated backbone

`site/data/graph.json` combines two sources:
- **Breadth:** Qwen3-14B (vLLM) extracted entities and relations from `corpus/` through graphwalk-rag/LightRAG
  (9 documents, 53 chunks).
- **Backbone:** a small curated graph (`build/curated.py`, 67 entities) that gives career and project questions
  clean answers.

The exporter resolves entities across both sources:
- It merges names that differ only in case or punctuation, plus known synonyms, so "Multi-hop QA" and
  "multi-hop question answering" become one node.
- It folds name variants into one owner node.
- It drops skill-list headings, file names and one-word generic nouns. Left in, these become hubs that connect
  everything to everything.

The result has 892 entities and 1,381 relations. To rebuild:

```bash
pip install "graphwalk-rag[build,local] @ git+https://github.com/gopalsharma1615-ai/graphwalk-rag"
export LLM_BASE_URL=http://<qwen-host>:8077/v1        # Qwen3-14B on vLLM, or any OpenAI-compatible server
bash build/build_graph.sh --dry-run                   # chunk count and token estimate, no LLM calls
bash build/build_graph.sh                              # extract -> verify provenance -> site/data/graph.json
```

The build resumes from where it stopped, and runs 4 parallel calls by default (`LLM_ASYNC`), matching vLLM's
`--max-num-seqs`. Output is capped at 4,096 tokens per call (`LLM_MAX_TOKENS`), so a looping generation can't
stall the build. The exporter re-embeds everything with all-MiniLM-L6-v2, the model the browser runs. Graphs
over 350 nodes are shown as a focused subgraph around each query.

## Local preview and tests

```bash
python -m http.server -d site 8000                     # open http://localhost:8000
node --test tests/graph.test.mjs                        # algorithm tests
```
