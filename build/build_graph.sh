#!/usr/bin/env bash
# Build the portfolio knowledge graph with your own LLM (Qwen3-14B on vLLM), then export it for the website.
#
#   export LLM_BASE_URL=http://<qwen-host>:8077/v1      # any OpenAI-compatible endpoint
#   bash build/build_graph.sh            # from the repo root; add --dry-run to only price the build
#
# Needs: pip install "graphwalk-rag[build,local] @ git+https://github.com/gopalsharma1615-ai/graphwalk-rag"
set -euo pipefail
cd "$(dirname "$0")/.."
: "${LLM_BASE_URL:?set LLM_BASE_URL to your OpenAI-compatible endpoint, e.g. http://<qwen-host>:8077/v1}"
export LLM_MODEL="${LLM_MODEL:-/disk1/models/Qwen3-14B}"
export LLM_API_KEY="${LLM_API_KEY:-EMPTY}"
export LLM_EXTRA_BODY="${LLM_EXTRA_BODY:-{\"chat_template_kwargs\": {\"enable_thinking\": false}}}"
EMBED="local:sentence-transformers/all-MiniLM-L6-v2"   # the same model the browser runs

graphwalk-rag build corpus/ --workdir build/graph_work --model "$LLM_MODEL" \
  --embed-model "$EMBED" --embed-dim 384 --entity-types build/entity_types.txt --chunk-tokens 600 \
  --llm-async "${LLM_ASYNC:-4}" --llm-timeout "${LLM_TIMEOUT:-600}" "$@"   # match vLLM --max-num-seqs
if [[ " $* " == *" --dry-run "* ]]; then exit 0; fi
graphwalk-rag verify --workdir build/graph_work
# curated backbone (build/curated_graph.json) + Qwen-extracted breadth, entities resolved across both
python build/export_site_graph.py --curated build/curated_graph.json --lightrag build/graph_work \
  --out site/data/graph.json --source "Qwen3-14B graph + curated backbone"
echo "done: site/data/graph.json (commit and push it; Netlify/Vercel redeploys)"
