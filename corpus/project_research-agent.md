# research-agent *(design stage: not implemented yet)*

An agent that answers research questions over private documents **with citations**, built as an MCP client of
[rag-mcp-server](https://github.com/gopalsharma1615-ai/rag-mcp-server) and
[qft-tools](https://github.com/gopalsharma1615-ai/qft-renorm-ai). It plans, retrieves, verifies, and only then
answers.

## Intended design

```
question ─► plan: classify (lookup | multi-hop | long-document | calculation) and estimate hops
         ─► act:  MCP tool calls: search_documents · graph_walk · guided_walk · get_entity / get_neighbors · CAS tools
         ─► verify: every claim must map to a returned chunk or fact chain; unsupported claims are dropped
         ─► answer with citations (document, page, section)  |  or abstain
```

Default policy: **single-first**. Answer from one retrieval, and decompose only on abstention. Measured with
graphwalk-rag on 952 HotpotQA questions, that took accuracy from 70.2% to 76.9%, with decomposition on only 18%
of questions. Always decomposing scored 62.8%.

## TODO

- [ ] Agent loop on an OpenAI-compatible chat API with MCP tool discovery (stdio and streamable HTTP)
- [ ] Planner: question-type and hop-count classifier; routing table to tools
- [ ] Single-first policy with abstention detection; decomposition as a rescue step
- [ ] Claim-to-evidence verifier: an NLI or cross-encoder entailment check per sentence, dropping unsupported claims
- [ ] Citations rendered as (document, page, section) with the supporting snippet
- [ ] Budget and safety: per-run token/$ cap (openai-budget-guard), tool-call limit, timeouts
- [ ] Evaluation: HotpotQA multi-hop and the regulatory question set, measuring answer accuracy, citation
      precision/recall, tool calls per question, p50/p95 latency and $/question against fixed pipelines
- [ ] Traces: every run saved as JSONL (plan, tool calls, evidence, verdicts) for error analysis

MIT licence. Built by [Gopal Sharma](https://github.com/gopalsharma1615-ai).
