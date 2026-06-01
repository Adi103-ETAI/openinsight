# DeepInsight — Agent Model Assignment Map
# Project: OpenInsight / SentArc Labs
# Last updated: 2026-06

## Design Principles
- Each agent picks the model optimised for its latency/accuracy tradeoff
- No agent is locked to one provider — fallbacks are defined
- Dynamic routing via `providers/openai_compatible.py` as the universal adapter
- NVIDIA NIM remains the primary for all heavy clinical reasoning (no external data leaves the stack)

---

## Agent → Model → Provider Matrix

| Agent | Primary Model | Provider | Fallback Model | Fallback Provider | Why |
|---|---|---|---|---|---|
| Orchestrator | `meta/llama-3.1-8b-instruct` | NVIDIA NIM | `gemini-2.0-flash` | Google | Fast intent classification; 8B is enough for routing, NIM gives low latency on your Codespaces setup |
| RAG Agent | `meta/llama-3.1-70b-instruct` | NVIDIA NIM | `meta/llama-3.1-70b-instruct` | Groq | Clinical synthesis needs the full 70B; already validated in Phase 1 |
| Web Search Agent | `gemini-2.0-flash` | Google | `gpt-4o-mini` | OpenAI | 1M context window handles long web pages; native tool use; fastest grounding for live retrieval |
| DocGen Agent | `claude-haiku-4-5-20251001` | Anthropic | `gpt-4o-mini` | OpenAI | Best structured output + JSON schema enforcement at low cost; formats synthesis result into DOCX/PDF template |
| Synthesis Agent | `meta/llama-3.1-70b-instruct` | NVIDIA NIM | `command-r-plus` | Cohere | Multi-source merging needs the same clinical-grade model as RAG; Cohere R+ is strong at document synthesis as fallback |
| Citation Validator | `claude-haiku-4-5-20251001` | Anthropic | `gpt-4o-mini` | OpenAI | Deterministic JSON extraction; Haiku's structured output guarantees valid citation schema on first attempt |

---

## Provider File Mapping

```
providers/
├── nvidia.py          → Orchestrator (primary), RAG Agent (primary), Synthesis Agent (primary)
├── google.py          → Web Search Agent (primary), Orchestrator (fallback)
├── anthropic.py       → DocGen Agent (primary), Citation Validator (primary)
├── openai.py          → Web Search Agent (fallback), DocGen Agent (fallback), Citation Validator (fallback)
├── cohere.py          → Synthesis Agent (fallback)
├── ollama.py          → Local dev/testing fallback for any agent (Llama 3.1 8B local)
└── openai_compatible.py → Universal adapter; used when routing to Groq, Together, Fireworks
```

---

## Model Spec Details

### Orchestrator — `meta/llama-3.1-8b-instruct` via NVIDIA NIM
- Task: single-pass intent classification → route decision (RAG | WEB | DOCGEN | BOTH | ALL)
- Why 8B not 70B: routing is a ~200-token classification task, overkill to use 70B here
- Why NIM: already in stack, keeps latency inside the same inference endpoint
- Context needed: ≤ 512 tokens (query + routing schema)
- Output: structured JSON `{ "intent": [...], "agents": [...], "priority": "speed|quality" }`

### RAG Agent — `meta/llama-3.1-70b-instruct` via NVIDIA NIM
- Task: answer generation from retrieved medical chunks (ICMR guidelines, PubMed, corpus)
- Why 70B: clinical accuracy, hallucination resistance, long context (128K for multi-chunk input)
- System prompt: loaded from `prompts/rag_agent.md` at runtime (Markdown, per Phase 1 design)
- Context budget: up to 32K tokens of retrieved chunks + query
- Output: answer string + `source_ids[]` list for citation validator

### Web Search Agent — `gemini-2.0-flash` via Google
- Task: live web retrieval, page summarisation, grounding queries outside the corpus
- Why Gemini Flash: 1M context handles full page dumps; native tool use; fastest first-token latency (~0.53s)
- Triggers: RAG confidence < threshold OR query contains temporal signals ("latest", "new", "2025")
- Output: `{ "summary": str, "sources": [url, title, snippet], "retrieved_at": timestamp }`
- Note: Vercel AI SDK browser agent can replace this entirely for full browser automation

### DocGen Agent — `claude-haiku-4-5-20251001` via Anthropic
- Task: convert synthesis output into structured DOCX or PDF via `python-docx` / `reportlab`
- Why Haiku 4.5: structured outputs are GA; fastest JSON schema enforcement in the Anthropic line; 200K context handles long reports
- Input: synthesis agent output + user format preference (DOCX | PDF)
- Output: populates document template → calls `tools/docgen_tool.py` → returns file path
- Does NOT re-generate content — only formats what synthesis produces

### Synthesis Agent — `meta/llama-3.1-70b-instruct` via NVIDIA NIM
- Task: merge RAG output + Web Search output when both fire; deduplicate; reconcile conflicts
- Why 70B: same clinical model as RAG — consistent voice and factual standards across both halves
- Triggers: both RAG and Web Search agents return results
- Context: RAG answer + web summary + original query
- Output: unified answer with conflict flags if RAG and web disagree

### Citation Validator — `claude-haiku-4-5-20251001` via Anthropic
- Task: map every claim in the final answer to a source chunk ID; output citation schema
- Why Haiku 4.5: deterministic structured output; citations require exact JSON, not prose; fast
- Input: final answer text + `source_ids[]` from RAG + `sources[]` from web search
- Output schema:
  ```json
  {
    "citations": [
      { "claim_text": str, "source_id": str, "source_type": "corpus|web", "confidence": float }
    ]
  }
  ```
- Runs AFTER synthesis, BEFORE final response delivery

---

## Parallel Execution Pattern

```
User Query
    │
    ▼
Orchestrator (NIM 8B) — 1 LLM call, ~200ms
    │
    ├──[RAG]──────► NIM 70B  ─┐
    │                          ├──► Synthesis Agent (NIM 70B)
    └──[WEB]──────► Gemini Flash ─┘
                                    │
                              Citation Validator (Haiku 4.5)
                                    │
                              DocGen Agent (Haiku 4.5)  ← only if format=doc
                                    │
                              Final Response
```

RAG + Web fire in parallel via `asyncio.gather`. Synthesis only runs if both return results.
If only RAG returns → skip synthesis, go straight to citation validator.

---

## Cost Estimates (per query, rough)

| Path | Models Called | Est. Input Tokens | Est. Cost |
|---|---|---|---|
| RAG-only | Orchestrator + RAG + Citation | ~4K | ~$0.004 (NIM) + ~$0.001 (Haiku) |
| RAG + Web | All 5 agents | ~8K | ~$0.008 (NIM) + ~$0.002 (Gemini) + ~$0.002 (Haiku) |
| RAG + Web + DocGen | All 6 agents | ~10K | ~$0.012 total |

NIM pricing assumed at self-hosted / free tier on Codespaces. Adjust for production GPU billing.
