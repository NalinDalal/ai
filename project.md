# Projects

## Foundations (Small Builds)

| # | Project | Core Skill |
|---|---------|------------|
| 1 | Context Assembler | Token budgeting, memory, retrieval |
| 2 | Retrieval Stack | Chunker, BM25, dense search, reranker |
| 3 | Model Router | Cost/latency/quality routing, fallbacks |
| 4 | Semantic Cache | Embedding similarity, hit-rate tracking |
| 5 | Agent Orchestrator | Deterministic state machine (no LangChain) |
| 6 | MCP Server & Client | Raw JSON-RPC, no SDK |
| 7 | Multi-Agent Consensus | Weighted voting, judge, escalation |
| 8 | Sandboxed Tool Executor | Isolated execution, resource limits |
| 9 | Guardrails Middleware | Injection detection, PII redaction |
| 10 | Durable Workflow Engine | Checkpoint/resume, mini-Temporal |
| 11 | Streaming Proxy | SSE, TTFT and ITL metrics |
| 12 | LLM Tracer | OpenTelemetry-style spans |
| 13 | Eval Harness | Trajectory grading, CI regression gates |
| 14 | Prompt Registry | Versioning, A/B routing, rollback |
| 15 | Data Flywheel | Feedback → synthetic data → LoRA loop |

## Production AI Systems

| # | Project | Core Skill |
|---|---------|------------|
| 1 | Production RAG Pipeline | Ingestion, chunking, hybrid search, reranking, cited answers over 1000+ docs |
| 2 | Structured Output Engine | Pydantic schema enforcement with retries, validation, graceful fallbacks on malformed JSON |
| 3 | Context Assembly Service | Dynamic context builder that budgets tokens across memory, docs & tools per request |
| 4 | LLM Evaluation Harness | Golden dataset + LLM-as-a-judge + CI gate that blocks merges on quality regression |
| 5 | Semantic Cache Layer | Embedding-based cache returning stored answers for similar queries, with hit-rate tracking |
| 6 | Model Routing Gateway | Complexity-based router across 3 model tiers with fallbacks and per-request cost tracking |
| 7 | Multi-Tenant LLM API | Per-tenant keys, rate limits, token budgets, isolated data access |
| 8 | Fine-Tuning Pipeline | LoRA/QLoRA fine-tune of a small model on synthetic data, benchmarked vs prompt-only baseline |
| 9 | Agent Memory System | Working + episodic + semantic memory with compression and eviction policies |
| 10 | Guardrails Middleware | Input/output filtering, PII redaction, injection detection as a reusable service layer |
| 11 | Streaming Response Infrastructure | SSE streaming with backpressure, reconnects, time-to-first-token tracking |
| 12 | Prompt Versioning and A/B System | Registry for prompts and configs with rollback, traffic splitting, outcome tracking |
| 13 | LLM Observability Stack | Tracing for prompts, tokens, latency, cost per request with alerting on anomalies |
| 14 | Tool-Calling Framework | Typed function schemas with discovery, retries, sandboxed execution |
| 15 | Self-Correcting RAG Agent | Query rewriting, retrieval critique, confidence-based fallback to web search |

---

## Capstone Projects

| # | Project | Status |
|---|---------|--------|
| 1 | Autonomous Ticket Resolution Engine | todo |
| 2 | Deep Research Agent with Citation Graph | todo |
| 3 | Self-Healing Data Pipeline Agent | todo |
| 4 | CI Triage Agent | todo |
| 5 | Multi-Agent Code Review Desk | todo |
| 6 | Computer-Use Back-Office Agent | todo |
| 7 | Invoice Processing Agent with 3-Way Match | todo |
| 8 | Incident Response Agent | todo |
| 9 | Real-Time Voice Ops Agent | todo |
| 10 | Adaptive Tutor Agent with Mastery Memory | todo |
| 11 | Agentic Sales Ops Assistant | todo |
| 12 | Agent Eval and Regression Platform | todo |

### Capstone Details

**1. Autonomous Ticket Resolution Engine**
Agent that reads the ticket, queries the database, applies the fix, and asks a human before anything destructive.
*Shows: You build agents that resolve, not just reply.*

**2. Deep Research Agent with Citation Graph**
Multi-hop research, source grading, contradiction detection, fully cited final report.
*Shows: You can orchestrate long-horizon reasoning safely.*

**3. Self-Healing Data Pipeline Agent**
Detects schema drift in ETL jobs, drafts transformation fixes, re-runs with rollback.
*Shows: You trust agents with production data, carefully.*

**4. CI Triage Agent**
Reads failing pipeline logs, reproduces the error, opens a fix PR with tests, waits for approval.
*Shows: You can embed agents into engineering workflows.*

**5. Multi-Agent Code Review Desk**
Reviewer + security scanner + test writer agents with consensus and conflict resolution.
*Shows: You orchestrate teams of agents, not toys.*

**6. Computer-Use Back-Office Agent**
Browser automation for legacy portals: forms, uploads, extraction, human takeover on CAPTCHA.
*Shows: You ship agents in the messy real world.*

**7. Invoice Processing Agent with 3-Way Match**
Reads invoices, matches POs and deliveries, flags exceptions, posts to ERP via MCP.
*Shows: You automate expensive enterprise workflows.*

**8. Incident Response Agent**
Correlates alerts, traces, and logs; drafts the post-mortem; suggests the rollback command.
*Shows: You make on-call humans faster, not obsolete.*

**9. Real-Time Voice Ops Agent**
Sub-second voice with tool calling, interruption handling, escalation paths.
*Shows: You can build multimodal agents that feel human.*

**10. Adaptive Tutor Agent with Mastery Memory**
Spaced repetition, difficulty routing, long-term memory of learner state.
*Shows: You use memory systems that personalize over time.*

**11. Agentic Sales Ops Assistant**
Enriches leads, drafts personalized outreach, syncs CRM, tracks replies, updates forecasts.
*Shows: You connect agents to revenue, not just demos.*

**12. Agent Eval and Regression Platform**
Golden trajectories, CI gates that block bad prompts, quality dashboards.
*Shows: You ship agents like production software.*

---

## Active Ideas

- **Personal AI Scheduler** — agent that manages your calendar, surfaces relevant content for upcoming study slots, and reminds you with context at the right time
- **Niche Persona Bot** — ChatGPT/Midjourney-style bot for a specific use case, integrated with WhatsApp, Slack, Discord, or a Streamlit/Gradio app
- **Browser Extension** — summarize, ideate, extract takeaways, and research from web pages
- **Targeted News Aggregator** — curated feed for a specific persona (PMs, AI engineers, etc.)
- **Multi-modal Discord Bot** — generate and interact with content through Discord

---

## Capstone Components

Every capstone should touch as many of these as possible:

1. Agent framework
2. RL fine-tuning + evals
3. Memory framework
4. Data pipeline
5. Classical ML baseline
6. Deep learning architecture
7. LLM integration
8. RAG system
9. Safety & alignment
10. Production deployment
11. User interface
12. Documentation

---

## References

- [Voice AI 1](https://youtu.be/oU_rr-bOrK8?si=krryAFS9_GfHt6IV)
- [Live AI Voice](https://youtu.be/vaCTaUEpqvE?si=o0yM9FzIAE3EwgDS)
