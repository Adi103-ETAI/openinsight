Multi-Agent Design (5 Agents)
Using LangGraph (LangChain's agent framework) - more mature than CrewAI for complex orchestration:
┌─────────────────────────────────────────────────────────────────┐
│                      DeepInsightsOrchestrator                   │
│                        (Main Agent)                             │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│QueryDecomposer│    │ IntentRouter  │    │PlanGenerator  │
│ (Sub-queries) │    │ (Route query) │    │(Plan steps)   │
└───────────────┘    └───────────────┘    └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ParallelRetrieverAgent                       │
│              (Spawns N retrievers for each sub-query)           │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌────────┬────────┬────────┬────────┐
        ▼        ▼        ▼        ▼        ▼
    Retriever Retriever Retriever Retriever Retriever
     (drug)   (dose)  (contra)  (guide)  (research)
        └────────┬────────┬────────┬────────┘
                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                  EvidenceSynthesizerAgent                       │
│        (Merge results, detect contradictions, rank)             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   ResponseFormatterAgent                        │
│        (Structure answer, citations, confidence)                │
└─────────────────────────────────────────────────────────────────┘