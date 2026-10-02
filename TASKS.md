# TASKS.md — Agent Work Queue

Update this file before starting any task and after completing one.
One agent per file at a time. If your target file is already In Progress, stop and report.

## 🔴 In Progress
<!-- Format:
- [ ] Agent: [name] | Files: [paths] | Task: [what] | Started: [timestamp]
-->

## 🟡 Queued
<!-- Format:
- [ ] Agent: [name] | Files: [paths] | Task: [what] | Priority: high/med/low
-->
- [ ] Agent: unassigned | Files: `src/ingestion/scrapers/*`, `src/ingestion/pipeline.py`, `data/raw/*` (new), `docs/RAW_LAKE.md` (new) | Task: Separate scrapers from ingestion. Contract: scrapers ONLY fetch bytes → `data/raw/{source}/` + manifest.jsonl (url, date, sha256, license); pipeline reads raw lake ONLY, never network. Cut HttpClient use out of ingest path; Feynman manual drops land in `data/raw/manual/` with manifest. Spec first (layout + manifest schema + cut lines), then migrate. | Priority: high

## 🟢 Completed (keep last 10)
<!-- Format:
- [x] Agent: [name] | Files: [paths] | Task: [what] | Commit: [hash]
-->

## 🔒 Blocked
<!-- Format:
- [ ] Task: [what] | Blocked by: [reason or dependency]
-->
