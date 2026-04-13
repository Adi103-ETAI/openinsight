# OpenInsight Backend API Guide (Conversation-First)

This guide defines the backend endpoints needed to make the UI behave like Claude/ChatGPT with recent conversations, per-conversation history, and delete controls.

## 1) Current UI status

Today, the UI stores recents in browser `localStorage` (`openinsight_history`) and can:
- show recent conversation items in sidebar
- delete single item
- clear all recents

To make it production-grade and synced across devices/users, move this to backend APIs below.

---

## 2) Base URL and auth

- Base URL: `VITE_API_BASE_URL` (e.g. `https://api.example.com`)
- Auth: Bearer JWT (`Authorization: Bearer <token>`)
- All endpoints should be user-scoped (no cross-user data leakage)

---

## 3) Required endpoints (minimum)

## Health

### `GET /health`
Quick status check for UI startup.

Response:
```json
{ "status": "ok", "service": "openinsight-api" }
```

## Query + Conversation write-through

### `POST /query`
Run retrieval + answer generation and optionally attach to a conversation.

Request:
```json
{
  "query": "Treatment for drug-resistant TB",
  "top_k": 8,
  "mode": "standard",
  "conversation_id": "conv_123",
  "create_conversation_if_missing": true
}
```

Response:
```json
{
  "answer": "...",
  "citations": [
    {
      "index": 1,
      "title": "...",
      "source_type": "pubmed",
      "chunk_text": "...",
      "score": 0.92,
      "mongo_id": "39812345"
    }
  ],
  "query": "Treatment for drug-resistant TB",
  "model": "llama-3.1-70b",
  "chunks_retrieved": 8,
  "mode": "standard",
  "conversation_id": "conv_123",
  "message_id": "msg_456"
}
```

## Conversations list (sidebar recents)

### `GET /conversations?limit=30&cursor=<cursor>`
Return recent conversations for sidebar.

Response:
```json
{
  "items": [
    {
      "id": "conv_123",
      "title": "Treatment for drug-resistant TB",
      "last_query": "Dengue warning signs",
      "updated_at": "2026-04-08T19:00:00Z",
      "message_count": 12
    }
  ],
  "next_cursor": null
}
```

### `POST /conversations`
Create new conversation explicitly.

Request:
```json
{ "title": "New Consultation" }
```

Response:
```json
{ "id": "conv_123", "title": "New Consultation", "created_at": "...", "updated_at": "..." }
```

### `GET /conversations/:conversationId`
Get metadata for one conversation.

### `PATCH /conversations/:conversationId`
Rename conversation title.

Request:
```json
{ "title": "Updated title" }
```

### `DELETE /conversations/:conversationId`
Delete one conversation and all its messages.

Response:
```json
{ "deleted": true }
```

### `DELETE /conversations`
Bulk delete all conversations for current user.

Response:
```json
{ "deleted_count": 42 }
```

## Messages (chat thread)

### `GET /conversations/:conversationId/messages?limit=100&cursor=<cursor>`
Load chat messages for selected conversation.

Response:
```json
{
  "items": [
    {
      "id": "msg_1",
      "role": "user",
      "content": "Treatment for drug-resistant TB",
      "created_at": "..."
    },
    {
      "id": "msg_2",
      "role": "assistant",
      "content": "...",
      "query_payload": {
        "model": "llama-3.1-70b",
        "chunks_retrieved": 8
      },
      "citations": []
    }
  ],
  "next_cursor": null
}
```

### `DELETE /conversations/:conversationId/messages/:messageId`
Delete single message.

Response:
```json
{ "deleted": true }
```

---

## 4) Frontend mapping (what to connect)

- Sidebar "Recent Conversations" -> `GET /conversations`
- Open history item -> `GET /conversations/:id/messages`
- Delete one recent item -> `DELETE /conversations/:id`
- Delete all recent items -> `DELETE /conversations`
- Ask query in active thread -> `POST /query` with `conversation_id`
- Start new consultation -> `POST /conversations` then first `POST /query`

---

## 5) Suggested DB schema (backend)

## `conversations`
- `id` (pk)
- `user_id` (indexed)
- `title`
- `created_at`
- `updated_at` (indexed desc for recents)
- `last_query`
- `message_count`

## `messages`
- `id` (pk)
- `conversation_id` (indexed)
- `user_id` (indexed)
- `role` (`user` | `assistant`)
- `content`
- `citations` (json)
- `query_payload` (json)
- `created_at`

---

## 6) Non-functional requirements

- P95 latency targets:
  - `GET /conversations`: < 150ms
  - `GET /messages`: < 200ms
  - `DELETE` endpoints: < 150ms
- Pagination everywhere list-like
- Idempotent deletes
- Soft-delete optional, hard-delete acceptable
- Audit logs for delete-all endpoint

---

## 7) Migration plan from localStorage

1. Keep localStorage fallback for one release.
2. Add backend sync read path (`GET /conversations`).
3. Write-through on each successful query.
4. After backend stable, disable local-only history.

---

## 8) Ready-to-implement checklist

- [ ] Implement all endpoints above
- [ ] Add auth middleware + user scoping
- [ ] Add DB indexes (`user_id`, `updated_at`, `conversation_id`)
- [ ] Add pagination cursors
- [ ] Add integration tests for create/list/delete flows
- [ ] Add rate-limits for write/delete endpoints

