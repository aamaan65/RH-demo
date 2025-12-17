# Calls Section (Classic Calls Summary) — README

This document describes the **Calls** workflow for transcript processing: what it does, what the UI shows, and the **exact prompts + functions** used in the backend.

---

## What Calls is (high level)

Calls mode behaves like a **coverage-focused assistant** for CSRs:

- **Input**: a call transcript (often tech + CSR authorization).
- **Output**:
  - per-question **RAG answers** (with cited chunks), and
  - one **final summary** (single text block) plus an overall `claimDecision`.

---

## Backend entrypoints (APIs)

Calls transcript processing is implemented by these endpoints in `RH-demo/contract-pdf-qna/app.py`:

- **Non-stream**: `POST /transcripts/process`
- **Streaming (SSE)**: `POST /transcripts/process/stream`

Calls UI sends `gptModel: "Infer"` (under the hood) but labels the conversation as **Calls** in the UI/history.

---

## Output contract (Calls)

### Response JSON: `/transcripts/process`

Calls transcript responses return:

- `questions: QuestionResult[]` (per extracted question with answer + `relevantChunks`)
- `claimDecision: { decision, shortAnswer, reasons, authorizedAmount, confidence }`
- `finalSummary: string` (single summary text shown as the “Final Answer for transcript”)

### Streaming SSE events: `/transcripts/process/stream`

The stream emits (in addition to `status` events):

- `answer` — per question (policy-grounded answer + chunks)
- `final` — once, containing `{ finalSummary, claimDecision }`
- `done` — completion metadata

---

## Calls pipeline (backend)

All Calls logic lives in `RH-demo/contract-pdf-qna/app.py`.

### Step 0: Transcript cleaning

Function:
- `_clean_calls_transcript_text(raw: str) -> str`

Purpose:
- Remove metadata/noise (`state=...`, `plan=...`, `contractType=...`, `transcribe:`/`transcript:` prefixes)
- Normalize whitespace so extraction doesn’t treat metadata as the “issue”

### Step 1: Question extraction (Agent-based)

Function:
- `extract_questions_with_agent(transcript_text, llm)`

### Step 2: Answer each question (RAG)

Function:
- `process_single_transcript_question(...)`

### Step 3: Final summary + overall claim decision

Functions:
- `generate_claim_decision_from_qa(qa_blob, ...)`
- `generate_calls_final_summary(qa_blob, claim_decision, ...)`

This step produces:
- `claimDecision` (overall decision + reasons + authorizedAmount string)
- `finalSummary` (single CSR-friendly summary text shown as “Final Answer for transcript”)

### Step 1: Split transcript into dispatch/work blocks (even without IDs)

Function:
- `_split_calls_transcript_into_dispatches(cleaned: str) -> List[Dict]`

Heuristics:
- Split on explicit markers when present (e.g., “dispatch number”, “work order”)
- Otherwise split on “total authorization / I come up with $X…” type boundaries
- Fallback: single dispatch `AUTO-1`

### Step 2: Extract atomic, contextual adjudication questions (Agent-based extraction)

Function (used for Calls extraction):
- `extract_questions_with_agent(transcript_content: str, llm) -> List[Dict]`

It uses:
- **Tool prompt**: `extraction_prompt_template` (3-step process + Calls hard requirements)
- **Agent system prompt**: `agent_sys_msg` (instructions to call the tool and return tool output)
- Post-processing:
  - `json.loads(...)`
  - `filter_relevant_customer_questions(...)`
  - assigns `questionId: q1..qN`

Key requirement enforced in the extraction prompt:
- Do **not** produce generic “Is it covered?” questions; every question must explicitly mention the **specific appliance/system and issue/service**.

### Step 3: Answer each question grounded in RAG (policy chunks)

Functions:
- `process_single_transcript_question(...)` (per-question retrieval + answer)
- `input_prompt(...)` (LangChain agent that calls the Knowledge Base tool)

Calls per-question answer prompt:
- `calls_decision_sys_msg` (system message)
  - Requires: CSR-friendly plain English + 2–4 policy-grounded bullets
  - Forbids: per-question “Decision/AuthorizedAmount/Why/Conditions” blocks
  - Forbids: “cannot determine / cannot confirm / policy does not mention…”
  - Allows: “Needs human review” **with concrete missing clause/info**

Safety post-processor:
- `_sanitize_calls_per_question_answer(answer_text: str) -> str`
  - Strips legacy “Decision:” formatting if the model returns it anyway.

### Step 4: Synthesis & Adjudication Layer (Rule vs Fact)

This is the USP layer: it **does not search**. It **compares**.

#### 4A) Build a structured “Context Object” per line item

Function:
- `_build_calls_dispatch_decision(dispatch_id, dispatch_text, dispatch_results, llm=None) -> Dict`

What it builds per line item:
- `claim_item`: inferred line item name (heuristic mapping)
- `diagnosis_fact`: excerpt from the dispatch transcript around the item
- `qa_facts`: condensed Q/A lines for that item
- `cost_quoted`: best-effort extracted amounts
- `rag_retrieved_policy`: joined retrieved chunk text

Supporting helpers:
- `_infer_calls_item_name(question_text, dispatch_text="") -> str`
- `_extract_money_amounts(text: str) -> List[float]`
- `_dispatch_excerpt_for_item(dispatch_text, item_name, max_len=700) -> str`
- `_parse_first_amount_from_text(text: str) -> float`

#### 4B) Adjudicate Rule vs Fact into strict JSON

Function:
- `_adjudicate_calls_line_item(context_obj: Dict, llm=None) -> Dict`

Prompt persona:
- “You are an Expert Claims Adjudicator for American Home Shield.”

Prompt behavior:
- Strict **Rule vs Fact** evaluation using only `rag_retrieved_policy`
- Output **ONLY** valid JSON with:
  - `decision: APPROVED|DENIED|FLAG_FOR_REVIEW|APPROVED_WITH_CAP`
  - `confidence_score`
  - `cost_impact`, `policy_limit`
  - `cited_chunks`

#### 4C) Roll up dispatch decision

Function:
- `_rollup_dispatch_decision(dispatch_id: str, line_items: List[Dict]) -> Dict`

Rollup rules:
- Mixed approved/denied/flagged -> `PARTIALLY_APPROVED`
- All approved -> `APPROVED`
- All denied -> `DENIED`
- Anything unclear -> `FLAG_FOR_REVIEW`

### Step 5: Persist to Mongo + serve via /history

Conversation document fields (for transcript conversations):
- `dispatch_decisions` is stored on the conversation doc
- `response_payload.dispatchDecisions` caches the API payload for fast reloads

History API:
- `GET /history?conversation-id=...` returns `dispatchDecisions` for Calls transcript conversations.

---

## Frontend (Calls UI)

Calls UI logic is in:
- `RH-demo/contract-pdf-qna-frontend/src/components/home/home.jsx`
- `RH-demo/contract-pdf-qna-frontend/src/components/home/home.scss`

Behavior:
- Calls transcript run consumes:
  - **Non-stream** response: `dispatchDecisions`
  - **Stream** events: `answer`, `dispatchDecision`, `dispatchDecisions`, `done`
- Calls UI renders:
  - Decision Dashboard cards per dispatch (status + total authorized + line items)
  - Per-question Q/A list below for traceability

---

## Exact prompt locations (backend)

All prompts live in `RH-demo/contract-pdf-qna/app.py`:

- **Per-question answering system message**: `calls_decision_sys_msg`
- **Agent-based extraction tool prompt**: `extraction_prompt_template` (inside `extract_questions_with_agent`)
- **Extraction agent system message**: `agent_sys_msg` (inside `extract_questions_with_agent`)
- **Adjudication prompt**: inside `_adjudicate_calls_line_item`


