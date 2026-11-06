# Workbench Agent: LLM-driven action agent (CS4680 AI Agent Project)

A web agent that sends natural-language requests to an LLM (any OpenAI-compatible API; it runs on the **free** Google Gemini or Groq tiers, so no paid credits are needed), parses the model's JSON plan, validates it, asks for confirmation on risky steps, executes real actions, and logs everything.

## Problem and relevance
Routine chores (making notes, tracking tasks, tidying files, checking a public API) mean clicking through several apps. LLMs understand plain English but can't act on their own, and letting them act unchecked is risky. This agent closes the gap: the model *proposes* a structured plan, and deterministic code *validates, confirms and executes* it. That trust boundary (LLM output is untrusted input) is the core idea behind real-world agents.

## Requirement checklist

| Requirement | Where |
|---|---|
| **1. LLM integration**: API calls, prompts, error + rate-limit handling | `agent.LLM.complete`: OpenAI-compatible client (free Gemini or Groq key, or any other compatible service), 30 s timeout, retries with exponential backoff (1/2/4 s) on 429, timeout, connection and 5xx errors, obeys a 429's `Retry-After` header (up to 10 s); 4xx raises a clear error. Per-session limit (`SESSION_RPM`, default 6/min) plus a global cap (`GLOBAL_RPM`, default 10/min) in `app.limited` keep a free quota from being burned; a session cap (`MAX_SESSIONS`) and per-session file quota (`MAX_FILES`) stop a public deployment from being flooded. |
| **2. Action interpreter/executor**: parse LLM output, run across domains | `parse` + `validate` turn text into a typed plan; `execute` runs it. 4 domains, 13 tools: **files** (list/read/write/append/mkdir/delete/move), **database** (SQLite tasks), **API calls** (`http_get`, allow-listed hosts), **system** (`system_info`, read-only). |
| **3. UI**: natural language input, results, status, feedback | `templates/index.html`: chat, plan card per request (reasoning, each action with risk tag, result, success/fail colour), approve/cancel buttons, live Files / Tasks / Audit-log side panel. |
| **4. Error handling + safety** | Schema validation of every LLM action; self-repair loop (rejected output is fed back to the model, 3 tries); sandboxed per-session directory with path-traversal and absolute-path blocking; URL allow-list, HTTPS only, no redirects; size limits; **confirmation** for deletes, moves, DB deletes, overwriting existing files and anything the model marks high-risk; failures stop the remaining steps; every request, block, approval, cancellation and execution is appended to `audit.jsonl` and shown in the UI. |

## Prompt engineering techniques (`agent.SYSTEM`)
1. **Role + task framing**: "planning module of a tool-using agent".
2. **Structured output contract**: one JSON object with an explicit schema, no fences; parser still tolerates fences.
3. **Tool schema injected from the registry**, so the prompt and validator never drift.
4. **Few-shot examples**: a create, a mixed delete + read, and a refusal.
5. **Brief chain-of-thought field** (`reasoning`) shown to the user for transparency.
6. **Model self-assessed risk** (`low/medium/high`); it can only *escalate* confirmation, never bypass the code-level rules.
7. **Negative constraints / refusal path**: no absolute paths or `..`; unrelated or unsafe requests return `actions: []`.
8. **Self-correction loop**: validation errors are returned to the model as a follow-up message.
9. **Two-step ReAct-style loop**: after read-type tools run, a second prompt turns results into an answer.
10. **Prompt-injection defense**: tool results are wrapped in `<tool_results>` and declared untrusted data.

## Prompt iteration (evidence for the Prompt Engineering criterion)
`tests/eval_prompts.py` scores four prompt versions against 21 fixed requests (13 normal, 4 destructive, 4 unsafe or out-of-scope) using the **live model**, then writes `eval_results.md` and `eval_results.json`. Run it once with your free key and commit those two files, so the numbers are visible to graders.

| Version | Adds |
|---|---|
| `v1_baseline` | bare instruction, no tools listed |
| `v2_tools_schema` | tool registry and JSON shape |
| `v3_rules` | path, risk-label and refusal rules |
| `v4_final` | few-shot examples, injection rule, action limit (the production prompt) |

Metrics: first-try valid JSON, valid after the repair loop, correct tools, correct risk label, average LLM calls and latency. Unsafe requests pass if the model refuses *or* the code blocks them. The failure list under each version is the error analysis that motivates the next change.

```bash
python tests/eval_prompts.py --pause 6                 # ~85 requests plus repair retries; --pause keeps you under free-tier requests/minute
python tests/eval_prompts.py --variants v1_baseline,v4_final --pause 6   # half the cost: just the "before" and "after"
python tests/eval_prompts.py --variants v2_tools_schema,v3_rules --merge # finish the rest later (e.g. after a daily quota reset), keeping saved results
```
If a quota error stops the run, the finished prompt versions are saved; rerun the rest with `--merge`. The offline harness test (`tests/test_llm.py`) checks the scoring logic without any API.

## Free LLM setup (no paid credits)
The agent uses the OpenAI-style chat API, which Google Gemini and Groq both offer on a free tier. Free-tier models, names and limits change, so check the provider page if something is rejected.

| Provider | Get a key | Set this variable | Default model |
|---|---|---|---|
| Google Gemini (default) | [Google AI Studio](https://aistudio.google.com/apikey) | `GEMINI_API_KEY` | `gemini-3.8-flash` |
| Groq | [Groq console](https://console.groq.com/keys) | `GROQ_API_KEY` | `openai/gpt-oss-20b` |
| Any other OpenAI-compatible service (OpenRouter, Cerebras, ...) | provider's site | `LLM_API_KEY` + `LLM_BASE_URL` + `LLM_MODEL` | none: you must set `LLM_MODEL` |

* The first key variable that is set wins. `LLM_MODEL` and `LLM_BASE_URL` override the defaults for any provider.
* Keep billing **disabled** on the key's project so it stays on the free tier.
* Google says free-tier prompts may be used to improve its products, so don't put secrets in requests or files.
* "model not found" (404): model names change. Open the provider's model list and set `LLM_MODEL`. For Gemini, the [pricing page](https://ai.google.dev/gemini-api/docs/pricing) shows which models are free of charge.
* 429 / quota errors: free tiers allow only a few requests per minute and a daily total. Wait, or try a lighter model (for example a Flash-Lite model on Gemini). Check your limits in AI Studio. On Groq, if you hit per-minute token limits, set `LLM_MAX_TOKENS=1500`.
* Header says "Mock mode": the server process can't see the key. Set the variable in the same terminal (or the host's environment settings) and restart.

## Run locally
```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your-free-key   # optional: without a key the app runs in offline mock mode
python app.py                         # http://127.0.0.1:5000
python tests/test_agent.py && python tests/test_llm.py && python tests/test_http.py   # no key or internet needed
```
Windows PowerShell: `$env:GEMINI_API_KEY="your-free-key"`. Windows cmd: `set GEMINI_API_KEY=your-free-key`.
Optional knobs: `LLM_MODEL`, `LLM_BASE_URL`, `LLM_MAX_TOKENS`, `SESSION_RPM`, `GLOBAL_RPM`, `MAX_FILES`, `MAX_SESSIONS`, `FLASK_DEBUG=1`.

## Deploy (Render, free tier) → this gives the "Deployed URL"
1. Push this folder to a new GitHub repo (this is the "Code" link).
2. render.com → New → Web Service → connect the repo.
3. Build command `pip install -r requirements.txt`; start command `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4`.
4. Add env var `GEMINI_API_KEY` (or `GROQ_API_KEY`). Deploy and submit the `.onrender.com` URL. A free key can't run up a bill, but its quota is shared by everyone who tries the app, so don't spend it on repeated testing before grading.
(Hugging Face Spaces, Railway or Fly.io work the same way; the included `Procfile` covers Railway/Heroku.)
Notes: free-tier disks are ephemeral, so files and tasks reset on redeploy. A free service sleeps after 15 minutes idle and takes about a minute to wake, so open the URL before a demo or grading session.

## 1–2 minute demo script
1. (10s) Intro: problem is "doing routine file/task chores by chatting instead of clicking".
2. (20s) `create todo.txt with buy milk and eggs`, then show the plan card and the file in the sidebar.
3. (15s) `add task: finish project`, then switch to the Tasks tab.
4. (20s) `delete todo.txt`: show the confirmation prompt, click Approve.
5. (15s) `read ../../etc/passwd`: show the safety block.
6. (10s) Open the Audit log tab. Close with the prompt-engineering list.

## Layout
`app.py` server · `agent.py` LLM, parser, validator, tools, safety · `templates/index.html` UI · `tests/` unit tests (`test_agent.py` mock LLM, `test_llm.py` stubbed SDK, `test_http.py` real SDK vs a local fake server) + `eval_prompts.py`
#   C S 4 6 8 0 - A 4  
 