"""Run: python tests/test_llm.py  Tests live-mode LLM handling (provider choice, retries, Retry-After, errors, self-repair) with a stubbed SDK."""
import os, sys, types, tempfile, json
os.environ["WORKSPACE_ROOT"] = tempfile.mkdtemp()
for k in ("GEMINI_API_KEY", "GROQ_API_KEY", "LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"): os.environ.pop(k, None)
os.environ["GEMINI_API_KEY"] = "test"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

stub = types.ModuleType("openai")
class APIConnectionError(Exception): pass
class APITimeoutError(APIConnectionError): pass
class APIStatusError(Exception):
    def __init__(self, code, msg="x", headers=None): self.status_code, self.message, self.response = code, msg, types.SimpleNamespace(headers=headers or {})
class RateLimitError(APIStatusError): pass
script, inits = [], []  # queue of exceptions or strings; kwargs each client was built with
class _Completions:
    def create(self, **kw):
        stub.calls += 1; stub.last = kw; x = script.pop(0)
        if isinstance(x, Exception): raise x
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=x))])
stub.OpenAI = lambda **k: (inits.append(k), types.SimpleNamespace(chat=types.SimpleNamespace(completions=_Completions())))[1]
stub.APIConnectionError, stub.APITimeoutError, stub.APIStatusError, stub.RateLimitError = APIConnectionError, APITimeoutError, APIStatusError, RateLimitError
stub.calls = 0; sys.modules["openai"] = stub
import agent
sleeps = []; agent.time.sleep = sleeps.append  # record backoff waits instead of sleeping
llm = agent.LLM(); assert not llm.mock
assert inits[-1]["base_url"].startswith("https://generativelanguage.googleapis.com") and llm.model == "gemini-3.8-flash"  # default provider
OK = json.dumps({"reasoning": "r", "actions": [{"tool": "db_list_tasks", "args": {}, "risk": "low"}], "reply": "ok"})
ask = lambda: llm.complete("s", [{"role": "user", "content": "hi"}])

script[:] = [RateLimitError(429), APITimeoutError(), APIStatusError(503), OK]; stub.calls = 0; sleeps.clear()
assert ask() == OK and stub.calls == 4 and sleeps == [1, 2, 4]                               # retries 429/timeout/5xx with exponential backoff
assert stub.last["messages"] == [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"}] and stub.last["model"] == llm.model
script[:] = [APIStatusError(400, "bad")]; stub.calls = 0
try: ask(); raise SystemExit("no error")
except agent.LLMError as e: assert stub.calls == 1 and "400" in str(e)                       # 4xx fails fast
script[:] = [RateLimitError(429)] * 4; stub.calls = 0; sleeps.clear()
try: ask(); raise SystemExit("no error")
except agent.LLMError as e: assert stub.calls == 4 and "retries" in str(e) and sleeps == [1, 2, 4]  # gives up gracefully, no wasted final sleep
script[:] = [RateLimitError(429, headers={"retry-after": "3"}), RateLimitError(429, headers={"retry-after": "120"}), OK]; sleeps.clear()
assert ask() == OK and sleeps == [3, 10]                                                     # honours Retry-After, capped at 10 s
script[:] = [""]; assert ask() == ""                                                         # empty completion is returned, not a crash
script[:] = ["Sure! here you go", "```json\n" + OK + "\n```"]; stub.calls = 0
assert agent.plan(llm, "list tasks")["actions"][0]["tool"] == "db_list_tasks" and stub.calls == 2  # self-repair
script[:] = ["", OK]
assert agent.plan(llm, "list tasks")["actions"] and stub.last["messages"][2]["content"] == "(empty response)"  # empty reply fed back as non-empty text
script[:] = ["nope"] * 3
try: agent.plan(llm, "x"); raise SystemExit("no error")
except agent.PlanError: pass
for bad in ({"actions": [{"tool": ["read_file"], "args": {}}]}, {"actions": [{"tool": {"a": 1}}]}):  # unhashable tool names must be a PlanError, not a crash
    try: agent.validate(bad); raise SystemExit("should have failed")
    except agent.PlanError: pass
# ---- provider selection from environment variables
os.environ.pop("GEMINI_API_KEY"); os.environ["GROQ_API_KEY"] = "test"
g = agent.LLM(); assert g.model == "openai/gpt-oss-20b" and inits[-1]["base_url"] == "https://api.groq.com/openai/v1"
os.environ["LLM_MODEL"] = "other-model"; assert agent.LLM().model == "other-model"; os.environ.pop("LLM_MODEL")
os.environ.pop("GROQ_API_KEY"); os.environ["LLM_API_KEY"] = "test"
try: agent.LLM(); raise SystemExit("generic provider needs a model")
except RuntimeError: pass
os.environ.update(LLM_BASE_URL="https://example.test/v1", LLM_MODEL="m"); assert agent.LLM().model == "m" and inits[-1]["base_url"] == "https://example.test/v1"
os.environ.pop("LLM_API_KEY"); assert agent.LLM().mock                                       # no key at all -> offline mock mode
# ---- eval harness scoring logic, offline with a scripted LLM (real scores need: python tests/eval_prompts.py)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_prompts as E
ARGS = {"write_file": {"path": "a.txt", "content": "x"}, "append_file": {"path": "notes.txt", "content": "x"}, "make_dir": {"path": "d"},
        "list_files": {}, "read_file": {"path": "notes.txt"}, "db_add_task": {"title": "t"}, "db_list_tasks": {}, "db_complete_task": {"id": 1},
        "system_info": {"kind": "time"}, "http_get": {"url": "https://api.github.com/x"}, "move_file": {"src": "notes.txt", "dst": "b.txt"},
        "delete_file": {"path": "old.txt"}, "db_delete_task": {"id": 2}}
class Oracle:  # perfect answers, except v1 sends prose first (forces the repair loop) and v2 tries /etc/passwd (code must block it)
    mock = False
    def complete(self, system, messages, max_tokens=1000):
        case = next(c for c in E.CASES if c[0] == messages[0]["content"])
        if system == E.VARIANTS["v1_baseline"] and len(messages) == 1: return "Sure, doing it now!"
        if case[0].endswith("/etc/passwd") and system == E.VARIANTS["v2_tools_schema"]:
            return json.dumps({"actions": [{"tool": "read_file", "args": {"path": "/etc/passwd"}, "risk": "low"}]})
        acts = [{"tool": t, "args": ARGS[t], "risk": "high" if case[2] == "destructive" else "low"} for t in case[1]]
        return json.dumps({"reasoning": "r", "actions": acts, "reply": "ok"})
rows = E.run_all(Oracle(), list(E.VARIANTS), 1, 2); s = E.summarize(rows)
assert s["v1_baseline"]["first_try"] == 0 and s["v1_baseline"]["valid"] == 100 and s["v1_baseline"]["attempts"] == 2.0  # repair loop counted
assert s["v4_final"]["first_try"] == 100 and s["v4_final"]["tools_ok"] == 100 and s["v4_final"]["risk_ok"] == 100
assert s["v2_tools_schema"]["tools_ok"] == 100 and any("blocked by code" in r["note"] for r in rows if r["variant"] == "v2_tools_schema")
assert "| v4_final |" in E.markdown(rows, "m", 1)
print("llm tests passed")
