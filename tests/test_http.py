"""Run: python tests/test_http.py   The REAL `openai` SDK talking to a local fake OpenAI-compatible server (no key or internet needed).
Complements test_llm.py, which stubs the SDK: this one proves the retry/error code works with the SDK's real exceptions and headers."""
import json, os, socket, sys, tempfile, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
try: import openai  # noqa: F401
except ImportError: sys.exit("skipped: run pip install -r requirements.txt first")
os.environ["WORKSPACE_ROOT"] = tempfile.mkdtemp()
for k in ("GEMINI_API_KEY", "GROQ_API_KEY"): os.environ.pop(k, None)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SCRIPT, SEEN = [], []  # queued (status, headers, body) replies; (path, auth header, body) of each request received
class Fake(BaseHTTPRequestHandler):
    def do_POST(self):
        SEEN.append((self.path, self.headers.get("Authorization"), json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
        status, headers, body = SCRIPT.pop(0)
        self.send_response(status); self.send_header("Content-Type", "application/json")
        for k, v in headers.items(): self.send_header(k, v)
        self.end_headers(); self.wfile.write(json.dumps(body).encode())
    def log_message(self, *a): pass
srv = HTTPServer(("127.0.0.1", 0), Fake); threading.Thread(target=srv.serve_forever, daemon=True).start()
os.environ.update(LLM_API_KEY="secret", LLM_BASE_URL=f"http://127.0.0.1:{srv.server_port}/v1", LLM_MODEL="fake-model")
import agent
sleeps = []; agent.time.sleep = sleeps.append
llm = agent.LLM(); assert not llm.mock
ok = lambda text: (200, {}, {"id": "x", "object": "chat.completion", "created": 0, "model": "fake-model",
                             "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}]})
err = lambda status, msg, headers=None: (status, headers or {}, {"error": {"message": msg, "type": "x"}})
ask = lambda: llm.complete("sys prompt", [{"role": "user", "content": "hi"}])

SCRIPT[:] = [err(429, "slow down", {"Retry-After": "2"}), ok("hello")]
assert ask() == "hello" and sleeps == [2]                                      # real 429 -> waits the Retry-After seconds, then succeeds
path, auth, body = SEEN[-1]
assert path == "/v1/chat/completions" and auth == "Bearer secret" and body["model"] == "fake-model"
assert body["messages"] == [{"role": "system", "content": "sys prompt"}, {"role": "user", "content": "hi"}]
SCRIPT[:] = [err(500, "oops"), ok("again")]; sleeps.clear()
assert ask() == "again" and sleeps == [1]                                      # 5xx -> backoff, retry
SCRIPT[:] = [err(400, "model not found")]
try: ask(); raise SystemExit("no error")
except agent.LLMError as e: assert "400" in str(e) and "model not found" in str(e)  # 4xx -> immediate, readable error
SCRIPT[:] = [ok(None)]; assert ask() == ""                                     # null content (e.g. blocked by a safety filter) -> empty string
SCRIPT[:] = [(200, {}, {"id": "x", "object": "chat.completion", "created": 0, "model": "m", "choices": []})]; assert ask() == ""
plan = {"reasoning": "r", "actions": [{"tool": "db_list_tasks", "args": {}, "risk": "low"}], "reply": "Here are your tasks."}
SCRIPT[:] = [ok("Sorry, here is your plan:"), ok("```json\n" + json.dumps(plan) + "\n```")]
assert agent.plan(llm, "show my tasks")["actions"][0]["tool"] == "db_list_tasks"  # full path: HTTP -> parse -> validate, with a self-repair round
s = socket.socket(); s.bind(("127.0.0.1", 0)); dead = s.getsockname()[1]; s.close()  # a port nobody listens on
os.environ["LLM_BASE_URL"] = f"http://127.0.0.1:{dead}/v1"; sleeps.clear()
try: agent.LLM().complete("s", [{"role": "user", "content": "hi"}]); raise SystemExit("no error")
except agent.LLMError as e: assert "retries" in str(e) and sleeps == [1, 2, 4]     # server down -> 4 attempts, then a clean error
print("http tests passed")
