"""Run: python tests/test_agent.py   (uses mock LLM, no API key or network needed)"""
import os, sys, tempfile, time
os.environ["WORKSPACE_ROOT"] = tempfile.mkdtemp()
for k in ("GEMINI_API_KEY", "GROQ_API_KEY", "LLM_API_KEY"): os.environ.pop(k, None)  # force the offline mock LLM
os.environ["SESSION_RPM"] = os.environ["GLOBAL_RPM"] = "1000"  # this test fires many requests quickly; the limiter itself is tested below
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import agent, app as A
c, SID = A.app.test_client(), "abcdef0123456789"
chat = lambda m: c.post("/api/chat", json={"sid": SID, "message": m}).get_json()

r = chat("create a file called notes.txt with hello world"); assert r["status"] == "done", r
r = chat("list my files"); assert "notes.txt" in r["results"][0]["output"]
r = chat("read notes.txt"); assert r["results"][0]["output"] == "hello world"
r = chat("create a file called notes.txt with new text"); assert r["status"] == "needs_confirmation"  # overwrite
c.post("/api/confirm", json={"sid": SID, "plan_id": r["plan_id"], "approve": False})
assert chat("read notes.txt")["results"][0]["output"] == "hello world"  # cancel changed nothing
r = chat("delete notes.txt"); assert r["status"] == "needs_confirmation"
r = c.post("/api/confirm", json={"sid": SID, "plan_id": r["plan_id"], "approve": True}).get_json(); assert r["status"] == "done"
assert "notes.txt" not in chat("list files")["results"][0]["output"]
assert chat("read ../../etc/passwd")["status"] == "error"  # path traversal blocked
assert chat("read /etc/passwd")["status"] == "error"       # absolute path blocked
chat("add task: buy milk"); assert "buy milk" in chat("show my tasks")["results"][0]["output"]
assert chat("sing a song")["plan"]["actions"] == []
for bad in ["not json", '{"actions":[{"tool":"rm_rf","args":{}}]}', '{"actions":[{"tool":"read_file","args":{}}]}',
            '{"actions":[{"tool":"delete_file","args":{"path":5}}]}', '{"actions":[{"tool":["read_file"],"args":{}}]}']:
    try: agent.validate(agent.parse(bad)); raise SystemExit("should have failed: " + bad)
    except agent.PlanError: pass
assert agent.parse('```json\n{"actions": []}\n```') == {"actions": []}  # fenced JSON tolerated
try: agent.check_url("https://evil.com/x"); raise SystemExit("ssrf not blocked")
except agent.SafetyError: pass
assert c.post("/api/chat", json={"sid": "../x", "message": "hi"}).status_code == 400
log = c.get(f"/api/state?sid={SID}").get_json()["log"]; assert {"request", "executed", "cancelled", "approved"} <= {e["event"] for e in log}
assert chat("read a\x00b")["status"] == "error"                       # null byte: clean block, not a server error
agent.MAX_FILES = 3                                                    # per-session quota
for n in "abc": assert chat(f"create a file called {n}.txt with x")["status"] == "done"
assert chat("create a file called d.txt with x")["status"] == "failed"
agent.MAX_FILES = 100
assert not A.limited("k", 2) and not A.limited("k", 2) and A.limited("k", 2)  # rate limiter
A.HITS["0123456789abcdef"].extend([time.time()] * 1000)                # a session already over its per-minute limit
r = c.post("/api/chat", json={"sid": "0123456789abcdef", "message": "hi"}); assert r.status_code == 429 and "Rate limit" in r.get_json()["error"]
agent.MAX_SESSIONS = len(os.listdir(agent.ROOT))                       # session cap
assert c.post("/api/chat", json={"sid": "fedcba9876543210", "message": "hi"}).status_code == 400
print("all tests passed")
