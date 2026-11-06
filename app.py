"""Flask web server: chat endpoint, confirmation flow, state/audit endpoints."""
import collections, os, time, uuid
from werkzeug.exceptions import HTTPException
from flask import Flask, jsonify, render_template, request, send_file
import agent

app = Flask(__name__)
llm = agent.LLM()
PENDING, HITS = {}, collections.defaultdict(collections.deque)
# Free LLM tiers allow only a few requests per minute (and one chat can use 2 LLM calls), so the defaults are small.
# Raise them if your provider's limits are higher.
GLOBAL_RPM = int(os.getenv("GLOBAL_RPM", 10))  # all sessions combined: keeps the server under the provider's quota
SESSION_RPM = int(os.getenv("SESSION_RPM", 6))  # per browser session


def limited(key, per_min=SESSION_RPM):
    q, now = HITS[key], time.time()
    while q and now - q[0] > 60: q.popleft()
    if len(q) >= per_min: return True
    q.append(now); return False


def workspace(sid):
    try: return agent.Workspace(sid), None
    except agent.SafetyError as e: return None, (jsonify(status="error", error=str(e)), 400)


def run(ws, plan, msg):
    results = agent.execute(ws, plan)
    ok = all(r["ok"] for r in results)
    reply = agent.summarize(llm, msg, results) if ok else None
    return dict(status="done" if ok else "failed", plan=plan, results=results, reply=reply or plan["reply"])


@app.errorhandler(Exception)
def unexpected(e):
    if isinstance(e, HTTPException): return e
    app.logger.exception(e)
    return jsonify(status="error", error="Unexpected server error. Try again."), 500


@app.get("/")
def index(): return render_template("index.html")


@app.post("/api/chat")
def chat():
    d = request.get_json(silent=True) or {}
    ws, err = workspace(d.get("sid"))
    if err: return err
    msg = str(d.get("message", "")).strip()[:1000]
    if not msg: return jsonify(status="error", error="Type a request first."), 400
    if limited(ws.sid): return jsonify(status="error", error=f"Rate limit: max {SESSION_RPM} requests per minute."), 429
    if limited("*", GLOBAL_RPM): return jsonify(status="error", error="The server is busy. Please try again in a minute."), 429
    ws.log("request", message=msg)
    try:
        plan = agent.prepare(ws, agent.plan(llm, msg))
    except (agent.PlanError, agent.SafetyError, agent.LLMError) as e:
        ws.log("blocked", error=str(e))
        return jsonify(status="error", error=str(e))
    if any(a["confirm"] for a in plan["actions"]):
        for k in [k for k, v in PENDING.items() if time.time() - v["ts"] > 600]: PENDING.pop(k)
        pid = uuid.uuid4().hex[:8]
        PENDING[(ws.sid, pid)] = dict(plan=plan, msg=msg, ts=time.time())
        ws.log("awaiting_confirmation", plan_id=pid, actions=[a["tool"] for a in plan["actions"] if a["confirm"]])
        return jsonify(status="needs_confirmation", plan_id=pid, plan=plan)
    return jsonify(run(ws, plan, msg))


@app.post("/api/confirm")
def confirm():
    d = request.get_json(silent=True) or {}
    ws, err = workspace(d.get("sid"))
    if err: return err
    item = PENDING.pop((ws.sid, d.get("plan_id")), None)
    if not item: return jsonify(status="error", error="That plan expired. Send your request again.")
    if not d.get("approve"):
        ws.log("cancelled", plan_id=d.get("plan_id"))
        return jsonify(status="cancelled", reply="Cancelled. Nothing was changed.")
    ws.log("approved", plan_id=d.get("plan_id"))
    return jsonify(run(ws, item["plan"], item["msg"]))


@app.get("/api/state")
def state():
    ws, err = workspace(request.args.get("sid"))
    if err: return err
    return jsonify(files=ws.tree(), tasks=ws.tasks(), log=ws.recent_log(), mock=llm.mock, model=llm.model)


def sandbox_file():
    """Resolve ?sid=&path= to a file inside that session's sandbox (same path checks the agent tools use)."""
    ws, err = workspace(request.args.get("sid"))
    if err: return None, None, err
    try: return ws, agent._file(ws, request.args.get("path", "")), None
    except agent.SafetyError as e: return None, None, (jsonify(status="error", error=str(e)), 400)
    except OSError as e: return None, None, (jsonify(status="error", error=str(e)), 404)


@app.get("/api/file")
def file_view():
    ws, f, err = sandbox_file()
    if err: return err
    if f.stat().st_size > agent.MAX_BYTES: return jsonify(status="error", error="File too large to preview. Use download."), 413
    ws.log("viewed", message=request.args["path"])
    return jsonify(status="ok", path=request.args["path"], content=f.read_text(errors="replace"))


@app.get("/api/download")
def file_download():
    ws, f, err = sandbox_file()
    if err: return err
    ws.log("downloaded", message=request.args["path"])
    r = send_file(f, as_attachment=True, download_name=f.name, mimetype="text/plain")  # plain text + nosniff: never rendered as a page
    r.headers["X-Content-Type-Options"] = "nosniff"
    return r


if __name__ == "__main__":
    app.run(debug=os.getenv("FLASK_DEBUG") == "1")  # debug mode exposes an interactive debugger, so it is opt-in