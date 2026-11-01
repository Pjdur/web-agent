import os
import uuid
import subprocess
from flask import Flask, render_template, request, jsonify
from google import genai
from google.genai import types

app = Flask(__name__)

# Create a dedicated workspace directory for the agent to operate in safely
WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "workspace"))
os.makedirs(WORKSPACE_DIR, exist_ok=True)

# In-memory store for conversation histories: { session_id: [types.Content, ...] }
SESSIONS = {}

SYSTEM_INSTRUCTION = f"""You are an autonomous, helpful software engineering AI agent running in a web UI.
You operate inside a workspace directory located at: {WORKSPACE_DIR}
All file paths you read or write should be relative to this workspace directory.

Important execution constraints:
1. The host agent is already using port 5000. Any web server you create must use a different port (e.g., 5001, 8080).
2. On Windows, never run Flask apps with debug=True or use_reloader=True inside captured subshells (this causes WinError 10038). Always use `debug=False, use_reloader=False`.
3. To test web servers without blocking, prefer using unit tests (e.g., Flask test_client) or start them in a new window via `start cmd /k python ...` on Windows.
"""

# --- Tool Declarations for Gemini (Manual Execution) ---
AGENT_TOOLS = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="list_files",
            description="List files and directories inside a relative path in the workspace.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "directory": types.Schema(
                        type="STRING",
                        description="Relative directory path (use '.' for workspace root)."
                    )
                },
            ),
        ),
        types.FunctionDeclaration(
            name="read_file",
            description="Read the text contents of a file in the workspace.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "filepath": types.Schema(
                        type="STRING",
                        description="Relative path to the file inside the workspace."
                    )
                },
                required=["filepath"],
            ),
        ),
        types.FunctionDeclaration(
            name="write_file",
            description="Create or overwrite a file in the workspace with new content.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "filepath": types.Schema(
                        type="STRING",
                        description="Relative path to the file inside the workspace."
                    ),
                    "content": types.Schema(
                        type="STRING",
                        description="The full text content to write into the file."
                    ),
                },
                required=["filepath", "content"],
            ),
        ),
        types.FunctionDeclaration(
            name="run_command",
            description="Run a shell/terminal command inside the workspace directory.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "command": types.Schema(
                        type="STRING",
                        description="The shell command to execute (e.g., 'python script.py', 'ls -la')."
                    )
                },
                required=["command"],
            ),
        ),
    ]
)


def resolve_safe_path(rel_path: str) -> str:
    """Ensure file operations stay strictly within WORKSPACE_DIR."""
    clean_rel = (rel_path or ".").lstrip("/\\")
    target = os.path.abspath(os.path.join(WORKSPACE_DIR, clean_rel))
    if not target.startswith(WORKSPACE_DIR):
        raise ValueError("Access denied: Path escapes the workspace directory.")
    return target


def execute_local_tool(name: str, args: dict) -> dict:
    """Executes the approved tool on the host system."""
    try:
        if name == "list_files":
            target = resolve_safe_path(args.get("directory", "."))
            if not os.path.exists(target):
                return {"error": f"Directory not found: {args.get('directory')}"}
            items = []
            for entry in sorted(os.listdir(target)):
                full_p = os.path.join(target, entry)
                items.append(f"{entry}/" if os.path.isdir(full_p) else entry)
            return {"files": items}

        elif name == "read_file":
            target = resolve_safe_path(args["filepath"])
            if not os.path.isfile(target):
                return {"error": f"File not found: {args['filepath']}"}
            with open(target, "r", encoding="utf-8", errors="replace") as f:
                return {"content": f.read()}

        elif name == "write_file":
            target = resolve_safe_path(args["filepath"])
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with open(target, "w", encoding="utf-8") as f:
                f.write(args["content"])
            return {"status": "success", "message": f"Wrote {len(args['content'])} characters to {args['filepath']}"}

        elif name == "run_command":
            cmd = args["command"]
            completed = subprocess.run(
                cmd,
                shell=True,
                cwd=WORKSPACE_DIR,
                capture_output=True,
                text=True,
                timeout=30,  # Prevent infinite hanging processes
            )
            return {
                "returncode": completed.returncode,
                "stdout": completed.stdout[:10000],  # Cap output size
                "stderr": completed.stderr[:10000],
            }

        return {"error": f"Unknown tool: {name}"}
    except subprocess.TimeoutExpired:
        return {"error": "Command timed out after 30 seconds."}
    except Exception as e:
        return {"error": str(e)}


def call_gemini(api_key: str, model_name: str, history: list):
    """Calls Gemini and extracts text + any requested function calls."""
    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        tools=[AGENT_TOOLS],
        temperature=0.2,
    )
    response = client.models.generate_content(
        model=model_name,
        contents=history,
        config=config,
    )

    if not response.candidates or not response.candidates[0].content:
        return "", []

    model_content = response.candidates[0].content
    # Preserve exact model parts (including any thought tokens or call IDs) in history
    history.append(model_content)

    text_chunks = []
    pending_calls = []

    for part in model_content.parts or []:
        if part.text:
            text_chunks.append(part.text)
        if part.function_call:
            fc = part.function_call
            pending_calls.append({
                "id": getattr(fc, "id", None),
                "name": fc.name,
                "args": dict(fc.args) if fc.args else {},
            })

    return "\n".join(text_chunks), pending_calls


# --- Routes ---

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/files", methods=["GET"])
def api_files():
    """Returns a recursive list of files in the workspace for the UI sidebar."""
    file_tree = []
    for root, _, files in os.walk(WORKSPACE_DIR):
        for f in sorted(files):
            full = os.path.join(root, f)
            rel = os.path.relpath(full, WORKSPACE_DIR)
            file_tree.append(rel)
    return jsonify({"workspace": WORKSPACE_DIR, "files": sorted(file_tree)})


@app.route("/api/file", methods=["GET"])
def api_read_single_file():
    """Allows the user to preview a workspace file in the UI."""
    rel_path = request.args.get("path", "")
    try:
        target = resolve_safe_path(rel_path)
        with open(target, "r", encoding="utf-8", errors="replace") as f:
            return jsonify({"path": rel_path, "content": f.read()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Starts or continues a chat turn with a user message."""
    data = request.json or {}
    api_key = data.get("apiKey", "").strip()
    model_name = data.get("model", "gemini-3.8-flash").strip()
    session_id = data.get("sessionId") or str(uuid.uuid4())
    user_message = data.get("message", "").strip()

    if not api_key:
        return jsonify({"error": "Please enter your Gemini API key first."}), 400
    if not user_message:
        return jsonify({"error": "Message cannot be empty."}), 400

    history = SESSIONS.setdefault(session_id, [])
    history.append(types.Content(role="user", parts=[types.Part.from_text(text=user_message)]))

    try:
        reply_text, tool_calls = call_gemini(api_key, model_name, history)
        return jsonify({
            "sessionId": session_id,
            "text": reply_text,
            "toolCalls": tool_calls,
        })
    except Exception as e:
        # Roll back the failed user message so session stays clean
        history.pop()
        return jsonify({"error": str(e)}), 500


@app.route("/api/execute", methods=["POST"])
def api_execute():
    """
    Executes (or rejects) pending tool calls based on user permission in the UI,
    sends the results back to Gemini, and returns Gemini's next step.
    """
    data = request.json or {}
    api_key = data.get("apiKey", "").strip()
    model_name = data.get("model", "gemini-3.8-flash").strip()
    session_id = data.get("sessionId")
    decisions = data.get("decisions", [])  # List of { id, name, args, approved }

    if not api_key or session_id not in SESSIONS:
        return jsonify({"error": "Invalid session or missing API key."}), 400

    history = SESSIONS[session_id]
    response_parts = []
    execution_logs = []

    for item in decisions:
        name = item["name"]
        args = item.get("args", {})
        call_id = item.get("id")
        approved = item.get("approved", False)

        if approved:
            result = execute_local_tool(name, args)
        else:
            result = {"status": "rejected", "message": "User denied permission to run this action."}

        execution_logs.append({
            "name": name,
            "args": args,
            "approved": approved,
            "result": result,
        })

        part = types.Part.from_function_response(name=name, response=result)
        if call_id and part.function_response:
            part.function_response.id = call_id
        response_parts.append(part)

    # Append tool execution results to conversation history
    history.append(types.Content(role="user", parts=response_parts))

    try:
        reply_text, next_tool_calls = call_gemini(api_key, model_name, history)
        return jsonify({
            "sessionId": session_id,
            "executionLogs": execution_logs,
            "text": reply_text,
            "toolCalls": next_tool_calls,
        })
    except Exception as e:
        return jsonify({"error": str(e), "executionLogs": execution_logs}), 500


@app.route("/api/reset", methods=["POST"])
def api_reset():
    session_id = (request.json or {}).get("sessionId")
    if session_id in SESSIONS:
        del SESSIONS[session_id]
    return jsonify({"status": "cleared"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)