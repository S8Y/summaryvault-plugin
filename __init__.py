"""
SummaryVault — Hermes Plugin

Captures final session summaries and submits them to SummaryVault
for encrypted archival.

Hooks (Hermes API):
  transform_llm_output(response_text, session_id, model, platform, **kwargs) -> str | None
    — Captures each turn's final response text. Returns None to pass through.
  on_session_finalize(session_id, platform, **kwargs) -> None
    — Session ending, submits last captured output to vault.

Only the last response per session is retained.
Intermediate messages, tool calls, and reasoning are never stored.
"""

import hashlib
import json
import logging
import threading
import uuid
from datetime import datetime, timezone

from .client import SummaryVaultClient
from .queue import SubmissionQueue

log = logging.getLogger("hermes.plugins.summaryvault")

# ── Module-level state ────────────────────────────────────────────────

_state = {
    "client": None,
    "queue": None,
    "config": {},
    "server_url": "http://192.168.0.209:6767",
    "api_key": "",
    "session_tags": ["hermes-auto"],
    "auto_capture": True,
    "max_length": 100000,
}

# In-memory storage: session_id -> last_assistant_output
_last_output: dict[str, str] = {}
_last_output_lock = threading.Lock()

# Dedup: set of content_hashes submitted this session
_submitted_hashes: set[str] = set()
_dedup_lock = threading.Lock()


# ── Config Loading ────────────────────────────────────────────────────

def _load_plugin_config():
    """Read plugin config from ~/.hermes/config.yaml directly.

    Hermes PluginContext does NOT expose config attributes. We read
    the YAML file directly since that's what `hermes config set` writes to.
    """
    import os
    from pathlib import Path

    hermes_home = Path(os.environ.get("HERMES_HOME", Path.home() / ".hermes"))
    config_path = hermes_home / "config.yaml"
    plugin_cfg = {}

    if config_path.exists():
        try:
            import yaml
            with open(config_path) as f:
                raw = yaml.safe_load(f) or {}
            plugin_cfg = raw.get("plugins", {}).get("summaryvault", {})
        except Exception:
            pass

    _state["config"] = plugin_cfg
    _state["server_url"] = plugin_cfg.get("server_url", "http://192.168.0.209:6767")
    _state["api_key"] = plugin_cfg.get("api_key", "")
    _state["session_tags"] = plugin_cfg.get("tags", ["hermes-auto"])
    _state["auto_capture"] = plugin_cfg.get("auto_capture", True)
    _state["max_length"] = plugin_cfg.get("max_content_length", 100000)


# ── Plugin Registration ───────────────────────────────────────────────

def register(ctx):
    """Called by Hermes when the plugin is loaded."""
    _load_plugin_config()

    _state["client"] = SummaryVaultClient(
        server_url=_state["server_url"],
        api_key=_state["api_key"],
    )
    _state["queue"] = SubmissionQueue()

    # Register hooks — parameter names MUST match Hermes API exactly
    ctx.register_hook("transform_llm_output", _on_transform_llm_output)
    ctx.register_hook("on_session_finalize", _on_session_finalize)

    # Register tool
    ctx.register_tool(
        name="vault_submit",
        toolset="hermes",
        schema={
            "name": "vault_submit",
            "description": (
                "Submit a summary to SummaryVault for encrypted "
                "archival. Use this to permanently save important "
                "work results, findings, or analysis."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Title for the summary",
                    },
                    "content": {
                        "type": "string",
                        "description": "The summary content to archive",
                    },
                    "tags": {
                        "type": "string",
                        "description": "Comma-separated tags",
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Session identifier",
                    },
                },
                "required": ["title", "content"],
            },
        },
        handler=_tool_vault_submit,
    )

    log.info(
        "SummaryVault plugin loaded (server: %s, auto_capture: %s)",
        _state["server_url"], _state["auto_capture"],
    )


def unregister(ctx):
    """Called by Hermes when the plugin is unloaded."""
    log.info("SummaryVault plugin unloaded")
    _state["client"] = None
    _state["queue"] = None


# ── transform_llm_output Hook ────────────────────────────────────────
# Hermes API: (response_text, session_id, model, platform, **kwargs) -> str | None
# Returns None to pass through unchanged (observer pattern).

def _on_transform_llm_output(
    response_text: str,
    session_id: str | None = None,
    model: str | None = None,
    platform: str | None = None,
    **kwargs,
) -> str | None:
    """Capture assistant's final response. Observer pattern — returns None."""
    if not response_text or not response_text.strip():
        return None

    sid = session_id or "default"
    with _last_output_lock:
        _last_output[sid] = response_text

    return None  # pass through unchanged


# ── on_session_finalize Hook ──────────────────────────────────────────
# Hermes API: (session_id, platform, **kwargs) -> None

def _on_session_finalize(
    session_id: str | None = None,
    platform: str | None = None,
    **kwargs,
) -> None:
    """Session ended — submit last captured output to vault."""
    if not _state["auto_capture"]:
        return

    sid = session_id or "default"

    with _last_output_lock:
        last_output = _last_output.pop(sid, None)

    if not last_output or len(last_output.strip()) < 10:
        log.debug("Session %s: no substantial output to capture", sid)
        return

    content = last_output[:_state["max_length"]]
    first_line = content.split("\n")[0].strip()
    title = first_line[:100] if first_line else f"Summary {sid[:8]}"

    # Try to get agent_name/model from kwargs (Hermes may or may not pass these)
    agent_name = kwargs.get("agent_name", "Hermes Agent") or "Hermes Agent"
    model = kwargs.get("model", "") or ""

    _submit_summary(
        content=content,
        title=title,
        session_id=sid,
        agent_name=agent_name,
        model=model,
        tags=_state["session_tags"],
        metadata={"session_id": sid, "plugin_version": "1.0.0"},
    )


# ── Tool Handler ──────────────────────────────────────────────────────
# Hermes handler API: (args: dict, **kwargs) -> str (JSON)

def _tool_vault_submit(args: dict, **kwargs) -> str:
    """Handle vault_submit tool call."""
    title = args.get("title", "Untitled Summary")
    content = args.get("content", "")
    tags_str = args.get("tags", "")
    session_id = args.get(
        "session_id",
        f"manual_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
    )

    if not content:
        return json.dumps({"error": "content is required"})

    result = _submit_summary(
        content=content,
        title=title,
        session_id=session_id,
        tags=[t.strip() for t in tags_str.split(",") if t.strip()],
        manual=True,
    )

    if result.get("success"):
        return json.dumps({
            "result": f"Summary submitted to SummaryVault. ID: {result.get('id', 'unknown')}",
        })
    return json.dumps({"error": f"Failed: {result.get('error', 'Unknown')}"})


# ── Submission Logic ──────────────────────────────────────────────────

def _submit_summary(content, title, session_id, agent_name="Hermes Agent",
                    model="", tags=None, metadata=None, manual=False):
    """Submit a summary to the vault with idempotency and offline queuing."""
    if tags is None:
        tags = list(_state["session_tags"])

    h = hashlib.sha256()
    h.update(session_id.encode("utf-8"))
    h.update(content.encode("utf-8"))
    content_hash = h.hexdigest()

    with _dedup_lock:
        if content_hash in _submitted_hashes:
            log.debug("Duplicate submission prevented: %s", content_hash[:12])
            return {"success": True, "status": "duplicate", "id": None}
        _submitted_hashes.add(content_hash)

    payload = {
        "content": content,
        "title": title,
        "session_id": session_id,
        "agent_name": agent_name,
        "model": model,
        "tags": tags,
        "metadata": metadata or {},
    }

    client = _state.get("client")
    queue = _state.get("queue")

    if client and client.is_configured:
        try:
            result = client.submit(payload)
            log.info("Summary submitted: %s (session: %s, id: %s)",
                     title[:50], session_id[:12], result.get("id", "?")[:8])
            return {"success": True, **result}
        except Exception as e:
            log.warning("Submission failed, queuing: %s - %s", title[:30], e)
            if queue:
                queue.enqueue(payload, content_hash)
            return {"success": False, "error": str(e), "queued": True}
    else:
        log.info("Vault not configured, queuing locally")
        if queue:
            queue.enqueue(payload, content_hash)
        return {"success": False, "error": "Vault not configured", "queued": True}
