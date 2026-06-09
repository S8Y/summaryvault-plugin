"""
SummaryVault — Hermes Plugin

Captures final session summaries and submits them to SummaryVault
for encrypted archival.

Hook Lifecycle:
  1. transform_llm_output: Captures each turn's final response
     (observer pattern — never modifies output)
  2. on_session_finalize: On session end, retrieves last captured output
     and submits to SummaryVault server

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

# ── Module-level state (used by register() pattern) ────────────────────
# Hermes calls register(ctx) on plugin load. We store state here so
# hook handler functions can access it without a class instance.

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


# ── Plugin Registration ───────────────────────────────────────────────

def register(ctx):
    """Called by Hermes when the plugin is loaded."""
    # Read plugin config from Hermes config
    # Hermes stores: config["plugins"]["summaryvault"]["key"] = value
    plugins_cfg = ctx.config.get("plugins", {})
    plugin_cfg = plugins_cfg.get("summaryvault", {})

    _state["config"] = plugin_cfg
    _state["server_url"] = plugin_cfg.get("server_url", "http://192.168.0.209:6767")
    _state["api_key"] = plugin_cfg.get("api_key", "")
    _state["session_tags"] = plugin_cfg.get("tags", ["hermes-auto"])
    _state["auto_capture"] = plugin_cfg.get("auto_capture", True)
    _state["max_length"] = plugin_cfg.get("max_content_length", 100000)

    # Create client and queue
    _state["client"] = SummaryVaultClient(
        server_url=_state["server_url"],
        api_key=_state["api_key"],
    )
    _state["queue"] = SubmissionQueue()

    # Register hooks
    ctx.register_hook("transform_llm_output", _on_transform_llm_output)
    ctx.register_hook("on_session_finalize", _on_session_finalize)

    # Register tools
    from hermes_plugin import ToolSchema, ToolParameter
    ctx.register_tool(
        name="vault_submit",
        toolset="hermes",
        schema=ToolSchema(
            name="vault_submit",
            description=(
                "Submit a summary to SummaryVault for encrypted "
                "archival. Use this to permanently save important "
                "work results, findings, or analysis."
            ),
            parameters={
                "title": ToolParameter(
                    type="string",
                    description="Title for the summary",
                    required=True,
                ),
                "content": ToolParameter(
                    type="string",
                    description="The summary content to archive",
                    required=True,
                ),
                "tags": ToolParameter(
                    type="string",
                    description="Comma-separated tags",
                    required=False,
                ),
                "session_id": ToolParameter(
                    type="string",
                    description="Session identifier (auto-detected if omitted)",
                    required=False,
                ),
            },
        ),
        handler=_tool_vault_submit,
    )

    log.info(
        "SummaryVault plugin loaded (server: %s, auto_capture: %s)",
        _state["server_url"], _state["auto_capture"],
    )


def unregister(ctx):
    """Called by Hermes when the plugin is unloaded."""
    log.info("SummaryVault plugin unloaded")
    if _state["queue"]:
        _state["queue"] = None
    if _state["client"]:
        _state["client"] = None


# ── Hooks ─────────────────────────────────────────────────────────────

def _on_transform_llm_output(output: str, **kwargs) -> str:
    """
    Capture the assistant's final response text.

    This fires every turn. We only keep the LAST response per session.
    The output is returned unmodified (observer pattern).
    """
    if not output or not output.strip():
        return output

    session_id = kwargs.get("session_id", "default")

    with _last_output_lock:
        _last_output[session_id] = output

    return output


def _on_session_finalize(**kwargs) -> None:
    """
    Session is ending. Capture the last output and submit to vault.

    This fires once when a session ends (not every turn).
    """
    if not _state["auto_capture"]:
        return

    session_id = kwargs.get("session_id", "default")
    agent_name = kwargs.get("agent_name", "Hermes Agent") or "Hermes Agent"
    model = kwargs.get("model", "") or ""

    # Get the last captured output
    with _last_output_lock:
        last_output = _last_output.pop(session_id, None)

    if not last_output or len(last_output.strip()) < 10:
        log.debug(
            "Session %s: no substantial output to capture", session_id
        )
        return

    # Truncate if needed
    content = last_output[:_state["max_length"]]

    # Build title from first line or first N chars
    first_line = content.split("\n")[0].strip()
    title = first_line[:100] if first_line else f"Summary {session_id[:8]}"

    # Build metadata
    metadata = {
        "session_id": session_id,
        "plugin_version": "1.0.0",
    }

    # Submit
    _submit_summary(
        content=content,
        title=title,
        session_id=session_id,
        agent_name=agent_name,
        model=model,
        tags=_state["session_tags"],
        metadata=metadata,
    )


# ── Tool Handler ──────────────────────────────────────────────────────

def _tool_vault_submit(args: dict, **kwargs) -> str:
    """Handle vault_submit tool call from the agent."""
    title = args.get("title", "Untitled Summary")
    content = args.get("content", "")
    tags_str = args.get("tags", "")
    session_id = args.get(
        "session_id",
        f"manual_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
    )

    if not content:
        return json.dumps({"error": "content is required"})

    tags = [t.strip() for t in tags_str.split(",") if t.strip()]

    result = _submit_summary(
        content=content,
        title=title,
        session_id=session_id,
        tags=tags,
        manual=True,
    )

    if result.get("success"):
        return json.dumps({
            "result": f"Summary submitted to SummaryVault. ID: {result.get('id', 'unknown')}, Status: {result.get('status', 'stored')}",
        })
    else:
        return json.dumps({
            "error": f"Failed to submit: {result.get('error', 'Unknown error')}",
        })


# ── Submission Logic ──────────────────────────────────────────────────

def _submit_summary(
    content: str,
    title: str,
    session_id: str,
    agent_name: str = "Hermes Agent",
    model: str = "",
    tags: list[str] | None = None,
    metadata: dict | None = None,
    manual: bool = False,
) -> dict:
    """
    Submit a summary to the vault.
    Handles idempotency and offline queuing.
    """
    if tags is None:
        tags = list(_state["session_tags"])

    # Compute content hash for dedup
    h = hashlib.sha256()
    h.update(session_id.encode("utf-8"))
    h.update(content.encode("utf-8"))
    content_hash = h.hexdigest()

    # Check in-memory dedup
    with _dedup_lock:
        if content_hash in _submitted_hashes:
            log.debug("Duplicate submission prevented: %s", content_hash[:12])
            return {"success": True, "status": "duplicate", "id": None}
        _submitted_hashes.add(content_hash)

    # Build request payload
    payload = {
        "content": content,
        "title": title,
        "session_id": session_id,
        "agent_name": agent_name,
        "model": model,
        "tags": tags,
        "metadata": metadata or {},
    }

    # Attempt submission
    client = _state.get("client")
    queue = _state.get("queue")

    if client and client.is_configured:
        try:
            result = client.submit(payload)
            log.info(
                "Summary submitted: %s (session: %s, id: %s)",
                title[:50], session_id[:12], result.get("id", "?")[:8],
            )
            return {"success": True, **result}
        except Exception as e:
            log.warning(
                "Submission failed, queuing: %s (session: %s) - %s",
                title[:30], session_id[:12], e,
            )
            if queue:
                queue.enqueue(payload, content_hash)

            if manual:
                return {
                    "success": False,
                    "error": f"Server unreachable, queued for retry: {e}",
                    "queued": True,
                }
            return {"success": False, "error": str(e), "queued": True}
    else:
        log.info("Vault not configured, queuing submission locally")
        if queue:
            queue.enqueue(payload, content_hash)
        return {"success": False, "error": "Vault not configured", "queued": True}
