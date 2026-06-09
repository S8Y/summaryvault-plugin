"""
SummaryVault Plugin — Offline Queue

Persistent queue for submissions that fail due to server unavailability.
Uses SQLite for persistence across Hermes restarts.
"""

import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone, timedelta
from pathlib import Path

log = logging.getLogger("hermes.plugins.summaryvault.queue")

# Retry schedule (seconds between attempts)
RETRY_SCHEDULE = [10, 30, 60, 300, 900, 1800, 3600, 7200, 14400, 28800]

MAX_RETRIES = len(RETRY_SCHEDULE)
MAX_QUEUE_SIZE = 1000


class SubmissionQueue:
    """Persistent submission queue with exponential backoff."""

    def __init__(self, db_path: str | None = None):
        if db_path is None:
            home = Path.home()
            db_path = str(
                home / ".hermes" / "plugins" / "summaryvault" / "queue.db"
            )

        self._db_path = db_path
        self._lock = threading.Lock()

        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _init_db(self):
        """Initialize the queue database schema."""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            try:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS submission_queue (
                        id TEXT PRIMARY KEY,
                        content TEXT NOT NULL,
                        title TEXT NOT NULL,
                        session_id TEXT NOT NULL,
                        agent_name TEXT DEFAULT '',
                        model TEXT DEFAULT '',
                        tags TEXT DEFAULT '[]',
                        metadata_json TEXT DEFAULT '{}',
                        content_hash TEXT NOT NULL,
                        retry_count INTEGER DEFAULT 0,
                        next_retry_at TEXT,
                        status TEXT DEFAULT 'pending',
                        created_at TEXT NOT NULL,
                        last_error TEXT
                    )
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_queue_status
                    ON submission_queue(status)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_queue_retry
                    ON submission_queue(next_retry_at)
                """)
                conn.commit()
            finally:
                conn.close()

    def enqueue(
        self,
        payload: dict,
        content_hash: str,
    ) -> str:
        """
        Add an item to the queue.

        Returns the queue item ID.
        """
        item_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        next_retry = (
            datetime.now(timezone.utc) + timedelta(seconds=RETRY_SCHEDULE[0])
        ).isoformat()

        with self._lock:
            # Check queue size
            conn = sqlite3.connect(self._db_path)
            try:
                count = conn.execute(
                    "SELECT count(*) FROM submission_queue WHERE status = 'pending'"
                ).fetchone()[0]

                if count >= MAX_QUEUE_SIZE:
                    # Remove oldest pending item
                    conn.execute(
                        """DELETE FROM submission_queue
                           WHERE id IN (
                               SELECT id FROM submission_queue
                               WHERE status = 'pending'
                               ORDER BY created_at ASC
                               LIMIT 1
                           )"""
                    )

                conn.execute(
                    """INSERT INTO submission_queue
                       (id, content, title, session_id, agent_name, model,
                        tags, metadata_json, content_hash, retry_count,
                        next_retry_at, status, created_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?, 'pending', ?)""",
                    (
                        item_id,
                        payload.get("content", ""),
                        payload.get("title", ""),
                        payload.get("session_id", ""),
                        payload.get("agent_name", ""),
                        payload.get("model", ""),
                        json.dumps(payload.get("tags", [])),
                        json.dumps(payload.get("metadata", {})),
                        content_hash,
                        next_retry,
                        now,
                    ),
                )
                conn.commit()
            finally:
                conn.close()

        log.info("Queued submission %s (retry in %ss)", item_id[:8], RETRY_SCHEDULE[0])
        return item_id

    def dequeue_pending(self) -> list[dict]:
        """
        Get all items that are ready for retry.

        Returns list of dicts with queue item data.
        """
        now = datetime.now(timezone.utc).isoformat()

        with self._lock:
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            try:
                rows = conn.execute(
                    """SELECT * FROM submission_queue
                       WHERE status = 'pending'
                       AND (next_retry_at IS NULL OR next_retry_at <= ?)
                       ORDER BY created_at ASC
                       LIMIT 50""",
                    (now,),
                ).fetchall()

                items = []
                for row in rows:
                    items.append({
                        "id": row["id"],
                        "content": row["content"],
                        "title": row["title"],
                        "session_id": row["session_id"],
                        "agent_name": row["agent_name"],
                        "model": row["model"],
                        "tags": json.loads(row["tags"] or "[]"),
                        "metadata": json.loads(row["metadata_json"] or "{}"),
                        "content_hash": row["content_hash"],
                        "retry_count": row["retry_count"],
                    })
                return items
            finally:
                conn.close()

    def mark_success(self, item_id: str) -> None:
        """Mark a queue item as successfully submitted."""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            try:
                conn.execute(
                    "DELETE FROM submission_queue WHERE id = ?",
                    (item_id,),
                )
                conn.commit()
            finally:
                conn.close()

    def mark_failed(self, item_id: str, error: str) -> None:
        """
        Mark a queue item as failed and schedule next retry.
        If max retries exceeded, mark as failed permanently.
        """
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            try:
                row = conn.execute(
                    "SELECT retry_count FROM submission_queue WHERE id = ?",
                    (item_id,),
                ).fetchone()

                if not row:
                    return

                retry_count = row[0] + 1

                if retry_count >= MAX_RETRIES:
                    conn.execute(
                        "UPDATE submission_queue SET status = 'failed', retry_count = ?, last_error = ? WHERE id = ?",
                        (retry_count, error[:500], item_id),
                    )
                    log.warning(
                        "Queue item %s failed permanently (max retries)", item_id[:8]
                    )
                else:
                    next_retry = (
                        datetime.now(timezone.utc)
                        + timedelta(seconds=RETRY_SCHEDULE[retry_count])
                    ).isoformat()
                    conn.execute(
                        """UPDATE submission_queue
                           SET retry_count = ?, next_retry_at = ?, last_error = ?
                           WHERE id = ?""",
                        (retry_count, next_retry, error[:500], item_id),
                    )
                conn.commit()
            finally:
                conn.close()

    def get_stats(self) -> dict:
        """Get queue statistics."""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            try:
                pending = conn.execute(
                    "SELECT count(*) FROM submission_queue WHERE status = 'pending'"
                ).fetchone()[0]
                failed = conn.execute(
                    "SELECT count(*) FROM submission_queue WHERE status = 'failed'"
                ).fetchone()[0]
                return {
                    "pending": pending,
                    "failed": failed,
                    "total": pending + failed,
                }
            finally:
                conn.close()

    def clear_failed(self) -> int:
        """Clear all permanently failed items. Returns count cleared."""
        with self._lock:
            conn = sqlite3.connect(self._db_path)
            try:
                count = conn.execute(
                    "DELETE FROM submission_queue WHERE status = 'failed'"
                ).rowcount
                conn.commit()
                return count or 0
            finally:
                conn.close()
