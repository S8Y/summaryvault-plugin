"""
SummaryVault Plugin — Data Schemas

Pydantic-like validation for submission payloads.
Uses dataclasses to avoid external dependencies.
"""

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class SummaryPayload:
    """Payload for submitting a summary to the vault."""

    content: str
    title: str
    session_id: str
    agent_name: str = "Hermes Agent"
    model: str = ""
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    turn_count: int = 0
    duration_seconds: int = 0
    provider: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> list[str]:
        """Validate payload. Returns list of error messages."""
        errors = []

        if not self.content or not self.content.strip():
            errors.append("content is required")
        elif len(self.content) > 100000:
            errors.append("content exceeds maximum length (100000)")

        if not self.title or not self.title.strip():
            errors.append("title is required")
        elif len(self.title) > 500:
            errors.append("title exceeds maximum length (500)")

        if not self.session_id:
            errors.append("session_id is required")

        if len(self.tags) > 20:
            errors.append("too many tags (max 20)")

        for tag in self.tags:
            if len(tag) > 50:
                errors.append(f"tag too long: '{tag[:20]}...' (max 50 chars)")

        return errors


@dataclass
class VaultStatus:
    """Vault status information."""

    initialized: bool = False
    unlocked: bool = False
    summary_count: int = 0
    cipher_suite: str = ""


@dataclass
class QueueItem:
    """An item in the submission queue."""

    id: str
    content: str
    title: str
    session_id: str
    agent_name: str = ""
    model: str = ""
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""
    retry_count: int = 0
    status: str = "pending"
    created_at: str = ""
    last_error: str = ""
