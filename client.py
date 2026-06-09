"""
SummaryVault Plugin — HTTP Client

Thin async HTTP client for submitting summaries to the SummaryVault server.
Uses Python stdlib only (urllib) — no external dependencies.
"""

import json
import logging
import urllib.request
import urllib.error
import urllib.parse

log = logging.getLogger("hermes.plugins.summaryvault.client")


class SummaryVaultClient:
    """HTTP client for the SummaryVault REST API."""

    def __init__(self, server_url: str, api_key: str):
        self._server_url = server_url.rstrip("/")
        self._api_key = api_key
        self._timeout = 10  # seconds

    @property
    def is_configured(self) -> bool:
        """Check if the client has enough config to submit."""
        return bool(self._server_url) and bool(self._api_key)

    def submit(self, payload: dict) -> dict:
        """
        Submit a summary to the vault.

        Args:
            payload: dict with content, title, session_id, agent_name,
                     model, tags, metadata

        Returns:
            dict with id, status, encrypted, created_at

        Raises:
            ConnectionError: if server is unreachable
            ValueError: if response is invalid
        """
        url = f"{self._server_url}/api/v1/submit"
        data = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
                "User-Agent": "SummaryVaultPlugin/1.0",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = resp.read().decode("utf-8")
                result = json.loads(body)
                return result
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            detail = f"HTTP {e.code}: {body[:200]}"
            log.error("Submission HTTP error: %s", detail)
            raise ConnectionError(detail) from e
        except urllib.error.URLError as e:
            log.error("Submission connection error: %s", e.reason)
            raise ConnectionError(str(e.reason)) from e
        except json.JSONDecodeError as e:
            log.error("Invalid JSON response: %s", e)
            raise ValueError(f"Invalid response: {e}") from e

    def check_health(self) -> dict | None:
        """
        Check if the vault server is reachable.

        Returns:
            Status dict if healthy, None if unreachable.
        """
        url = f"{self._server_url}/api/v1/status"

        req = urllib.request.Request(url, method="GET")

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)
        except Exception:
            return None
