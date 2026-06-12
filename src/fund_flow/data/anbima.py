"""ANBIMA API client — OAuth2 client-credentials flow + authenticated GET.

Auth flow (per ANBIMA dev guide):
  1. POST /oauth/access-token with Authorization: Basic base64(client_id:secret)
     and body {"grant_type": "client_credentials"} → short-lived access_token.
  2. Data calls send headers `client_id` and `access_token`.

The token is cached in-memory until shortly before expiry and refreshed
transparently.
"""
from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.parse
import urllib.request

PROD_BASE = "https://api.anbima.com.br"
SANDBOX_BASE = "https://api.sandbox.anbima.com.br"


class AnbimaError(RuntimeError):
    """Raised on ANBIMA auth or data-request failure."""


class AnbimaClient:
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        base_url: str = PROD_BASE,
        timeout: float = 30.0,
    ):
        self._id = client_id
        self._secret = client_secret
        self._base = base_url.rstrip("/")
        self._timeout = timeout
        self._token: str | None = None
        self._expires_at: float = 0.0

    # --- OAuth ---

    def _fetch_token(self) -> tuple[str, int]:
        auth = base64.b64encode(f"{self._id}:{self._secret}".encode()).decode()
        req = urllib.request.Request(
            f"{self._base}/oauth/access-token",
            data=json.dumps({"grant_type": "client_credentials"}).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Basic {auth}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise AnbimaError(
                f"ANBIMA OAuth failed ({exc.code}); check client_id/secret"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise AnbimaError(f"ANBIMA OAuth request failed: {exc}") from exc
        return payload["access_token"], int(payload.get("expires_in", 3600))

    def access_token(self) -> str:
        """Return a valid token, refreshing ~60s before expiry."""
        if self._token is None or time.time() >= self._expires_at - 60:
            token, ttl = self._fetch_token()
            self._token = token
            self._expires_at = time.time() + ttl
        return self._token

    # --- Data ---

    def get(self, path: str, params: dict | None = None) -> dict | list:
        """Authenticated GET against a data endpoint (path relative to base)."""
        url = f"{self._base}/{path.lstrip('/')}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(
            url,
            headers={
                "Content-Type": "application/json",
                "client_id": self._id,
                "access_token": self.access_token(),
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:300]
            raise AnbimaError(f"ANBIMA GET {path} failed ({exc.code}): {body}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise AnbimaError(f"ANBIMA GET {path} request failed: {exc}") from exc
