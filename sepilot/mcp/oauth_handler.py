"""OAuth 2.0 handler for MCP remote servers.

Supports OAuth 2.0 authorization code flow with PKCE for
authenticating with remote MCP servers.
"""

import asyncio
import base64
import hashlib
import json
import logging
import secrets
import urllib.parse
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass
class OAuthConfig:
    """OAuth 2.0 configuration"""
    client_id: str
    auth_url: str
    token_url: str
    scopes: list[str] = field(default_factory=list)
    redirect_uri: str = "http://localhost:8765/callback"
    client_secret: str | None = None  # Optional for public clients with PKCE


@dataclass
class OAuthToken:
    """OAuth 2.0 token"""
    access_token: str
    token_type: str = "Bearer"
    expires_at: datetime | None = None
    refresh_token: str | None = None
    scope: str = ""

    def is_expired(self) -> bool:
        """Check if token is expired"""
        if self.expires_at is None:
            return False
        return datetime.now() >= self.expires_at

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "access_token": self.access_token,
            "token_type": self.token_type,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "refresh_token": self.refresh_token,
            "scope": self.scope,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OAuthToken":
        """Create from dictionary"""
        expires_at = None
        if data.get("expires_at"):
            expires_at = datetime.fromisoformat(data["expires_at"])

        return cls(
            access_token=data["access_token"],
            token_type=data.get("token_type", "Bearer"),
            expires_at=expires_at,
            refresh_token=data.get("refresh_token"),
            scope=data.get("scope", ""),
        )


class OAuthCallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler for OAuth callback"""

    def __init__(self, *args, callback_event: asyncio.Event, **kwargs):
        self.callback_event = callback_event
        super().__init__(*args, **kwargs)

    def log_message(self, format, *args):
        """Suppress default logging"""
        pass

    def do_GET(self):
        """Handle OAuth callback"""
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)

        # Store authorization code on server instance
        self.server.auth_code = params.get("code", [None])[0]
        self.server.error = params.get("error", [None])[0]
        self.server.error_description = params.get("error_description", [""])[0]

        # Send response
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()

        if self.server.auth_code:
            html = """
            <html><body style="font-family: system-ui; text-align: center; padding: 50px;">
                <h1>Authentication Successful</h1>
                <p>You can close this window and return to SE Pilot.</p>
                <script>setTimeout(() => window.close(), 3000);</script>
            </body></html>
            """
        else:
            error_msg = self.server.error_description or self.server.error or "Unknown error"
            html = f"""
            <html><body style="font-family: system-ui; text-align: center; padding: 50px;">
                <h1>Authentication Failed</h1>
                <p style="color: red;">{error_msg}</p>
                <p>Please close this window and try again.</p>
            </body></html>
            """

        self.wfile.write(html.encode())

        # Signal completion
        self.callback_event.set()


class OAuthHandler:
    """OAuth 2.0 handler with PKCE support"""

    TOKEN_CACHE_FILE = Path.home() / ".sepilot" / "oauth_tokens.json"

    def __init__(self, config: OAuthConfig):
        """Initialize OAuth handler

        Args:
            config: OAuth configuration
        """
        self.config = config
        self._token_cache: dict[str, OAuthToken] = {}
        self._load_token_cache()

    def _load_token_cache(self) -> None:
        """Load tokens from cache file"""
        if self.TOKEN_CACHE_FILE.exists():
            try:
                with open(self.TOKEN_CACHE_FILE, encoding="utf-8") as f:
                    data = json.load(f)
                    for key, token_data in data.items():
                        self._token_cache[key] = OAuthToken.from_dict(token_data)
                logger.debug(f"Loaded {len(self._token_cache)} cached OAuth tokens")
            except Exception as e:
                logger.warning(f"Failed to load OAuth token cache: {e}")

    def _save_token_cache(self) -> None:
        """Save tokens to cache file with restricted permissions (0600)"""
        try:
            self.TOKEN_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            data = {
                key: token.to_dict()
                for key, token in self._token_cache.items()
            }
            with open(self.TOKEN_CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            # Restrict token file to owner-only (contains secrets)
            self.TOKEN_CACHE_FILE.chmod(0o600)
        except Exception as e:
            logger.warning(f"Failed to save OAuth token cache: {e}")

    def _get_cache_key(self) -> str:
        """Get cache key for current OAuth config"""
        return f"{self.config.auth_url}:{self.config.client_id}"

    def get_cached_token(self) -> OAuthToken | None:
        """Get cached token if valid

        Returns:
            Valid token or None
        """
        cache_key = self._get_cache_key()
        token = self._token_cache.get(cache_key)

        if token and not token.is_expired():
            return token

        return None

    def _generate_pkce(self) -> tuple[str, str]:
        """Generate PKCE code verifier and challenge

        Returns:
            Tuple of (code_verifier, code_challenge)
        """
        # Generate code verifier
        code_verifier = secrets.token_urlsafe(64)

        # Generate code challenge (S256)
        digest = hashlib.sha256(code_verifier.encode()).digest()
        code_challenge = base64.urlsafe_b64encode(digest).decode().rstrip("=")

        return code_verifier, code_challenge

    async def authenticate(self) -> OAuthToken:
        """Perform OAuth 2.0 authentication with PKCE

        Opens browser for user authentication and waits for callback.

        Returns:
            OAuth token

        Raises:
            Exception: If authentication fails
        """
        # Check for cached token
        cached = self.get_cached_token()
        if cached:
            logger.info("Using cached OAuth token")
            return cached

        # Try to refresh if we have a refresh token
        cache_key = self._get_cache_key()
        old_token = self._token_cache.get(cache_key)
        if old_token and old_token.refresh_token:
            try:
                refreshed = await self._refresh_token(old_token.refresh_token)
                self._token_cache[cache_key] = refreshed
                self._save_token_cache()
                return refreshed
            except Exception as e:
                logger.debug(f"Token refresh failed: {e}")

        # Generate PKCE
        code_verifier, code_challenge = self._generate_pkce()

        # Generate state
        state = secrets.token_urlsafe(32)

        # Build authorization URL
        auth_params = {
            "client_id": self.config.client_id,
            "response_type": "code",
            "redirect_uri": self.config.redirect_uri,
            "scope": " ".join(self.config.scopes),
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
        auth_url = f"{self.config.auth_url}?{urllib.parse.urlencode(auth_params)}"

        # Start callback server
        callback_event = asyncio.Event()
        port = int(self.config.redirect_uri.split(":")[-1].split("/")[0])

        server = HTTPServer(
            ("localhost", port),
            lambda *args, **kwargs: OAuthCallbackHandler(
                *args, callback_event=callback_event, **kwargs
            ),
        )
        server.auth_code = None
        server.error = None
        server.error_description = ""

        # Run server in thread
        server_thread = Thread(target=server.handle_request, daemon=True)
        server_thread.start()

        # Open browser
        logger.info("Opening browser for OAuth authentication...")
        print("\n🔐 Opening browser for authentication...")
        print(f"   If browser doesn't open, visit: {auth_url}\n")
        webbrowser.open(auth_url)

        # Wait for callback
        await callback_event.wait()
        server.server_close()

        # Check for errors
        if server.error:
            raise Exception(
                f"OAuth error: {server.error} - {server.error_description}"
            )

        if not server.auth_code:
            raise Exception("No authorization code received")

        # Exchange code for token
        token = await self._exchange_code(server.auth_code, code_verifier)

        # Cache token
        self._token_cache[cache_key] = token
        self._save_token_cache()

        logger.info("OAuth authentication successful")
        return token

    async def _exchange_code(
        self, code: str, code_verifier: str
    ) -> OAuthToken:
        """Exchange authorization code for token

        Args:
            code: Authorization code
            code_verifier: PKCE code verifier

        Returns:
            OAuth token
        """
        data = {
            "grant_type": "authorization_code",
            "client_id": self.config.client_id,
            "code": code,
            "redirect_uri": self.config.redirect_uri,
            "code_verifier": code_verifier,
        }

        if self.config.client_secret:
            data["client_secret"] = self.config.client_secret

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.config.token_url,
                data=data,
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()
            token_data = response.json()

        return self._parse_token_response(token_data)

    async def _refresh_token(self, refresh_token: str) -> OAuthToken:
        """Refresh an expired token

        Args:
            refresh_token: Refresh token

        Returns:
            New OAuth token
        """
        data = {
            "grant_type": "refresh_token",
            "client_id": self.config.client_id,
            "refresh_token": refresh_token,
        }

        if self.config.client_secret:
            data["client_secret"] = self.config.client_secret

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.config.token_url,
                data=data,
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()
            token_data = response.json()

        return self._parse_token_response(token_data)

    def _parse_token_response(self, data: dict[str, Any]) -> OAuthToken:
        """Parse token response

        Args:
            data: Token response data

        Returns:
            OAuth token
        """
        expires_at = None
        if "expires_in" in data:
            expires_at = datetime.now() + timedelta(seconds=data["expires_in"])

        return OAuthToken(
            access_token=data["access_token"],
            token_type=data.get("token_type", "Bearer"),
            expires_at=expires_at,
            refresh_token=data.get("refresh_token"),
            scope=data.get("scope", ""),
        )

    def get_auth_headers(self) -> dict[str, str]:
        """Get authorization headers for requests

        Returns:
            Dictionary with Authorization header

        Raises:
            Exception: If no valid token available
        """
        token = self.get_cached_token()
        if not token:
            raise Exception("No valid OAuth token. Call authenticate() first.")

        return {
            "Authorization": f"{token.token_type} {token.access_token}"
        }

    def clear_cache(self) -> None:
        """Clear token cache for this config"""
        cache_key = self._get_cache_key()
        self._token_cache.pop(cache_key, None)
        self._save_token_cache()


async def authenticate_mcp_server(
    oauth_config: dict[str, Any]
) -> dict[str, str]:
    """Authenticate with an MCP server using OAuth

    Args:
        oauth_config: OAuth configuration dictionary with keys:
            - client_id: OAuth client ID
            - auth_url: Authorization URL
            - token_url: Token URL
            - scopes: List of scopes (optional)

    Returns:
        Dictionary with authorization headers
    """
    config = OAuthConfig(
        client_id=oauth_config["client_id"],
        auth_url=oauth_config["auth_url"],
        token_url=oauth_config["token_url"],
        scopes=oauth_config.get("scopes", []),
        redirect_uri=oauth_config.get("redirect_uri", "http://localhost:8765/callback"),
    )

    handler = OAuthHandler(config)
    await handler.authenticate()

    return handler.get_auth_headers()
