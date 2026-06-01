# BUILT: CDPBrowser
"""
CDP Browser Client — Raw WebSocket CDP protocol, zero dependencies.
Connects to Lightpanda or Chrome via Chrome DevTools Protocol.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import secrets
import struct
from dataclasses import dataclass
from typing import Any, Optional

from loguru import logger


@dataclass
class CDPResult:
    """Result from a CDP command."""
    success: bool
    data: dict[str, Any] | None = None
    error: str | None = None


class CDPBrowser:
    """
    Minimal CDP (Chrome DevTools Protocol) client over raw WebSocket.

    No dependencies beyond Python stdlib. Connects to:
    - Lightpanda (headless browser, CDP-compatible)
    - Chrome/Chromium with --remote-debugging-port
    - Any CDP-compatible browser

    Usage:
        browser = CDPBrowser("ws://localhost:9222/devtools/browser/...")
        await browser.connect()
        await browser.navigate("https://example.com")
        content = await browser.get_page_text()
        await browser.close()
    """

    def __init__(self, cdp_url: str, timeout: float = 30.0):
        """
        Args:
            cdp_url: WebSocket URL for CDP connection.
                     Lightpanda: ws://localhost:9222
                     Chrome: ws://localhost:9222/devtools/browser/<id>
            timeout: Command timeout in seconds
        """
        self._cdp_url = cdp_url
        self._timeout = timeout
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._msg_id = 0
        self._connected = False
        self._response_futures: dict[int, asyncio.Future] = {}
        self._event_handlers: dict[str, list] = {}
        self._recv_task: asyncio.Task | None = None

    async def connect(self) -> bool:
        """
        Connect to the CDP browser via WebSocket.
        Returns True if connected successfully.
        """
        try:
            # Parse WebSocket URL
            ws_url = self._cdp_url
            if not ws_url.startswith("ws"):
                # If just a host:port, construct the CDP URL
                ws_url = f"ws://{ws_url}/devtools/browser"

            # Parse host and port from ws://host:port/path
            match = re.match(r"ws://([^:]+):(\d+)(/.*)?", ws_url)
            if not match:
                logger.error(f"[CDP] Invalid WebSocket URL: {ws_url}")
                return False

            host = match.group(1)
            port = int(match.group(2))
            path = match.group(3) or "/devtools/browser"

            # TCP connect
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=self._timeout,
            )

            # WebSocket handshake
            success = await self._ws_handshake(host, port, path)
            if not success:
                logger.error("[CDP] WebSocket handshake failed")
                return False

            self._connected = True

            # Start receiving messages
            self._recv_task = asyncio.create_task(self._recv_loop())

            # Wait for Target.attachedToTarget or similar event
            await asyncio.sleep(0.5)

            logger.info(f"[CDP] Connected to {host}:{port}")
            return True

        except asyncio.TimeoutError:
            logger.error(f"[CDP] Connection timeout to {self._cdp_url}")
            return False
        except Exception as e:
            logger.error(f"[CDP] Connection failed: {e}")
            return False

    async def _ws_handshake(self, host: str, port: int, path: str) -> bool:
        """Perform WebSocket handshake (RFC 6455)."""
        key = base64.b64encode(secrets.token_bytes(16)).decode()

        request = (
            f"GET {path} HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            f"Upgrade: websocket\r\n"
            f"Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            f"Sec-WebSocket-Version: 13\r\n"
            f"\r\n"
        )

        self._writer.write(request.encode())
        await self._writer.drain()

        # Read response
        response = await asyncio.wait_for(
            self._reader.readuntil(b"\r\n\r\n"),
            timeout=self._timeout,
        )

        if b"101" in response:
            logger.debug("[CDP] WebSocket handshake OK")
            return True
        else:
            logger.error(f"[CDP] Handshake failed: {response[:200]}")
            return False

    async def _recv_loop(self) -> None:
        """Background task to receive WebSocket frames and dispatch responses."""
        try:
            while self._connected:
                frame = await self._ws_read_frame()
                if frame is None:
                    break

                opcode, payload = frame

                if opcode == 0x1:  # Text frame
                    try:
                        msg = json.loads(payload.decode("utf-8"))
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue

                    # CDP response
                    if "id" in msg:
                        msg_id = msg["id"]
                        if msg_id in self._response_futures:
                            future = self._response_futures.pop(msg_id)
                            if "error" in msg:
                                future.set_result(CDPResult(
                                    success=False,
                                    error=msg["error"].get("message", "Unknown error"),
                                ))
                            else:
                                future.set_result(CDPResult(
                                    success=True,
                                    data=msg.get("result", {}),
                                ))

                    # CDP event
                    elif "method" in msg:
                        method = msg["method"]
                        for handler in self._event_handlers.get(method, []):
                            try:
                                handler(msg.get("params", {}))
                            except Exception as e:
                                logger.warning(f"[CDP] Event handler error: {e}")

                elif opcode == 0x9:  # Ping
                    await self._ws_send_frame(b"", opcode=0xA)  # Pong

                elif opcode == 0x8:  # Close
                    break

        except asyncio.CancelledError:
            pass
        except Exception as e:
            if self._connected:
                logger.warning(f"[CDP] Recv loop error: {e}")
        finally:
            self._connected = False

    async def _ws_read_frame(self) -> tuple[int, bytes] | None:
        """Read a single WebSocket frame."""
        try:
            # Read header (2 bytes)
            header = await asyncio.wait_for(
                self._reader.readexactly(2),
                timeout=self._timeout,
            )

            byte1 = header[0]
            byte2 = header[1]

            opcode = byte1 & 0x0F
            masked = bool(byte2 & 0x80)
            payload_len = byte2 & 0x7F

            # Extended payload length
            if payload_len == 126:
                ext = await asyncio.wait_for(
                    self._reader.readexactly(2),
                    timeout=self._timeout,
                )
                payload_len = struct.unpack("!H", ext)[0]
            elif payload_len == 127:
                ext = await asyncio.wait_for(
                    self._reader.readexactly(8),
                    timeout=self._timeout,
                )
                payload_len = struct.unpack("!Q", ext)[0]

            # Masking key (client-to-server only, we're server perspective here)
            mask_key = None
            if masked:
                mask_key = await asyncio.wait_for(
                    self._reader.readexactly(4),
                    timeout=self._timeout,
                )

            # Payload
            if payload_len > 0:
                payload = await asyncio.wait_for(
                    self._reader.readexactly(payload_len),
                    timeout=self._timeout,
                )
                if mask_key:
                    payload = bytes(b ^ mask_key[i % 4] for i, b in enumerate(payload))
            else:
                payload = b""

            return (opcode, payload)

        except (asyncio.TimeoutError, asyncio.IncompleteReadError):
            return None

    async def _ws_send_frame(self, payload: bytes, opcode: int = 0x1) -> None:
        """Send a WebSocket frame (client → server, masked)."""
        if not self._writer:
            return

        frame = bytearray()
        frame.append(0x80 | opcode)  # FIN + opcode

        length = len(payload)
        if length < 126:
            frame.append(0x80 | length)  # MASK bit set
        elif length < 65536:
            frame.append(0x80 | 126)
            frame.extend(struct.pack("!H", length))
        else:
            frame.append(0x80 | 127)
            frame.extend(struct.pack("!Q", length))

        # Masking key
        mask = secrets.token_bytes(4)
        frame.extend(mask)

        # Masked payload
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        frame.extend(masked)

        self._writer.write(bytes(frame))
        await self._writer.drain()

    async def send_command(self, method: str, params: dict | None = None) -> CDPResult:
        """Send a CDP command and wait for response."""
        if not self._connected:
            return CDPResult(success=False, error="Not connected")

        self._msg_id += 1
        msg_id = self._msg_id

        msg = {"id": msg_id, "method": method}
        if params:
            msg["params"] = params

        future: asyncio.Future[CDPResult] = asyncio.get_event_loop().create_future()
        self._response_futures[msg_id] = future

        await self._ws_send_frame(json.dumps(msg).encode())

        try:
            result = await asyncio.wait_for(future, timeout=self._timeout)
            return result
        except asyncio.TimeoutError:
            self._response_futures.pop(msg_id, None)
            return CDPResult(success=False, error=f"Command timeout: {method}")

    # ── High-level browser operations ────────────────────────────────────────

    async def navigate(self, url: str) -> CDPResult:
        """Navigate to a URL."""
        return await self.send_command("Page.navigate", {"url": url})

    async def wait_for_load(self, state: str = "networkidle", timeout_ms: int = 15000) -> CDPResult:
        """Wait for page load state."""
        # Use Page.lifecycleEvent or simple delay
        await asyncio.sleep(min(timeout_ms / 1000, 3.0))

        # Check if page is loaded via Runtime.evaluate
        result = await self.send_command("Runtime.evaluate", {
            "expression": "document.readyState",
            "returnByValue": True,
        })
        return result

    async def get_title(self) -> str:
        """Get page title."""
        result = await self.send_command("Runtime.evaluate", {
            "expression": "document.title",
            "returnByValue": True,
        })
        if result.success and result.data:
            return result.data.get("result", {}).get("value", "")
        return ""

    async def get_url(self) -> str:
        """Get current URL."""
        result = await self.send_command("Runtime.evaluate", {
            "expression": "window.location.href",
            "returnByValue": True,
        })
        if result.success and result.data:
            return result.data.get("result", {}).get("value", "")
        return ""

    async def get_page_text(self, max_chars: int = 5000) -> str:
        """Extract readable text from the page."""
        js = f"""
        (() => {{
            // Remove scripts, styles, nav, footer
            const selectors = 'script, style, nav, footer, header, aside, .sidebar, .menu, .nav';
            document.querySelectorAll(selectors).forEach(el => el.remove());

            // Extract text from main content areas
            const main = document.querySelector('main, article, .content, .main, [role="main"], #content')
                || document.body;

            // Get text with structure
            const parts = [];
            const walker = document.createTreeWalker(main, NodeFilter.SHOW_TEXT, {{
                acceptNode: (node) => {{
                    const text = node.textContent.trim();
                    if (text.length < 5) return NodeFilter.FILTER_REJECT;
                    const parent = node.parentElement;
                    if (!parent) return NodeFilter.FILTER_REJECT;
                    const tag = parent.tagName.toLowerCase();
                    if (['script', 'style', 'noscript'].includes(tag)) return NodeFilter.FILTER_REJECT;
                    return NodeFilter.FILTER_ACCEPT;
                }}
            }});

            let node;
            while (node = walker.nextNode()) {{
                const text = node.textContent.trim();
                if (text) parts.push(text);
            }}

            return parts.join(' ').substring(0, {max_chars});
        }})()
        """
        result = await self.send_command("Runtime.evaluate", {
            "expression": js,
            "returnByValue": True,
        })
        if result.success and result.data:
            return result.data.get("result", {}).get("value", "")
        return ""

    async def get_accessibility_tree(self, max_depth: int = 3) -> str:
        """Get accessibility tree snapshot (like agent-browser snapshot -i)."""
        js = f"""
        (() => {{
            function getNode(el, depth) {{
                if (depth > {max_depth}) return '';
                const role = el.getAttribute?.('role') || el.tagName?.toLowerCase() || '';
                const text = el.textContent?.trim()?.substring(0, 100) || '';
                const name = el.getAttribute?.('aria-label') || el.getAttribute?.('title') || '';
                if (!text && !name) return '';

                let line = '- ' + role;
                if (name) line += ' "' + name + '"';
                if (text && text !== name) line += ' "' + text.substring(0, 80) + '"';

                const children = [];
                for (const child of el.children || []) {{
                    const childText = getNode(child, depth + 1);
                    if (childText) children.push(childText);
                }}
                return line + (children.length ? '\\n' + children.join('\\n') : '');
            }}
            return getNode(document.body, 0);
        }})()
        """
        result = await self.send_command("Runtime.evaluate", {
            "expression": js,
            "returnByValue": True,
        })
        if result.success and result.data:
            return result.data.get("result", {}).get("value", "")
        return ""

    async def close(self) -> None:
        """Close the browser connection."""
        self._connected = False
        if self._recv_task:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass

        if self._writer:
            try:
                # Send close frame
                await self._ws_send_frame(b"", opcode=0x8)
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass

        self._reader = None
        self._writer = None
        logger.debug("[CDP] Connection closed")


# ── Helper: Auto-discover CDP endpoint ───────────────────────────────────────

async def discover_cdp_url(host: str = "localhost", ports: list[int] | None = None) -> str | None:
    """
    Auto-discover CDP WebSocket URL from running browser.
    Checks Lightpanda and Chrome default ports.
    """
    import httpx

    if ports is None:
        ports = [9222, 9229, 8080]  # Chrome, Chrome alt, Lightpanda

    async with httpx.AsyncClient(timeout=3.0) as client:
        for port in ports:
            try:
                # Try Chrome-style /json/version
                resp = await client.get(f"http://{host}:{port}/json/version")
                if resp.status_code == 200:
                    data = resp.json()
                    ws_url = data.get("webSocketDebuggerUrl", "")
                    if ws_url:
                        logger.info(f"[CDP] Found browser at {host}:{port}")
                        return ws_url
            except Exception:
                pass

            try:
                # Try Lightpanda-style root endpoint
                resp = await client.get(f"http://{host}:{port}/")
                if resp.status_code == 200:
                    logger.info(f"[CDP] Found browser at {host}:{port}")
                    return f"ws://{host}:{port}/devtools/browser"
            except Exception:
                pass

    return None
