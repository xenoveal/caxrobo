"""
Transport for the builder UI (v0.3.0 Phase 7): http.server.ThreadingHTTPServer,
the static file allowlist, the SSE endpoint, and loopback enforcement.

Everything request-shaped lives in api.py and is tested there with no socket
bound. This module is the thin, largely-untested-by-design layer that turns
api.handle()'s ApiResponse into bytes on a wire (contract §8).
"""

import http.server
import ipaddress
import json
import logging
import socket
import socketserver
import time
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from trading_bot import config
from trading_bot.ui import api

logger = logging.getLogger("trading_bot")

STATIC_DIR = Path(__file__).resolve().parent / "static"
# A FIXED allowlist, not a directory walk: SimpleHTTPRequestHandler maps URLs
# onto the filesystem and is a traversal surface; three entries cannot be
# traversed. New file -> new entry, deliberately.
STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}

SSE_POLL_SECONDS = 1.0
SSE_HEARTBEAT_SECONDS = 15.0
SSE_MAX_CHUNK_BYTES = 64 * 1024
MAX_SSE_STREAMS = 8  # daemon_threads keeps Ctrl-C working; this guards against
# a reload loop accumulating streams. Enforced loosely: each SSE connection is
# one thread for the run's duration, which is the property that matters here.


def _static_body(name: str) -> bytes:
    """Read a static file from disk PER REQUEST, never cached at import —
    editing app.js and reloading must show the change immediately."""
    return (STATIC_DIR / name).read_bytes()


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # MANDATORY: 1.0 makes SSE buffer to EOF
    server_version = "trading_bot.ui"

    def log_message(self, fmt, *args):  # noqa: A003 - stdlib signature
        logger.info("ui %s", fmt % args)

    # -- helpers ------------------------------------------------------- #

    def _send_json(self, resp: "api.ApiResponse") -> None:
        body = json.dumps(resp.payload).encode("utf-8")
        self.send_response(resp.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in resp.headers.items():
            self.send_header(k, v)
        if resp.status >= 400:
            self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, name: str, content_type: str) -> None:
        try:
            body = _static_body(name)
        except OSError:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.send_header("Connection", "close")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict | None:
        """Read (and discard, if oversize) the request body honouring
        Content-Length, up to api.MAX_BODY_BYTES.

        Deliberately does NOT attempt to parse JSON when Content-Type is not
        application/json: api.handle() itself is the single place a wrong
        Content-Type becomes 415 (Task 7), so the bytes are read off the wire
        to keep the connection in sync but never passed through json.loads —
        otherwise a plain-text body would 400 here before handle() ever gets
        the chance to answer 415.
        """
        length_hdr = self.headers.get("Content-Length")
        length = 0
        if length_hdr:
            try:
                length = int(length_hdr)
            except ValueError:
                length = 0
        if length > api.MAX_BODY_BYTES:
            self.send_response(413)
            self.send_header("Content-Length", "0")
            self.send_header("Connection", "close")
            self.end_headers()
            return None
        raw = self.rfile.read(length) if length else b""
        ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip()
        if ctype != "application/json":
            return {}  # api.handle() rejects on Content-Type regardless of body
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(api._err(400, "body is not valid JSON"))
            return None

    # -- HTTP verbs ------------------------------------------------------ #

    def do_GET(self):  # noqa: N802 - stdlib method name
        parts = urlsplit(self.path)
        if parts.path in STATIC:
            name, ctype = STATIC[parts.path]
            self._send_static(name, ctype)
            return
        if _is_sse_path(parts.path):
            _serve_sse(self, parts.path)
            return
        query = {k: v for k, v in parse_qs(parts.query).items()}
        resp = api.handle("GET", parts.path, query=query, headers=dict(self.headers))
        self._send_json(resp)

    def do_POST(self):  # noqa: N802 - stdlib method name
        parts = urlsplit(self.path)
        body = self._read_body()
        if body is None:  # already responded (bad JSON or oversize body)
            return
        query = {k: v for k, v in parse_qs(parts.query).items()}
        resp = api.handle(
            "POST", parts.path, query=query, body=body, headers=dict(self.headers)
        )
        self._send_json(resp)

    def _do_other(self, method: str) -> None:
        """PUT/DELETE/PATCH etc — routed through api.handle() so an
        unsupported-but-known path answers 405 + Allow (Task 7), rather than
        BaseHTTPRequestHandler's default bare 501 for any do_<METHOD> that was
        never defined."""
        parts = urlsplit(self.path)
        query = {k: v for k, v in parse_qs(parts.query).items()}
        resp = api.handle(method, parts.path, query=query, headers=dict(self.headers))
        self._send_json(resp)

    def do_PUT(self):  # noqa: N802 - stdlib method name
        self._do_other("PUT")

    def do_DELETE(self):  # noqa: N802 - stdlib method name
        self._do_other("DELETE")

    def do_PATCH(self):  # noqa: N802 - stdlib method name
        self._do_other("PATCH")


def _is_sse_path(path: str) -> bool:
    return path.startswith("/api/runs/") and path.endswith("/events")


def _run_id_from_sse_path(path: str) -> str:
    # "/api/runs/<rid>/events" -> "<rid>"
    return path[len("/api/runs/") : -len("/events")]


def _serve_sse(handler: _Handler, path: str) -> None:
    """GET /api/runs/<rid>/events — the only streaming route.

    Headers only, no Content-Length (a streaming body has no length to
    declare); frames come from api.sse_frames so the formatting logic stays
    socket-free and unit-testable.
    """
    run_id = _run_id_from_sse_path(path)
    if not api.RUN_ID_RE.match(run_id):
        handler.send_response(404)
        handler.send_header("Content-Length", "0")
        handler.end_headers()
        return

    query = parse_qs(urlsplit(handler.path).query)
    offset = 0
    last_event_id = handler.headers.get("Last-Event-ID")
    if last_event_id:
        try:
            offset = int(last_event_id)
        except ValueError:
            offset = 0
    elif "offset" in query:
        try:
            offset = int(query["offset"][0])
        except (ValueError, IndexError):
            offset = 0

    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("X-Accel-Buffering", "no")
    handler.end_headers()
    try:
        handler.wfile.write(b"retry: 2000\n\n")
        handler.wfile.flush()
    except (BrokenPipeError, ConnectionResetError):
        return

    run_dir = api._run_dir(run_id)
    log_path = run_dir / "stdout.log"
    last_heartbeat = time.monotonic()
    while True:
        status = api._reconcile_stale_tier_b(
            run_dir, api._read_json(run_dir / "status.json") or {"state": "orphaned"}
        )
        chunk = ""
        if log_path.exists():
            with open(log_path, "rb") as fh:
                fh.seek(offset)
                raw = fh.read(SSE_MAX_CHUNK_BYTES)
            if raw:
                chunk = raw.decode("utf-8", errors="replace")
        status["has_result"] = (run_dir / "result.json").exists()
        frames = api.sse_frames(offset, chunk, status)
        if chunk:
            offset += len(chunk.encode("utf-8"))
        sent_any = False
        try:
            for frame in frames:
                if frame.startswith(": heartbeat"):
                    if time.monotonic() - last_heartbeat < SSE_HEARTBEAT_SECONDS:
                        continue
                    last_heartbeat = time.monotonic()
                handler.wfile.write(frame.encode("utf-8"))
                sent_any = True
            if sent_any:
                handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            # The browser closed mid-run. The run is UNTOUCHED: Tier A's
            # thread and Tier B's detached subprocess hold no reference to
            # this request. Nothing is cancelled by a closed tab.
            logger.debug("ui: SSE client for %s disconnected", run_id)
            return
        if status.get("state") in ("done", "failed", "orphaned", "aborted"):
            return
        time.sleep(SSE_POLL_SECONDS)


class _Server(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True  # an open SSE stream must not block Ctrl-C
    allow_reuse_address = True


def serve(host: str | None = None, port: int | None = None) -> int:
    """Serve the builder UI until interrupted (Ctrl-C).

    Returns:
        0 on clean shutdown, 2 on a refused bind (non-loopback host, or the
        port is already in use).
    """
    host = host or config.UI_HOST
    port = port or config.UI_PORT
    try:
        resolved = socket.gethostbyname(host)
        if not ipaddress.ip_address(resolved).is_loopback:
            logger.error(
                "refusing to bind %s: loopback only. This server has NO auth "
                "and this machine holds the trading logic and the OHLCV "
                "store.", host,
            )
            return 2
    except OSError as exc:
        logger.error("could not resolve host %s: %s", host, exc)
        return 2

    api.set_origin_port(port)
    api.recover_runs()

    try:
        httpd = _Server((host, port), _Handler)
    except OSError as exc:
        if getattr(exc, "errno", None) == 48 or "Address already in use" in str(exc):
            print(f"port {port} in use — another `cli ui` is probably running")
            return 2
        logger.error("could not bind %s:%d: %s", host, port, exc)
        return 2

    print(f"trading_bot.ui listening on http://{host}:{port}  (loopback only, no auth)")
    print(f"  strategies: {config.STRATEGY_DIR}   state: {config.STATE_DB_PATH}   "
          f"runs: {api.RUN_ROOT}")
    try:
        registry_count = _plugin_count()
        print(f"  registry: {registry_count} plug-ins")
    except Exception:  # noqa: BLE001 - a listing failure must not block serving
        pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.shutdown()
        httpd.server_close()
    return 0


def _plugin_count() -> int:
    from trading_bot.framework import registry

    registry.load_all()
    return len(registry.REGISTRY)
