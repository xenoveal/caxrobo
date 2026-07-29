"""
Builder UI (v0.3.0 Phase 7): a local, loopback-only, no-auth web page that
makes the Data / Strategy / Feedback lanes visible and operable.

Split deliberately in two, so the request logic can be tested without a
socket (contract §8):

  api.py     Pure request -> JSON over framework/feedback/evolution. Imports
             NOTHING from http or socket. `handle()` is the single entry
             point tests call directly.
  server.py  Transport only: http.server.ThreadingHTTPServer, the static
             file allowlist, the SSE endpoint, and loopback enforcement.

Zero new dependencies (contract §10 row 6): stdlib http.server plus one
vanilla HTML/CSS/JS page under ui/static/.
"""

from trading_bot.ui.api import handle
from trading_bot.ui.server import serve

__all__ = ("handle", "serve")
