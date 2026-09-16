"""Serve the dashboard locally, with an optional public tunnel.

    python scripts/serve.py                 # localhost + LAN
    python scripts/serve.py --tunnel        # + a public https URL
    python scripts/serve.py --preview       # also expose design-demos/ (LAN only)

Serves `docs/` and nothing else, deliberately. The project root holds `.env`,
the broker keys and 75 MB of cache; none of that should be one path traversal
away from a public tunnel. `--preview` widens the root to the project for
design review and is refused outright when `--tunnel` is on.

Data files are sent with no-cache headers. A dashboard that polls every 30s
and gets a 304 from its own browser cache is a dashboard that quietly stops
being live, which is the one failure this page must not have.
"""
from __future__ import annotations

import argparse
import functools
import http.server
import shutil
import socket
import socketserver
import subprocess
import sys
import threading
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loonie.config import ROOT  # noqa: E402


class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # JSON is state, not an asset. Never let a cache answer for it.
        if self.path.endswith(".json"):
            self.send_header("Cache-Control", "no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def log_message(self, fmt, *args):
        if "404" in (fmt % args):
            sys.stderr.write("[serve] 404 %s\n" % self.path)


class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def free_port(start: int = 8900, end: int = 8999) -> int:
    for p in range(start, end):
        with socket.socket() as s:
            s.settimeout(0.2)
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    raise RuntimeError("no free port in %d-%d" % (start, end))


def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def start_tunnel(port: int):
    """Cloudflare quick tunnel: a public https URL, no account, no open port.

    This is how the dashboard reaches the internet without a public GitHub
    repo. The URL is random and changes on every restart -- fine for checking
    from a phone, not something to hand out.
    """
    exe = shutil.which("cloudflared")
    if not exe:
        print("[tunnel] cloudflared not found. Install with:")
        print("           winget install --id Cloudflare.cloudflared")
        print("         then re-run with --tunnel")
        return None

    proc = subprocess.Popen(
        [exe, "tunnel", "--url", "http://localhost:%d" % port],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace", bufsize=1)

    def pump():
        for line in proc.stdout:
            if "trycloudflare.com" in line:
                for tok in line.split():
                    if tok.startswith("https://") and "trycloudflare" in tok:
                        print("\n  PUBLIC URL  %s" % tok.strip())
                        print("  (random, changes on restart; anyone with it can view)\n")
                        break
    threading.Thread(target=pump, daemon=True).start()
    return proc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--tunnel", action="store_true",
                    help="expose a public https URL via cloudflared")
    ap.add_argument("--preview", action="store_true",
                    help="serve the project root so design-demos/ is reachable")
    ap.add_argument("--no-open", action="store_true")
    a = ap.parse_args()

    if a.preview and a.tunnel:
        print("REFUSING: --preview serves the project root, which contains .env\n"
              "          and your broker keys. It cannot be combined with --tunnel.")
        return 2

    root = ROOT if a.preview else ROOT / "docs"
    if not a.preview and not (root / "index.html").exists():
        print("[serve] docs/index.html does not exist yet.")
        print("        Run with --preview to browse design-demos/ instead.")

    port = a.port or free_port()
    handler = functools.partial(Handler, directory=str(root))

    with Server(("0.0.0.0", port), handler) as httpd:
        url = "http://localhost:%d/" % port
        print("=" * 66)
        print("  loonie dashboard")
        print("=" * 66)
        print("  serving   %s" % root)
        print("  local     %s" % url)
        print("  this LAN  http://%s:%d/   (phone, same wifi)" % (lan_ip(), port))
        if a.preview:
            print("  compare   %sdesign-demos/compare.html" % url)
        tunnel = start_tunnel(port) if a.tunnel else None
        print("=" * 66)
        print("  Ctrl-C to stop\n")

        if not a.no_open:
            target = (url + "design-demos/compare.html") if a.preview else url
            threading.Timer(1.0, lambda: webbrowser.open(target)).start()

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n[serve] stopped")
        finally:
            if tunnel:
                tunnel.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
