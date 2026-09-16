"""Reporting: HTML files on disk, optional email.

Email is plain SMTP rather than an agent-driven mail tool on purpose -- this
runs unattended at 06:00 with nobody watching, and it must not depend on an
interactive session being alive. Configure it in .env; if it is not configured
the reports still get written to reports/ and nothing fails.
"""
from __future__ import annotations

import os
import smtplib
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path

from .config import resolve

CSS = """
:root{--bg:#0f1115;--card:#171a21;--fg:#e6e8ee;--dim:#9aa3b2;--line:#252a35;
--pos:#3fb950;--neg:#f85149;--warn:#d29922}
*{box-sizing:border-box}
body{margin:0;padding:24px;background:var(--bg);color:var(--fg);
font:14px/1.55 ui-sans-serif,-apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:900px;margin:0 auto}
h1{font-size:20px;margin:0 0 4px} h2{font-size:15px;margin:24px 0 8px;color:var(--dim);
text-transform:uppercase;letter-spacing:.06em}
.sub{color:var(--dim);font-size:13px;margin-bottom:20px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:16px;margin-bottom:12px}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line)}
th{color:var(--dim);font-weight:500;font-size:12px;text-transform:uppercase;
letter-spacing:.04em}
td.num,th.num{text-align:right}
.pos{color:var(--pos)} .neg{color:var(--neg)} .warn{color:var(--warn)}
.pill{display:inline-block;padding:2px 8px;border-radius:99px;font-size:12px;
border:1px solid var(--line)}
.pill.ok{color:var(--pos);border-color:var(--pos)}
.pill.bad{color:var(--neg);border-color:var(--neg)}
code{background:#0b0d11;padding:2px 6px;border-radius:4px;font-size:12.5px;
word-break:break-all}
.foot{color:var(--dim);font-size:12px;margin-top:28px;border-top:1px solid
var(--line);padding-top:12px}
"""


def _fmt(v, pct=False, dp=2):
    if v is None:
        return "-"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f != f:
        return "-"
    s = ("%+.*f%%" % (dp, 100 * f)) if pct else ("%.*f" % (dp, f))
    cls = "pos" if f > 0 else ("neg" if f < 0 else "")
    return '<span class="%s">%s</span>' % (cls, s) if pct else s


def table(rows, cols) -> str:
    if not rows:
        return '<div class="card" style="color:#9aa3b2">nothing to report</div>'
    head = "".join('<th class="%s">%s</th>' % ("num" if c[2] else "", c[0])
                   for c in cols)
    body = ""
    for r in rows:
        tds = ""
        for label, key, num, fmt in cols:
            v = r.get(key) if isinstance(r, dict) else getattr(r, key, None)
            tds += '<td class="%s">%s</td>' % ("num" if num else "", fmt(v))
        body += "<tr>%s</tr>" % tds
    return ('<div class="card"><table><thead><tr>%s</tr></thead>'
            "<tbody>%s</tbody></table></div>" % (head, body))


def render(title: str, sections: list, subtitle: str = "") -> str:
    body = "".join("<h2>%s</h2>%s" % (h, c) for h, c in sections)
    return (
        "<!doctype html><meta charset='utf-8'><title>%s</title><style>%s</style>"
        "<div class='wrap'><h1>%s</h1><div class='sub'>%s</div>%s"
        "<div class='foot'>Generated %s &middot; results are hypothetical unless "
        "marked LIVE &middot; not investment advice</div></div>"
        % (title, CSS, title, subtitle, body,
           datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")))


def write_report(name: str, html: str) -> Path:
    d = resolve("reports")
    d.mkdir(parents=True, exist_ok=True)
    p = d / ("%s_%s.html" % (name, datetime.now().strftime("%Y%m%d_%H%M%S")))
    p.write_text(html, encoding="utf-8")
    latest = d / ("%s_latest.html" % name)
    latest.write_text(html, encoding="utf-8")
    return p


def send_email(subject: str, html: str, to: str | None = None) -> bool:
    host = os.getenv("SMTP_HOST")
    user = os.getenv("SMTP_USER")
    pw = os.getenv("SMTP_PASS")
    to = to or os.getenv("REPORT_TO")
    if not all([host, user, pw, to]):
        return False
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to
    msg.set_content("HTML report attached inline; view in an HTML-capable client.")
    msg.add_alternative(html, subtype="html")
    try:
        with smtplib.SMTP(host, int(os.getenv("SMTP_PORT", "587")), timeout=30) as s:
            s.starttls()
            s.login(user, pw)
            s.send_message(msg)
        return True
    except Exception as e:
        print("[notify] email failed: %s: %s" % (type(e).__name__, e))
        return False


def publish(name: str, title: str, sections: list, subtitle: str = "",
            email: bool = True) -> Path:
    html = render(title, sections, subtitle)
    p = write_report(name, html)
    if email:
        send_email(title, html)
    return p
