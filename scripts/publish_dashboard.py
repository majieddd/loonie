"""Write the dashboard data, and optionally push it to GitHub Pages.

    python scripts/publish_dashboard.py              # write docs/data/ only
    python scripts/publish_dashboard.py --push       # write, commit, push
    python scripts/publish_dashboard.py --push --throttle 900

Why the throttle exists: the search finishes a generation roughly every twenty
seconds. Committing each one would be ~4,000 commits a day and a repository
that is mostly noise. So the data files are rewritten every time (the local
dashboard is always current) but the push happens at most once per
`--throttle` seconds. Fifteen minutes is a sensible default -- it is far below
the granularity at which any of these numbers actually mean something.

Nothing here force-pushes and nothing rewrites history. If the remote has
moved, it rebases; if that fails it aborts the rebase before reporting, so a
failed publish cannot leave the repository mid-rebase with every later git
command refusing to run.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# The Windows console defaults to cp1252, which cannot encode the arrows and
# dashes used below; without this every run dies on a UnicodeEncodeError in a
# print statement rather than in anything that matters.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


from loonie import config, publish  # noqa: E402

STAMP = "state/.last_push"


def git(*args, check=False, cwd=None):
    r = subprocess.run(["git", *args], cwd=cwd or str(config.ROOT),
                       capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError("git %s failed: %s" % (" ".join(args), r.stderr.strip()))
    return r


def have_repo() -> bool:
    return git("rev-parse", "--is-inside-work-tree").returncode == 0


def have_remote() -> bool:
    return bool(git("remote").stdout.strip())


def throttled(seconds: float) -> bool:
    p = config.resolve(STAMP)
    if not p.exists():
        return False
    try:
        return (time.time() - float(p.read_text().strip())) < seconds
    except Exception:
        return False


def mark_push():
    p = config.resolve(STAMP)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(str(time.time()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true",
                    help="commit and push docs/data to the git remote")
    ap.add_argument("--throttle", type=float, default=900,
                    help="minimum seconds between pushes (default 900)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    config.load_env()
    cfg = config.load()
    r = publish.publish(cfg)

    snap = r["doc"]
    if not a.quiet:
        print("[publish] gen %s | %s trials | %d positions | $%s equity | %.1f KB"
              % (snap["search"]["generation"], snap["search"]["trials"],
                 snap["portfolio"]["n_positions"],
                 format(snap["portfolio"]["equity"], ",.2f"), r["bytes"] / 1024))

    if not a.push:
        return 0
    if not have_repo():
        print("[publish] not a git repository -- run scripts/setup_pages.sh first")
        return 1
    if not have_remote():
        print("[publish] no git remote configured; data written but not pushed")
        return 0
    if throttled(a.throttle):
        if not a.quiet:
            print("[publish] throttled (last push < %.0fs ago)" % a.throttle)
        return 0

    git("add", "docs/data")
    if not git("diff", "--cached", "--quiet").returncode:
        if not a.quiet:
            print("[publish] no data changes to commit")
        return 0

    msg = ("data: gen %s, %s trials, %d positions, $%s"
           % (snap["search"]["generation"], snap["search"]["trials"],
              snap["portfolio"]["n_positions"],
              format(snap["portfolio"]["equity"], ",.0f")))
    c = git("commit", "-m", msg)
    if c.returncode != 0:
        print("[publish] commit failed: %s" % c.stderr.strip()[:200])
        return 1

    pull = git("pull", "--rebase", "--autostash")
    if pull.returncode != 0:
        # A failed rebase does NOT leave the tree alone: it leaves
        # .git/rebase-merge behind, and every subsequent git command in the
        # repository refuses to run until someone clears it by hand. That is
        # how this job wedged the repo after a history rewrite moved the
        # remote out from under an in-flight rebase -- the daemon then failed
        # every publish for hours with a message about `git rebase --continue`.
        #
        # Aborting restores the pre-pull state including the autostash, which
        # is the outcome the docstring always claimed and never delivered.
        ab = git("rebase", "--abort")
        print("[publish] rebase onto remote failed; not pushing.\n%s"
              % pull.stderr.strip()[:300])
        if ab.returncode == 0:
            print("[publish] rebase aborted; working tree restored")
        else:
            print("[publish] WARNING: could not abort the rebase. The repo is "
                  "mid-rebase and later git operations will refuse to run "
                  "until it is cleared:\n  git -C %s rebase --abort"
                  % config.ROOT)
        return 1

    push = git("push")
    if push.returncode != 0:
        print("[publish] push failed: %s" % push.stderr.strip()[:300])
        return 1

    mark_push()
    if not a.quiet:
        print("[publish] pushed: %s" % msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
