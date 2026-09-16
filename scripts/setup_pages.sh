#!/usr/bin/env bash
# One-time GitHub Pages setup.
#
#   bash scripts/setup_pages.sh <repo-name> [--public|--private]
#
# Creates the repo, pushes, enables Pages, and prints the URL.
#
# ─────────────────────────────────────────────────────────────────────────────
#  READ THIS BEFORE CHOOSING --public
# ─────────────────────────────────────────────────────────────────────────────
#  GitHub Pages on a PRIVATE repo requires a paid plan (Pro/Team/Enterprise).
#  On the free tier, Pages only serves PUBLIC repos — which means:
#
#    * your paper positions, equity curve and fills are world-readable
#    * your evolved strategy expressions are world-readable
#    * anyone with the URL can see them; search engines may index them
#
#  Nothing secret is published — .env is gitignored and no API key is ever
#  written into docs/data. The exposure is your positions and your research,
#  not your credentials. For paper trading that is usually fine and sometimes
#  the point. For live trading it is a real information leak: a public,
#  timestamped record of what you hold is something another participant can
#  read. Decide deliberately.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO="${1:-}"
VIS="${2:---private}"

if [[ -z "$REPO" ]]; then
  echo "usage: bash scripts/setup_pages.sh <repo-name> [--public|--private]" >&2
  exit 2
fi

cd "$(dirname "$0")/.."

command -v gh >/dev/null || { echo "gh CLI not found: https://cli.github.com" >&2; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "run: gh auth login" >&2; exit 1; }

if [[ ! -d .git ]]; then
  git init -q -b main
  echo "initialised git repository"
fi

# A live-armed config must never reach a public repo.
if grep -qE '^\s*allow_live:\s*true' config.yaml; then
  echo "REFUSING: config.yaml has trade.allow_live: true." >&2
  echo "Set it back to false before publishing this repository." >&2
  exit 1
fi

git add -A
git diff --cached --quiet || git commit -q -m "loonie: self-improving strategy search + live dashboard

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"

if ! git remote get-url origin >/dev/null 2>&1; then
  echo "creating ${VIS#--} repository: $REPO"
  gh repo create "$REPO" "$VIS" --source=. --remote=origin --push
else
  git push -u origin main
fi

OWNER="$(gh api user --jq .login)"

echo "enabling GitHub Pages (branch main, /docs)…"
gh api -X POST "repos/$OWNER/$REPO/pages" \
  -f "source[branch]=main" -f "source[path]=/docs" >/dev/null 2>&1 \
  || gh api -X PUT "repos/$OWNER/$REPO/pages" \
       -f "source[branch]=main" -f "source[path]=/docs" >/dev/null 2>&1 \
  || echo "  (could not enable automatically — Settings → Pages → main /docs)"

cat <<EOF

────────────────────────────────────────────────────────────────────────
  Dashboard URL (live in ~1 minute):

      https://$OWNER.github.io/$REPO/

  Keep it fed:

      python scripts/run_evolve.py --daemon --push    # search, pushes every 15 min
      python scripts/run_trade.py                     # paper rebalance + publish

  Optional — let GitHub trade when your machine is off:

      gh secret set ALPACA_API_KEY
      gh secret set ALPACA_API_SECRET

  The scheduled workflow (.github/workflows/dashboard.yml) then runs one
  PAPER rebalance each weekday after the close and republishes. It cannot
  trade live: that needs a config edit and a CLI flag this workflow never
  passes.
────────────────────────────────────────────────────────────────────────
EOF
