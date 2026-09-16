# Direction approved

**Date:** 2026-09-16
**Decision by:** user, verbatim:

> "go with A but use B's threshold margin chart and commit
>
> also look into tradingview for some more ui examples but also make it look
> like robinhood where you easily see your p/l position overall right away and
> really try to stick to it.
>
> Also use taste skill on the ui to make this look much better."

## What was shown

Three real rendered drafts, all on live data (gen 1,729 · 156,366 trials ·
15 positions), served at `localhost:8900/design-demos/compare.html`:

| # | logic | file | screenshot |
|---|---|---|---|
| A | style roulette → #19 Swiss Monochrome (Vercel/Geist, Vignelli) | `design-demos/swiss-monochrome.html` | `swiss-monochrome.png` |
| B | real-world benchmark transfer → Financial Times (Origami / Chart Doctor), verified by search | `design-demos/benchmark-transfer.html` | `benchmark-transfer.png` |
| C | unlimited-budget studio → Edward Tufte / Graphics Press lineage | `design-demos/studio-commission.html` | `studio-commission.png` |

## Resolved direction

**A as the shell**, with two grafts and one hierarchy change:

1. **From B:** the threshold-margin chart — every gate expressed as
   `(value − threshold) / |threshold|`, sign-flipped for `<=` gates so right is
   always safer. It is the only transform that puts a t-statistic, a trade
   count and three probabilities on one comparable axis.
2. **From Robinhood:** the information hierarchy. Portfolio value is the hero,
   immediately, above the fold — value, then delta, then chart, then holdings.
   User asked to "really stick to it", so this is not a gesture.
3. **From TradingView:** chart and data-density conventions.

`taste` run against the live sites for concrete tokens rather than memory.

## Tension noted and how it is resolved

Swiss Monochrome is achromatic and instrument-like; Robinhood is chromatic,
consumer and celebratory. The graft takes Robinhood's **hierarchy** (what you
see first, and how big) while keeping A's **palette and geometry**. Signal
colour stays rationed to P&L direction and gate pass/fail, so the page still
reads as an instrument rather than a brokerage app.
