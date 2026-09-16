# tradingview.com — Design Map + Taste DNA

Captured 2026-09-16, 1440×900, `https://www.tradingview.com/chart/` (the live
chart application, not the marketing site).

## Design Map

### Color
| role | value | evidence |
|---|---|---|
| app ground | `#EBEBEB` | `body` background |
| surface | `#FFFFFF` | 93.6% of sampled surface area |
| ink | `#0F0F0F` | 171 text nodes |
| muted ink | `#707070`, `#B8B8B8` | 20 / 10 nodes |
| **up** | `#22AB94` (text), `#089981` (candle) | 27 / 11 nodes |
| **down** | `#F23645` | 19 nodes |
| interactive | `#2962FF` / `#448AFF` | borders + fills |
| hairline | `#F2F2F2` at **0.667px** | 21 uses |

The up colour is a **teal**-green, not a pure green, and the down red is
pushed orange. Both stay distinguishable in the most common form of colour
blindness, where a pure red/green pair collapses.

### Type
| role | value |
|---|---|
| family | `-apple-system` only — **no webfont anywhere in the app** |
| dominant sizes | 14px (99), 13px (70), 11px (64), 12px (43) |
| large | one 28px node, one 20px node |
| weights | 400 (210), 600 (64), 700 (21) |

The entire application lives between **11px and 14px**. Exactly one element on
screen is 28px. There is no type scale in the editorial sense — there is a
working size and a headline size.

### Space, shape, depth
- Spacing: **3px** (91), **4px** (55), 8px (32), 5px (23), 2px (22). Roughly half the step of a marketing page.
- Radius: `4px` is the workhorse (21 uses); `50%` for avatars (54).
- Borders: `0.666667px` — deliberately sub-pixel hairlines, 21 uses.

## Taste DNA

### 1. One big number, everything else at 11–14px
**Trigger:** a screen that must show a price, a chart, a watchlist, a toolbar
and an order panel at once.
**Decision:** cap the working type scale at 14px and permit exactly one 28px
element.
**Reason:** at 11–14px a 1440px viewport fits roughly four times the
information of a 16px baseline, and a trader reads this screen for hours at a
fixed distance. The single large number then needs no other emphasis — it is
the only thing on screen that is big.
**Evidence:** 276 sampled text nodes at 11–14px against 1 node at 28px.

### 2. Hairlines instead of cards (Restraint)
**Trigger:** a dozen functional regions that need separating.
**Decision:** divide with `0.667px` borders and refuse container cards.
**Reason:** a card costs padding on four sides plus a radius plus usually a
shadow — call it 32px of chrome per region. At a dozen regions that is a
third of the viewport spent on boundaries. A sub-pixel rule separates just as
clearly for free.
**Evidence:** 21 hairline borders; the only radius in general use is 4px, and
shadows are near-absent from the chart surface.

### 3. Teal-green, not green (Restraint)
**Trigger:** the up/down semantic every financial interface needs.
**Decision:** `#22AB94` against `#F23645` rather than pure `#00C805`/`#FF0000`.
**Reason:** red/green is the exact pair that deuteranopia collapses, and it is
the single most consequential colour decision in a trading interface. Shifting
the green toward teal separates the pair by hue *and* lightness, so direction
survives even when hue does not.
**Evidence:** both greens sampled (`#22AB94`, `#089981`) are teal-shifted; the
red is orange-shifted.

### 4. No webfont in the working application
**Trigger:** a product with a strong marketing-site typeface.
**Decision:** the chart app is `-apple-system` throughout — 1,552 nodes, one
family.
**Reason:** a webfont costs a network round trip and a reflow on a screen
where numbers change several times a second, and at 11px a system UI face is
already hinted for the platform. Brand personality is spent on the marketing
site; the tool is spent on legibility.
**Evidence:** exactly one font family across the entire sampled application.

## What transfers to a quant instrument

The 11–14px working scale, the sub-pixel hairline as the only divider, and the
teal/red pair all transfer directly. The single-28px-element rule is the
structural lesson: **decide which one number is the headline, then make
nothing else compete with it.** Combined with Robinhood's hierarchy, that
means portfolio value is the one large number and every other figure on the
page — gates, trials, positions — sits at working size.
