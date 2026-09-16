# robinhood.com — Design Map + Taste DNA

Captured 2026-09-16, 1440×900, `https://robinhood.com/us/en/`.
Measured from the live DOM, not from memory. That distinction earned its keep
here: from memory this site is white with `#00C805` trading green, and it is
neither.

## Design Map

### Color
| role | value | evidence |
|---|---|---|
| page ground | `#000000` | `body` background; 38.2% of sampled surface area |
| signature accent | `#CCFF00` | 27.5% of surface area — the single largest coloured region |
| ink on accent | `#110E08` | 103 text nodes; a warm near-black, not `#000` |
| text on ground | `#FFFFFF` | 31 text nodes |
| panel tint | `#1C180D` | 12.8% area — black warmed toward the accent hue |
| muted text | `#D9D9D9`, `#4D4A46`, `#35322D` | 1–2 nodes each |

Two colours carry the page. There is no third brand colour, no gradient on any
sampled element, and the "panel" is the same black pushed 4% toward the accent
hue rather than a grey.

### Type
| role | value |
|---|---|
| display | **Phonic** — 163 nodes |
| text | **Capsule Sans Text** — 546 nodes |
| serif | Martina Plantijn — 4 nodes (incidental) |
| h1 | 64px / 71px, **weight 400**, tracking −1px |
| h2 | 52px / 62px, weight 400, tracking −1.5px |
| h3 | 40px / 48px, weight 400, tracking −1px |
| body | 16px (117 nodes — the dominant size) |
| weights present | **400 and 700 only** |

Every heading is weight **400**. Hierarchy is carried entirely by size and
negative tracking, never by bolding. Tracking tightens as size grows
(−1px at 64px, −1.5px at 52px).

### Space, shape, depth
- Spacing: **8px** dominates (86 uses), then 24px (30), 12px (9), 32px (6). Every dominant value is a multiple of 8.
- Radius: only two values in the whole page — `36px` (10 uses, pill buttons) and `20px` (2). Everything else is square.
- **Shadows: zero.** Not one `box-shadow` on any of 8,000 sampled elements.
- Buttons: `border-radius: 0` on the sampled inline CTAs; 13px and 16px, weight 400, no letterspacing, no uppercase.

## Taste DNA

### 1. Black is the ground, and the accent does the shouting
**Trigger:** a brokerage page has to feel both serious (your money) and alive
(come trade).
**Decision:** commit the entire page to `#000000` and spend the whole colour
budget on one acid `#CCFF00` covering 27.5% of the surface.
**Reason:** a single saturated colour against true black reads as an
instrument light, not decoration. Splitting that budget across three brand
colours would have produced a dashboard-y look where nothing is emphatic.
**Evidence:** two colours account for 65.7% of sampled surface area; no third
brand colour appears anywhere.

### 2. Never bold the display face (Restraint)
**Trigger:** 64px headlines that need to feel authoritative.
**Decision:** set every heading at weight 400 and tighten tracking instead.
**Reason:** bolding a display face at 64px makes it shout at a reader already
standing close. Size alone establishes the hierarchy; the negative tracking
supplies the density that weight would otherwise provide, without the noise.
**Evidence:** h1/h2/h3/h5 are all `font-weight: 400`; only two weights exist
site-wide (400, 700), and 700 never appears on a heading.

### 3. Refuse depth entirely (Restraint)
**Trigger:** cards, panels and CTAs that conventionally get elevation.
**Decision:** zero box-shadows on the entire page. Separation comes from
colour blocks and a warmed-black panel tint.
**Reason:** shadow is a skeuomorphic cue that implies physical stacking.
Dropping it forces every boundary to be a real edge or a real colour change,
which stays crisp on OLED and at any zoom. It also removes the single most
common tell of a template.
**Evidence:** `boxShadow !== 'none'` matched 0 of ~8,000 rendered elements.

### 4. Round only what you press
**Trigger:** a page full of containers, images, panels and buttons.
**Decision:** exactly two radius values exist, `36px` and `20px`, on 12
elements total. Everything else is square.
**Reason:** reserving roundness for interactive targets turns radius into a
signal rather than a texture. When only pressable things are round, the eye
finds the affordance without a colour cue.
**Evidence:** 12 rounded elements against thousands of square ones; the 36px
value is a full pill at the sampled button height.

## What transfers to a quant instrument

The ground (`#000`), the zero-shadow rule, the 400-weight display and the 8px
grid transfer directly. `#CCFF00` transfers only as a rationed attention
signal — at 27.5% coverage it is a consumer brand statement, and a page about
statistical honesty should not shout in acid green. The hierarchy lesson is
the valuable one: **the number you most want read is set largest and lightest,
at the top, with nothing competing.**
