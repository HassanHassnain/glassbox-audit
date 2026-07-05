"""docs/assets/fig_stability_grid.png — six SAE retrainings, zero rank flips.

Numbers come from results/sae-stability/stability_grid.json.
"""

from pathlib import Path

from PIL import ImageDraw
from style import (
    CYAN,
    EMERALD,
    FAINT,
    HAIRLINE,
    HAIRLINE_SOFT,
    MUTED,
    ROSE,
    SLATE,
    TEXT,
    VIOLET,
    canvas,
    chip,
    glass_panel,
    hex_rgba,
    jb,
    save,
    sg,
    text_tracked,
)

OUT = Path(__file__).resolve().parents[2] / "docs" / "assets" / "fig_stability_grid.png"

W, H = 3360, 1440
img = canvas(W, H)
d = ImageDraw.Draw(img)

text_tracked(d, (120, 96), "IS THE NEGATIVE RESULT A FLUKE?", sg(88, "Bold"), TEXT, tracking=2)
chip(img, (W - 120 - 646, 106), "3 seeds × 2 widths · fixed", VIOLET,
     font=jb(30, "Medium"), pad_x=28, pad_y=16)
d = ImageDraw.Draw(img)
d.text((124, 224), "Six independent SAE retrainings on the same audit. The ranking never flips.",
       font=sg(44, "Regular"), fill=MUTED)

# ---- left: bar chart ---------------------------------------------------------
LB = (120, 360, 2240, 1240)
glass_panel(img, LB, radius=30)
d = ImageDraw.Draw(img)
text_tracked(d, (172, 404), "SAE FEATURE ABLATION · HARMFUL-SCORE Δ PER CELL",
             jb(34, "Bold"), TEXT, tracking=3)
d.line([172, 470, 2188, 470], fill=HAIRLINE_SOFT, width=2)

cells = [  # (label, delta) from stability_grid.json
    ("seed 17", -0.0354), ("seed 23", -0.0613), ("seed 42", -0.0450),
    ("seed 17", -0.0161), ("seed 23", -0.0332), ("seed 42", -0.0313),
]
MEAN_REF = -0.1412
PROBE_REF = -0.0091

zero_y = 560
SCALE = 3300  # px per unit delta
bw, gap, group_gap = 150, 70, 190
x = 470

# y ticks
for t in [0.0, 0.05, 0.10, 0.15]:
    ty = zero_y + t * SCALE
    d.line([320, ty, 2130, ty], fill=HAIRLINE_SOFT, width=2)
    lbl = "0" if t == 0 else f"−{t:.2f}"
    f_t = jb(26, "Regular")
    d.text((300 - d.textlength(lbl, font=f_t), ty - 16), lbl, font=f_t, fill=FAINT)

# reference lines (under the bars)
for ref, c in [(PROBE_REF, SLATE), (MEAN_REF, CYAN)]:
    ry = zero_y + -ref * SCALE
    xx = 320
    while xx < 2130:
        d.line([xx, ry, xx + 26, ry], fill=hex_rgba(c, 220), width=5)
        xx += 44
f_ref = jb(30, "Medium")
lbl = "probe −0.009"
d.text((2130 - d.textlength(lbl, font=f_ref), zero_y + -PROBE_REF * SCALE + 18), lbl, font=f_ref, fill=SLATE)
d.text((320, zero_y + -MEAN_REF * SCALE + 16), "mean-diff direction  −0.141", font=f_ref, fill=CYAN)

bar_x = []
for i, (label, delta) in enumerate(cells):
    depth = -delta * SCALE
    d.rounded_rectangle([x, zero_y, x + bw, zero_y + depth], radius=10, fill=hex_rgba(VIOLET, 165))
    d.rounded_rectangle([x, zero_y, x + bw, zero_y + depth], radius=10, outline=hex_rgba(VIOLET, 230), width=2)
    f_v = jb(28, "Bold")
    v = f"{delta:.3f}".replace("-", "−")
    d.text((x + bw / 2 - d.textlength(v, font=f_v) / 2, zero_y + depth + 14), v, font=f_v, fill=VIOLET)
    f_l = jb(28, "Regular")
    d.text((x + bw / 2 - d.textlength(label, font=f_l) / 2, zero_y - 48), label, font=f_l, fill=MUTED)
    bar_x.append(x)
    x += bw + (group_gap if i == 2 else gap)

# group labels
f_g = jb(30, "Medium")
for (i0, i1, name) in [(0, 2, "expansion ×2"), (3, 5, "expansion ×4")]:
    cx = (bar_x[i0] + bar_x[i1] + bw) / 2
    d.text((cx - d.textlength(name, font=f_g) / 2, zero_y - 110), name, font=f_g, fill=TEXT)

d.text((172, 1160), "every cell beats its matched random-SAE control — and loses to the mean-diff line",
       font=jb(28, "Regular"), fill=FAINT)

# ---- right: verdict panel ----------------------------------------------------
RB = (2300, 360, 3240, 1240)
glass_panel(img, RB, radius=30)
d = ImageDraw.Draw(img)
text_tracked(d, (2360, 404), "SAE SUPERIORITY", jb(34, "Bold"), TEXT, tracking=3)
d.line([2360, 470, 3180, 470], fill=HAIRLINE_SOFT, width=2)

big = sg(300, "Bold")
tw = d.textlength("0/6", font=big)
d.text(((RB[0] + RB[2]) / 2 - tw / 2, 520), "0/6", font=big, fill=ROSE)
sub = "cells where SAE features beat\nboth mean-diff and probe\nbaselines on held-out records"
d.multiline_text(((RB[0] + RB[2]) / 2, 940), sub, font=jb(30, "Regular"), fill=MUTED,
                 anchor="ma", align="center", spacing=16)

chip(img, (2422, 1120), "6/6 cells completed · none excluded", EMERALD,
     font=jb(26, "Medium"), pad_x=24, pad_y=13)

# footer
d = ImageDraw.Draw(img)
d.line([120, H - 150, W - 120, H - 150], fill=HAIRLINE, width=2)
d.text((124, H - 118), "fixed grid: seeds {17, 23, 42} × dictionary widths {×2, ×4} · mean/probe lines shared per artifact",
       font=jb(28, "Regular"), fill=FAINT)
right = "results/sae-stability/stability_grid.json"
f_r = jb(28, "Regular")
d.text((W - 124 - d.textlength(right, font=f_r), H - 118), right, font=f_r, fill=FAINT)

save(img, OUT)
