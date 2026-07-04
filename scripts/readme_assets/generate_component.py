"""docs/assets/fig_component_path.png — component localization, no circuit claim.

Numbers come from results/component-path/component_path_summary.json
(qwen2_5_1_5b_medium_audit reference, n=260).
"""

from pathlib import Path

from PIL import ImageDraw
from style import (
    AMBER,
    CYAN,
    EMERALD,
    FAINT,
    HAIRLINE,
    HAIRLINE_SOFT,
    MUTED,
    ROSE,
    SLATE,
    TEXT,
    canvas,
    chip,
    glass_panel,
    hex_rgba,
    jb,
    measure_tracked,
    save,
    sg,
    text_tracked,
)

OUT = Path(__file__).resolve().parents[2] / "docs" / "assets" / "fig_component_path.png"

W, H = 3360, 1440
img = canvas(W, H)
d = ImageDraw.Draw(img)

text_tracked(d, (120, 96), "WHERE DOES THE SIGNAL LIVE?", sg(88, "Bold"), TEXT, tracking=2)
chip(img, (W - 120 - 700, 106), "component ablation · n=260", CYAN,
     font=jb(30, "Medium"), pad_x=28, pad_y=16)
d = ImageDraw.Draw(img)
d.text((124, 224), "Project the direction out of each component's output: the residual stream carries it, attention and MLP don't.",
       font=sg(44, "Regular"), fill=MUTED)

# ---- left: grouped bars ------------------------------------------------------
LB = (120, 360, 2240, 1240)
glass_panel(img, LB, radius=30)
d = ImageDraw.Draw(img)
text_tracked(d, (172, 404), "HARMFUL-SCORE Δ BY COMPONENT OUTPUT", jb(34, "Bold"), TEXT, tracking=3)
d.line([172, 470, 2188, 470], fill=HAIRLINE_SOFT, width=2)

COMP_COLORS = {"residual": CYAN, "attention": AMBER, "mlp": SLATE}
layers = [
    ("L16", [("residual", -0.0709), ("attention", -0.0141), ("mlp", -0.0228)]),
    ("L20", [("residual", -0.1008), ("attention", 0.0009), ("mlp", -0.0112)]),
    ("L24", [("residual", -0.0575), ("attention", -0.0023), ("mlp", -0.0056)]),
    ("L27", [("residual", -0.1412), ("attention", -0.0022), ("mlp", 0.0128)]),
]

zero_y = 620
SCALE = 3300
bw, gap, group_gap = 96, 20, 140
x = 360

for t in [0.0, 0.05, 0.10, 0.15]:
    ty = zero_y + t * SCALE
    d.line([320, ty, 2130, ty], fill=HAIRLINE_SOFT, width=2)
    lbl = "0" if t == 0 else f"−{t:.2f}"
    f_t = jb(26, "Regular")
    d.text((300 - d.textlength(lbl, font=f_t), ty - 16), lbl, font=f_t, fill=FAINT)

for name, comps in layers:
    gx0 = x
    for cname, delta in comps:
        c = COMP_COLORS[cname]
        depth = -delta * SCALE
        if depth >= 0:
            box = [x, zero_y, x + bw, zero_y + max(depth, 6)]
        else:
            box = [x, zero_y + depth, x + bw, zero_y]
        d.rounded_rectangle(box, radius=8, fill=hex_rgba(c, 165))
        d.rounded_rectangle(box, radius=8, outline=hex_rgba(c, 230), width=2)
        if cname == "residual":
            f_v = jb(28, "Bold")
            v = f"{delta:.3f}".replace("-", "−")
            d.text((x + bw / 2 - d.textlength(v, font=f_v) / 2, zero_y + depth + 14),
                   v, font=f_v, fill=c)
        x += bw + gap
    gx1 = x - gap
    f_l = jb(32, "Medium")
    label = name + (" · audit layer" if name == "L27" else "")
    col = CYAN if name == "L27" else MUTED
    lw = d.textlength(label, font=f_l)
    lx = min((gx0 + gx1) / 2 - lw / 2, 2110 - lw)
    d.text((lx, 508), label, font=f_l, fill=col)
    x += group_gap

# legend — on the title row, right-aligned
lx = 1420
for cname in ["residual", "attention", "mlp"]:
    c = COMP_COLORS[cname]
    d.rounded_rectangle([lx, 408, lx + 40, 448], radius=10, fill=hex_rgba(c, 165),
                        outline=hex_rgba(c, 230), width=2)
    d.text((lx + 56, 410), cname, font=jb(28, "Medium"), fill=MUTED)
    lx += 56 + int(d.textlength(cname, font=jb(28, "Medium"))) + 60

# ---- right: verdict ----------------------------------------------------------
RB = (2300, 360, 3240, 1240)
glass_panel(img, RB, radius=30)
d = ImageDraw.Draw(img)
text_tracked(d, (2360, 404), "CIRCUIT CLAIM", jb(34, "Bold"), TEXT, tracking=3)
d.line([2360, 470, 3180, 470], fill=HAIRLINE_SOFT, width=2)

big = sg(280, "Bold")
tw = d.textlength("NO", font=big)
d.text(((RB[0] + RB[2]) / 2 - tw / 2, 520), "NO", font=big, fill=ROSE)
sub = "attention & MLP outputs carry\nlittle of the effect — path-level\nmediation was not established"
d.multiline_text(((RB[0] + RB[2]) / 2, 900), sub, font=jb(30, "Regular"), fill=MUTED,
                 anchor="ma", align="center", spacing=16)
chip(img, (2404, 1108), "residual localization: supported", EMERALD,
     font=jb(26, "Medium"), pad_x=24, pad_y=13)

# footer
d = ImageDraw.Draw(img)
d.line([120, H - 150, W - 120, H - 150], fill=HAIRLINE, width=2)
d.text((124, H - 118), "medium-audit reference artifact · residual result reproduced at n=500 (−0.136)",
       font=jb(28, "Regular"), fill=FAINT)
right = "results/component-path/component_path_summary.json"
f_r = jb(28, "Regular")
d.text((W - 124 - d.textlength(right, font=f_r), H - 118), right, font=f_r, fill=FAINT)

save(img, OUT)
