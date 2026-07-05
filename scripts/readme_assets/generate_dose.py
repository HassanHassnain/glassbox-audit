"""docs/assets/fig_dose_response.png — fixed-grid steering dose-response.

Numbers from results/final/dose_response.json (held-out, descriptive).
"""

from pathlib import Path

from PIL import ImageDraw
from style import (
    CYAN,
    FAINT,
    HAIRLINE,
    HAIRLINE_SOFT,
    MUTED,
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

OUT = Path(__file__).resolve().parents[2] / "docs" / "assets" / "fig_dose_response.png"

W, H = 3360, 1400
img = canvas(W, H)
d = ImageDraw.Draw(img)

text_tracked(d, (120, 96), "DOSE-RESPONSE ON THE FIXED GRID", sg(88, "Bold"), TEXT, tracking=2)
chip(img, (W - 120 - 800, 106), "steering · held-out · descriptive", CYAN,
     font=jb(30, "Medium"), pad_x=28, pad_y=16)
d = ImageDraw.Draw(img)
d.text((124, 224), "Monotone, ordered, and an order of magnitude smaller than ablation — scale was never retuned.",
       font=sg(44, "Regular"), fill=MUTED)

PB = (120, 360, 3240, 1220)
glass_panel(img, PB, radius=30)
d = ImageDraw.Draw(img)
text_tracked(d, (172, 404), "HARMFUL-SCORE Δ VS STEERING SCALE", jb(34, "Bold"), TEXT, tracking=3)
d.line([172, 470, 3188, 470], fill=HAIRLINE_SOFT, width=2)

DATA = {  # scale: (sae, mean, probe)
    -4: (-0.00222, -0.00979, -0.00159),
    -2: (-0.00110, -0.00492, -0.00087),
    -1: (-0.00059, -0.00244, -0.00039),
    1: (0.00059, 0.00247, 0.00044),
    2: (0.00114, 0.00498, 0.00093),
    4: (0.00233, 0.00994, 0.00180),
}
METHODS = [("mean-diff", CYAN, 1), ("SAE features", VIOLET, 0), ("probe", SLATE, 2)]

ax0, ax1, ay0, ay1 = 320, 3080, 540, 1120
YMAX = 0.011


def xy(scale, v):
    x = ax0 + (scale + 4) / 8 * (ax1 - ax0)
    y = (ay0 + ay1) / 2 - v / YMAX * (ay1 - ay0) / 2
    return x, y


# axes
for t in [-0.010, -0.005, 0.0, 0.005, 0.010]:
    _, y = xy(0, t)
    d.line([ax0, y, ax1, y], fill=HAIRLINE_SOFT, width=4 if t == 0 else 2)
    lbl = "0" if t == 0 else f"{t:+.3f}".replace("-", "−")
    f_t = jb(26, "Regular")
    d.text((ax0 - 30 - d.textlength(lbl, font=f_t), y - 16), lbl, font=f_t, fill=FAINT)
for s in [-4, -2, -1, 1, 2, 4]:
    x, _ = xy(s, 0)
    d.line([x, ay0, x, ay1], fill=HAIRLINE_SOFT, width=2)
    lbl = f"{s:+d}".replace("-", "−")
    f_t = jb(28, "Regular")
    d.text((x - d.textlength(lbl, font=f_t) / 2, ay1 + 22), lbl, font=f_t, fill=FAINT)
d.text((ax1 - 300, ay1 + 70), "steering scale α", font=jb(28, "Regular"), fill=FAINT)

# lines
scales = sorted(DATA)
for name, c, idx in METHODS:
    pts = [xy(s, DATA[s][idx]) for s in scales]
    for p0, p1 in zip(pts, pts[1:]):
        d.line([p0, p1], fill=hex_rgba(c, 230), width=6)
    for px, py in pts:
        d.ellipse([px - 12, py - 12, px + 12, py + 12], fill=hex_rgba(c, 255),
                  outline=hex_rgba("#0B101E", 255), width=4)

# legend on title row
lx = 2280
for name, c, _ in METHODS:
    d.line([lx, 428, lx + 56, 428], fill=hex_rgba(c, 255), width=8)
    d.text((lx + 72, 410), name, font=jb(28, "Medium"), fill=MUTED)
    lx += 72 + int(d.textlength(name, font=jb(28, "Medium"))) + 56

d.line([120, H - 150, W - 120, H - 150], fill=HAIRLINE, width=2)
d.text((124, H - 118), "α ∈ {±1, ±2, ±4}, frozen on validation · slopes: mean-diff ≈ 4× SAE ≈ 6× probe · ablation (fig. 1) ≈ 14× the strongest point here",
       font=jb(28, "Regular"), fill=FAINT)
right = "results/final/dose_response.json"
f_r = jb(28, "Regular")
d.text((W - 124 - d.textlength(right, font=f_r), H - 118), right, font=f_r, fill=FAINT)

save(img, OUT)
