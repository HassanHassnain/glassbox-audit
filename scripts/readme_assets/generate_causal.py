"""docs/assets/fig_causal_effects.png — held-out + external causal effects.

Numbers come from:
  results/final/qwen2_5_1_5b_audit.json
  results/final/statistical_tests.json
  results/external-causal/or_bench_qwen15b_1000_summary.json
"""

from pathlib import Path

from PIL import ImageDraw
from style import (
    CYAN,
    FAINT,
    GRAY,
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

OUT = Path(__file__).resolve().parents[2] / "docs" / "assets" / "fig_causal_effects.png"

W, H = 3360, 1560
img = canvas(W, H)
d = ImageDraw.Draw(img)

text_tracked(d, (120, 96), "HELD-OUT CAUSAL EFFECTS", sg(88, "Bold"), TEXT, tracking=2)
chip(img, (W - 120 - 1096, 106), "harmful-score Δ · more negative = stronger", CYAN,
     font=jb(30, "Medium"), pad_x=28, pad_y=16)
d = ImageDraw.Draw(img)
d.text((124, 224), "The simple mean-difference direction out-suppresses SAE features — internally and on OR-Bench.",
       font=sg(44, "Regular"), fill=MUTED)

MAX = 0.18  # common magnitude scale across both panels

PANELS = [
    {
        "box": (120, 360, 1660, 1300),
        "title": "INTERNAL · HELD-OUT TEST",
        "tag": "n=500 · layer 27",
        "rows": [
            ("mean-diff direction", CYAN, 0.1355, (0.1433, 0.1277), "−0.136", None),
            ("SAE features", VIOLET, 0.0363, (0.0388, 0.0341), "−0.036", None),
            ("linear probe", SLATE, 0.0025, (0.0031, 0.0017), "−0.002", None),
            ("matched random-SAE", GRAY, 0.0009, None, "≈ 0", None),
        ],
        "note": "controls straddle zero — the SAE effect is real, just small",
    },
    {
        "box": (1700, 360, 3240, 1300),
        "title": "EXTERNAL · OR-BENCH",
        "tag": "frozen artifacts · no retuning",
        "rows": [
            ("mean-diff direction", CYAN, 0.1629, (0.1754, 0.1511), "−0.163", "refusal-rate Δ −0.68"),
            ("SAE features", VIOLET, 0.0462, (0.0534, 0.0396), "−0.046", "refusal-rate Δ −0.12"),
            ("linear probe", SLATE, 0.0036, (0.0042, 0.0031), "−0.004", "refusal-rate Δ  0.00"),
        ],
        "note": "same directions, new distribution — the ordering survives transfer",
    },
]

for p in PANELS:
    x0, y0, x1, y1 = p["box"]
    glass_panel(img, p["box"], radius=30)
    d = ImageDraw.Draw(img)
    text_tracked(d, (x0 + 52, y0 + 44), p["title"], jb(36, "Bold"), TEXT, tracking=3)
    tag_f = jb(28, "Regular")
    d.text((x1 - 52 - d.textlength(p["tag"], font=tag_f), y0 + 50), p["tag"], font=tag_f, fill=FAINT)
    d.line([x0 + 52, y0 + 116, x1 - 52, y0 + 116], fill=HAIRLINE_SOFT, width=2)

    # plot area
    ax0, ax1 = x0 + 76, x1 - 420   # bar span
    base_y = y0 + 170
    row_h = 168 if len(p["rows"]) == 4 else 220
    bar_h = 52

    # gridlines + ticks
    for t in [0.0, 0.05, 0.10, 0.15]:
        gx = ax0 + (ax1 - ax0) * (t / MAX)
        d.line([gx, base_y - 20, gx, y1 - 130], fill=HAIRLINE_SOFT, width=2)
        lbl = "0" if t == 0 else f"−{t:.2f}"
        f_t = jb(26, "Regular")
        d.text((gx - d.textlength(lbl, font=f_t) / 2, y1 - 118), lbl, font=f_t, fill=FAINT)

    for i, (label, c, mag, ci, val, sub) in enumerate(p["rows"]):
        yy = base_y + i * row_h
        d.text((ax0, yy - 44), label, font=jb(30, "Medium"), fill=MUTED)
        bl = (ax1 - ax0) * (mag / MAX)
        # bar
        d.rounded_rectangle([ax0, yy, ax0 + max(bl, 10), yy + bar_h], radius=10,
                            fill=hex_rgba(c, 165))
        d.rounded_rectangle([ax0, yy, ax0 + max(bl, 10), yy + bar_h], radius=10,
                            outline=hex_rgba(c, 230), width=2)
        # CI whisker
        if ci:
            hi = (ax1 - ax0) * (ci[0] / MAX)
            lo = (ax1 - ax0) * (ci[1] / MAX)
            cym = yy + bar_h // 2
            d.line([ax0 + lo, cym, ax0 + hi, cym], fill=hex_rgba("#FFFFFF", 210), width=4)
            for wx in (lo, hi):
                d.line([ax0 + wx, cym - 12, ax0 + wx, cym + 12], fill=hex_rgba("#FFFFFF", 210), width=4)
        # value
        f_v = jb(44, "Bold")
        d.text((ax1 + 56, yy - 4), val, font=f_v, fill=c)
        if sub:
            d.text((ax1 + 56, yy + 54), sub, font=jb(26, "Regular"), fill=FAINT)

    d.text((x0 + 52, y1 - 64), p["note"], font=jb(28, "Regular"), fill=FAINT)

# footer
d.line([120, H - 150, W - 120, H - 150], fill=HAIRLINE, width=2)
d.text((124, H - 118),
       "whiskers: bootstrap 95% CI  ·  all effects survive BH correction — the contest is effect size, not significance",
       font=jb(28, "Regular"), fill=FAINT)
right = "results/final · results/external-causal"
f_r = jb(28, "Regular")
d.text((W - 124 - d.textlength(right, font=f_r), H - 118), right, font=f_r, fill=FAINT)

save(img, OUT)
