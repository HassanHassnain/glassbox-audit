"""docs/assets/fig_claim_ladder.png — the claim ladder from docs/release-report.md."""

from pathlib import Path

from PIL import ImageDraw
from style import (
    AMBER,
    EMERALD,
    FAINT,
    HAIRLINE,
    MUTED,
    ROSE,
    TEXT,
    canvas,
    chip,
    draw_check,
    draw_cross,
    glass_panel,
    hex_rgba,
    jb,
    save,
    sg,
    text_tracked,
)

OUT = Path(__file__).resolve().parents[2] / "docs" / "assets" / "fig_claim_ladder.png"

W, H = 3360, 1720
img = canvas(W, H)
d = ImageDraw.Draw(img)

text_tracked(d, (120, 96), "THE CLAIM LADDER", sg(88, "Bold"), TEXT, tracking=2)
chip(img, (W - 120 - 810, 106), "each rung earned separately", AMBER,
     font=jb(30, "Medium"), pad_x=28, pad_y=16)
d = ImageDraw.Draw(img)
d.text((124, 224), "Glassbox climbs until the evidence stops — and says so. Higher rungs were tested and rejected.",
       font=sg(44, "Regular"), fill=MUTED)

PB = (120, 360, 3240, 1560)
glass_panel(img, PB, radius=30)

rungs = [  # bottom-up ladder, drawn top-down as strongest-claim-first
    ("Circuit-level mechanism", "attention/MLP mediation small or non-specific", "FAILED", ROSE),
    ("SAE beats mean-diff + probe baselines", "0/6 stability cells pass preregistered H3", "FAILED", ROSE),
    ("Cross-model replication", "Qwen2.5-3B late-layer effect, specificity failed", "PARTIAL", AMBER),
    ("External causal transfer", "OR-Bench refusal-rate Δ −0.68 via mean direction", "SUPPORTED", EMERALD),
    ("Held-out causal residual direction", "layer 27, harmful-score Δ −0.136 [95% CI ±0.008]", "SUPPORTED", EMERALD),
    ("Harmful/benign activation separation", "late-layer linear separation, balanced acc 0.746", "SUPPORTED", EMERALD),
]

x_rail = 320
row_h = 176
y = 460

for i, (claim, detail, status, c) in enumerate(rungs):
    cy = y + i * row_h + row_h // 2

    # rail + node
    if i < len(rungs) - 1:
        d.line([x_rail, cy, x_rail, cy + row_h], fill=hex_rgba(c if False else "#2A3A5E", 255), width=6)
    ink = "#0B101E"
    d.ellipse([x_rail - 34, cy - 34, x_rail + 34, cy + 34], fill=c, outline=hex_rgba(c, 220), width=4)
    if status == "SUPPORTED":
        draw_check(d, x_rail, cy, 15, ink, width=8)
    elif status == "FAILED":
        draw_cross(d, x_rail, cy - 1, 13, ink, width=8)
    else:
        d.line([x_rail - 14, cy, x_rail + 14, cy], fill=ink, width=8)

    d.text((x_rail + 96, cy - 52), claim, font=sg(52, "Medium"), fill=TEXT)
    d.text((x_rail + 98, cy + 14), detail, font=jb(30, "Regular"), fill=FAINT)

    chip(img, (2760, cy - 34), status, c, font=jb(30, "Bold"), pad_x=30, pad_y=15, dot=False)
    d = ImageDraw.Draw(img)
    if i < len(rungs) - 1:
        d.line([460, cy + row_h // 2 + 12, 3140, cy + row_h // 2 + 12], fill="#1A2540", width=2)

# footer
d.line([120, H - 150, W - 120, H - 150], fill=HAIRLINE, width=2)
d.text((124, H - 118), "a rung is claimed only if it survives held-out, control-matched, preregistered tests",
       font=jb(28, "Regular"), fill=FAINT)
right = "docs/release-report.md · results/final/claim_summary.json"
f_r = jb(28, "Regular")
d.text((W - 124 - d.textlength(right, font=f_r), H - 118), right, font=f_r, fill=FAINT)

save(img, OUT)
