"""docs/assets/pipeline.png — how the audit works (split discipline diagram)."""

from pathlib import Path

from PIL import ImageDraw
from style import (
    AMBER,
    CYAN,
    EMERALD,
    FAINT,
    HAIRLINE,
    MUTED,
    ROSE,
    SLATE,
    TEXT,
    VIOLET,
    canvas,
    chip,
    draw_check,
    draw_cross,
    glass_panel,
    hex_rgba,
    jb,
    measure_tracked,
    save,
    sg,
    text_tracked,
    tinted_box,
)

OUT = Path(__file__).resolve().parents[2] / "docs" / "assets" / "pipeline.png"

W, H = 3360, 1520
img = canvas(W, H)
d = ImageDraw.Draw(img)

# header ---------------------------------------------------------------------
text_tracked(d, (120, 96), "HOW THE AUDIT WORKS", sg(88, "Bold"), TEXT, tracking=2)
chip(img, (W - 120 - 1010, 106), "preregistered · no test-set tuning", AMBER,
     font=jb(30, "Medium"), pad_x=28, pad_y=16)
d = ImageDraw.Draw(img)
d.text((124, 224), "One pass, two walls: discovery, selection, and judgement never touch the same records.",
       font=sg(44, "Regular"), fill=MUTED)

# stage panels ----------------------------------------------------------------
PX = [120, 940, 1760, 2580]
PW, PY0, PY1 = 660, 380, 1220
STEP_COLORS = [SLATE, VIOLET, AMBER, CYAN]
TITLES = ["DATA", "DISCOVER", "SELECT", "JUDGE"]
SUBS = ["1,000 matched pairs", "train split only", "validation split only", "held-out test split"]

for i, x in enumerate(PX):
    glass_panel(img, (x, PY0, x + PW, PY1), radius=30)
    c = STEP_COLORS[i]
    tinted_box(img, (x + 44, PY0 + 44, x + 44 + 96, PY0 + 44 + 96), c, radius=20, fill_alpha=30)
    d = ImageDraw.Draw(img)
    num = f"0{i + 1}"
    f_num = jb(44, "Bold")
    tw = d.textlength(num, font=f_num)
    d.text((x + 44 + 48 - tw / 2, PY0 + 44 + 22), num, font=f_num, fill=c)
    text_tracked(d, (x + 172, PY0 + 48), TITLES[i], sg(58, "Bold"), TEXT, tracking=2)
    d.text((x + 174, PY0 + 122), SUBS[i], font=jb(30, "Regular"), fill=c)

# arrows between panels
d = ImageDraw.Draw(img)
for i in range(3):
    x0 = PX[i] + PW + 8
    x1 = PX[i + 1] - 8
    y = (PY0 + PY1) // 2
    d.line([x0 + 14, y, x1 - 30, y], fill=hex_rgba(CYAN, 200), width=6)
    d.polygon([(x1 - 6, y), (x1 - 36, y - 16), (x1 - 36, y + 16)], fill=hex_rgba(CYAN, 220))

# firewall walls between stages 2|3 and 3|4
for xw in [(PX[1] + PW + PX[2]) // 2, (PX[2] + PW + PX[3]) // 2]:
    yy = PY0 - 30
    while yy < PY1 + 40:
        d.line([xw, yy, xw, yy + 26], fill=hex_rgba(ROSE, 120), width=4)
        yy += 44
    lbl = "WALL"
    f_w = jb(24, "Bold")
    lw = measure_tracked(d, lbl, f_w, 6)
    d.rounded_rectangle([xw - lw / 2 - 18, PY1 + 52, xw + lw / 2 + 18, PY1 + 116],
                        radius=14, fill=hex_rgba("#0B101E", 255), outline=hex_rgba(ROSE, 140), width=2)
    text_tracked(d, (int(xw - lw / 2), PY1 + 66), lbl, f_w, hex_rgba(ROSE, 220), tracking=6)

# ---- panel 1 body: harmful/benign pair
x = PX[0]
by = PY0 + 220
pair = [("harmful prompt", ROSE), ("benign twin", EMERALD)]
d.line([x + 96, by + 104, x + 96, by + 156], fill=hex_rgba(SLATE, 180), width=4)
for j, (label, c) in enumerate(pair):
    yy = by + j * 152
    tinted_box(img, (x + 52, yy, x + PW - 52, yy + 104), c)
    d = ImageDraw.Draw(img)
    d.ellipse([x + 84, yy + 40, x + 108, yy + 64], fill=hex_rgba(c, 255))
    d.text((x + 134, yy + 30), label, font=jb(32, "Medium"), fill=TEXT)
d.text((x + 52, by + 356), "same topic, one intent flips —\npairing controls for confounds",
       font=jb(28, "Regular"), fill=MUTED, spacing=14)

# ---- panel 2 body: three competing explanations
x = PX[1]
rows = [
    ("SAE features (top-k)", VIOLET),
    ("mean-diff direction", CYAN),
    ("linear-probe direction", SLATE),
]
for j, (label, c) in enumerate(rows):
    yy = PY0 + 220 + j * 122
    tinted_box(img, (x + 52, yy, x + PW - 52, yy + 94), c)
    d = ImageDraw.Draw(img)
    d.ellipse([x + 84, yy + 35, x + 108, yy + 59], fill=hex_rgba(c, 255))
    d.text((x + 134, yy + 25), label, font=jb(32, "Medium"), fill=TEXT)
d.text((x + 52, PY0 + 220 + 3 * 122 + 34), "three rival explanations,\ntrained on the same records",
       font=jb(28, "Regular"), fill=MUTED, spacing=14)

# ---- panel 3 body: what validation may touch
x = PX[2]
sel = [("refusal threshold", True), ("steering scales", True), ("features / layers", False),
       ("directions", False)]
for j, (label, allowed) in enumerate(sel):
    yy = PY0 + 220 + j * 108
    c = EMERALD if allowed else ROSE
    tinted_box(img, (x + 52, yy, x + 118, yy + 66), c, radius=14, fill_alpha=26)
    d = ImageDraw.Draw(img)
    if allowed:
        draw_check(d, x + 85, yy + 32, 14, hex_rgba(c, 255))
    else:
        draw_cross(d, x + 85, yy + 33, 12, hex_rgba(c, 255))
    d.text((x + 150, yy + 14), label, font=jb(32, "Medium"), fill=TEXT if allowed else MUTED)
d.text((x + 52, PY0 + 220 + 4 * 108 + 26), "scales picked here are frozen\nbefore the test split is opened",
       font=jb(28, "Regular"), fill=MUTED, spacing=14)

# ---- panel 4 body: held-out causal tests + controls
x = PX[3]
tests = [("ablate · patch · steer at L27", CYAN), ("random-direction controls", SLATE),
         ("matched random-SAE controls", SLATE), ("layerwise controls", SLATE)]
for j, (label, c) in enumerate(tests):
    yy = PY0 + 220 + j * 108
    d.ellipse([x + 60, yy + 16, x + 90, yy + 46], outline=hex_rgba(c, 220), width=5)
    if j == 0:
        d.ellipse([x + 69, yy + 25, x + 81, yy + 37], fill=hex_rgba(c, 255))
    d.text((x + 118, yy + 4), label, font=jb(29, "Medium"), fill=TEXT if j == 0 else MUTED)
d.text((x + 52, PY0 + 220 + 4 * 108 + 26), "every headline number is read\noff this split exactly once",
       font=jb(28, "Regular"), fill=MUTED, spacing=14)

# footer ----------------------------------------------------------------------
d.line([120, H - 150, W - 120, H - 150], fill=HAIRLINE, width=2)
d.text((124, H - 118), "splits: 1,000 pairs → train / validation / held-out test  ·  scorer: contrastive prefix log-prob",
       font=jb(28, "Regular"), fill=FAINT)
right = "glassbox-audit · docs/paper.md"
f_r = jb(28, "Regular")
d.text((W - 124 - d.textlength(right, font=f_r), H - 118), right, font=f_r, fill=FAINT)

save(img, OUT)
