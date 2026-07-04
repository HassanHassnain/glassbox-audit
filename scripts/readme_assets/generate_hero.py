"""docs/assets/hero.png — repo banner."""

from pathlib import Path

from PIL import ImageDraw
from style import (
    AMBER,
    CYAN,
    EMERALD,
    FAINT,
    MUTED,
    ROSE,
    TEXT,
    canvas,
    chip,
    cube_mark,
    glass_panel,
    jb,
    save,
    sg,
    text_tracked,
)

OUT = Path(__file__).resolve().parents[2] / "docs" / "assets" / "hero.png"

W, H = 3360, 1000
img = canvas(W, H)
glass_panel(img, (56, 56, W - 56, H - 56), radius=40)

# mark
mark = cube_mark(size=760)
img.paste(mark, (140, H // 2 - 380), mark)

d = ImageDraw.Draw(img)
x0 = 960

# wordmark
name_font = sg(172, "Bold")
end = text_tracked(d, (x0, 170), "GLASSBOX", name_font, TEXT, tracking=4)
d.line([end + 44, 200, end + 44, 330], fill=CYAN, width=6)
text_tracked(d, (end + 92, 196), "AUDIT", sg(118, "Medium"), CYAN, tracking=26)

# tagline
d.text(
    (x0 + 6, 424),
    "A held-out causal activation audit of refusal behavior",
    font=sg(64, "Medium"),
    fill=MUTED,
)
text_tracked(
    d,
    (x0 + 8, 532),
    "QWEN2.5-1.5B-INSTRUCT  ·  LAYER 27  ·  1,000 PAIRED PROMPTS  ·  PREREGISTERED",
    jb(34, "Regular"),
    FAINT,
    tracking=2,
)

# verdict chips — the honest headline
cy = 680
f = jb(34, "Medium")
box = chip(img, (x0, cy), "residual direction: supported", EMERALD, font=f, pad_x=30, pad_y=18)
box = chip(img, (box[2] + 36, cy), "3B replication: partial", AMBER, font=f, pad_x=30, pad_y=18)
chip(img, (box[2] + 36, cy), "SAE > baselines: rejected", ROSE, font=f, pad_x=30, pad_y=18)

d = ImageDraw.Draw(img)
d.text((x0 + 4, 830), "The negative result is the point.", font=sg(44, "Medium"), fill=FAINT)

save(img, OUT)
