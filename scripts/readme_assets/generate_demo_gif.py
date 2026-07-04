"""docs/assets/demo.gif — styled replay of the real audit session.

Every value shown is taken verbatim from committed evidence files
(results/final/claim_summary.json); stage names mirror the pipeline's
checkpoint stages one-to-one.
"""

from pathlib import Path

from PIL import Image, ImageDraw
from style import (
    AMBER,
    CYAN,
    EMERALD,
    FAINT,
    HAIRLINE,
    MUTED,
    ROSE,
    TEXT,
    VIOLET,
    canvas,
    glass_panel,
    hex_rgba,
    jb,
)

OUT = Path(__file__).resolve().parents[2] / "docs" / "assets" / "demo.gif"

W, H = 1920, 1240
PAD_X, TOP_Y = 120, 210
LINE_H = 46
FONT = jb(30, "Regular")
FONT_B = jb(30, "Bold")
VISIBLE = 20  # rows in the terminal body

STAGES = [
    "data_load_split_validation",
    "model_load",
    "layer_scan",
    "sae_training",
    "feature_discovery",
    "validation_and_baseline_scoring",
    "held_out_steering_sweeps",
    "held_out_causal_tests",
    "random_direction_and_matched_sae_controls",
    "layerwise_controls",
    "failure_analysis_and_final_artifacts",
]

JSON_LINES = [
    ("{", TEXT),
    ('  "baseline_benign_overrefusal_rate": |0.148|,', None),
    ('  "baseline_harmful_refusal_rate": |0.64|,', None),
    ('  "cleanroom_reproduced": |true|,', "TRUE"),
    ('  "mean_direction_harmful_delta": |-0.1355500293970108|,', None),
    ('  "model": |"Qwen/Qwen2.5-1.5B-Instruct"|,', "STR"),
    ('  "probe_harmful_delta": |-0.0024561622142791747|,', None),
    ('  "sae_beats_mean_probe_held_out": |false|,', "FALSE"),
    ('  "sae_feature_harmful_delta": |-0.03633926343917847|,', None),
    ('  "target_layer": |27|', None),
    ("}", TEXT),
]


def base_frame() -> Image.Image:
    img = canvas(W, H, dots=True, glow=True)
    glass_panel(img, (56, 56, W - 56, H - 56), radius=36)
    d = ImageDraw.Draw(img)
    # title bar
    for i, c in enumerate([ROSE, AMBER, EMERALD]):
        x = 110 + i * 56
        d.ellipse([x, 104, x + 30, 134], fill=hex_rgba(c, 220))
    title = "glassbox-audit — audit session"
    f_t = jb(28, "Regular")
    d.text((W / 2 - d.textlength(title, font=f_t) / 2, 102), title, font=f_t, fill=FAINT)
    d.line([56, 168, W - 56, 168], fill=HAIRLINE, width=2)
    return img


BASE = base_frame()


def draw_segments(d, x, y, segments):
    for text, color, bold in segments:
        f = FONT_B if bold else FONT
        d.text((x, y), text, font=f, fill=color)
        x += d.textlength(text, font=f)
    return x


def line_segments(line):
    """Convert a logical line spec into colored segments."""
    kind, payload = line
    if kind == "cmd":
        segs = [("❯ ", CYAN, True)]
        first, *rest = payload.split(" ", 1)
        segs.append((first, EMERALD, True))
        if rest:
            segs.append((" " + rest[0], TEXT, False))
        return segs
    if kind == "cont":
        return [("    " + payload, TEXT, False)]
    if kind == "stage":
        return [("  › ", CYAN, False), ("completed ", FAINT, False), (payload, MUTED, False)]
    if kind == "stage_hl":
        return [("  › ", CYAN, False), ("completed ", FAINT, False), (payload, TEXT, False)]
    if kind == "json":
        text, tag = payload
        if "|" in text:
            pre, val, post = text.split("|")
            vcol = AMBER
            vbold = False
            if tag == "FALSE":
                vcol, vbold = ROSE, True
            elif tag == "TRUE":
                vcol = EMERALD
            elif tag == "STR":
                vcol = CYAN
            key_pre = pre
            return [(key_pre, VIOLET, False), (val, vcol, vbold), (post, TEXT, False)]
        return [(text, TEXT, False)]
    if kind == "comment":
        return [("❯ ", CYAN, True), (payload, FAINT, False)]
    if kind == "blank":
        return []
    return [(payload, TEXT, False)]


def render(lines, cursor_line=None, cursor_x_segments=None, show_cursor=False):
    img = BASE.copy()
    d = ImageDraw.Draw(img)
    view = lines[-VISIBLE:]
    for i, line in enumerate(view):
        y = TOP_Y + i * LINE_H
        x_end = draw_segments(d, PAD_X, y, line_segments(line))
        if show_cursor and i == len(view) - 1:
            d.rectangle([x_end + 6, y + 2, x_end + 24, y + 40], fill=hex_rgba(CYAN, 220))
    return img.resize((W // 2, H // 2), Image.LANCZOS)


frames: list[Image.Image] = []
durations: list[int] = []


def emit(lines, ms, cursor=False):
    frames.append(render(lines, show_cursor=cursor))
    durations.append(ms)


lines: list = []

# --- command 1: the audit run -------------------------------------------------
cmd1a = "glassbox run --config configs/qwen2.5-1.5b-expanded-audit.yaml \\"
cmd1b = "--output artifacts/qwen-expanded-1000"
typed = ""
for i in range(0, len(cmd1a) + 1, 5):
    typed = cmd1a[:i]
    emit(lines + [("cmd", typed)], 70, cursor=True)
lines.append(("cmd", cmd1a))
typed = ""
for i in range(0, len(cmd1b) + 1, 5):
    typed = cmd1b[:i]
    emit(lines + [("cont", typed)], 70, cursor=True)
lines.append(("cont", cmd1b))
emit(lines, 500, cursor=True)

for si, stage in enumerate(STAGES):
    kind = "stage_hl" if stage in ("layer_scan", "held_out_causal_tests") else "stage"
    lines.append((kind, stage))
    emit(lines, 340 if si < 4 else 200)
emit(lines, 900)

# --- command 2: read the committed verdict ------------------------------------
lines.append(("blank", ""))
cmd2 = "jq .headline_metrics results/final/claim_summary.json"
for i in range(0, len(cmd2) + 1, 4):
    emit(lines + [("cmd", cmd2[:i])], 70, cursor=True)
lines.append(("cmd", cmd2))
emit(lines, 400, cursor=True)

for j0 in range(0, len(JSON_LINES), 2):
    for jl in JSON_LINES[j0:j0 + 2]:
        lines.append(("json", jl))
    emit(lines, 140)
emit(lines, 2600)

# --- closing comment ------------------------------------------------------------
lines.append(("blank", ""))
comment = "# simple direction −0.136 vs SAE −0.036 → 0/6 preregistered passes"
for i in range(0, len(comment) + 1, 6):
    emit(lines + [("comment", comment[:i])], 70, cursor=True)
lines.append(("comment", comment))

for _ in range(3):
    emit(lines, 420, cursor=True)
    emit(lines, 420, cursor=False)
emit(lines, 1400)

OUT.parent.mkdir(parents=True, exist_ok=True)
quantized = [f.quantize(colors=128, method=Image.MEDIANCUT, dither=Image.NONE) for f in frames]
quantized[0].save(
    OUT,
    save_all=True,
    append_images=quantized[1:],
    duration=durations,
    loop=0,
    optimize=True,
)
print(f"wrote {OUT} ({len(frames)} frames, {OUT.stat().st_size / 1e6:.2f} MB)")
