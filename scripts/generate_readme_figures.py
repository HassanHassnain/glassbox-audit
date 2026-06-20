from __future__ import annotations

import json
import os
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "docs" / "figures"

COLORS = {
    "bg_top": "#FBFCFE",
    "bg_bottom": "#EEF4F8",
    "panel": "#FFFFFF",
    "panel_soft": "#F5F8FB",
    "border": "#D7E0EA",
    "grid": "#E7EDF3",
    "ink": "#172033",
    "muted": "#617089",
    "faint": "#8B9AAF",
    "blue": "#2563EB",
    "blue_soft": "#DBEAFE",
    "teal": "#0F9F95",
    "teal_soft": "#DDF7F3",
    "coral": "#D94D3D",
    "coral_soft": "#FCE7E4",
    "gold": "#B7791F",
    "gold_soft": "#FEF3C7",
    "purple": "#6D5BD0",
    "purple_soft": "#ECE9FF",
    "slate": "#5B6B82",
}


def read_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def font(size: int, *, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    env_key = "GLASSBOX_FONT_BOLD" if bold else "GLASSBOX_FONT_REGULAR"
    candidates = [
        os.environ.get(env_key),
        "/System/Library/Fonts/SFNSMono.ttf" if mono else None,
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else None,
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else None,
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


FONTS = {
    "label": font(24, bold=True),
    "eyebrow": font(22, bold=True),
    "title": font(54, bold=True),
    "subtitle": font(30),
    "h2": font(36, bold=True),
    "h3": font(28, bold=True),
    "body": font(24),
    "small": font(20),
    "tiny": font(17),
    "mono": font(18, mono=True),
    "value": font(44, bold=True),
}


def rgb(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def canvas(width: int, height: int) -> Image.Image:
    image = Image.new("RGB", (width, height), rgb(COLORS["bg_top"]))
    draw = ImageDraw.Draw(image)
    top = rgb(COLORS["bg_top"])
    bottom = rgb(COLORS["bg_bottom"])
    for y in range(height):
        t = y / max(1, height - 1)
        row = tuple(round(top[i] * (1 - t) + bottom[i] * t) for i in range(3))
        draw.line([(0, y), (width, y)], fill=row)
    return image


def shadowed_round_rect(
    image: Image.Image,
    box: tuple[int, int, int, int],
    *,
    radius: int = 28,
    fill: str = "panel",
    outline: str = "border",
) -> None:
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    odraw = ImageDraw.Draw(overlay)
    x0, y0, x1, y1 = box
    odraw.rounded_rectangle((x0, y0 + 12, x1, y1 + 12), radius, fill=(25, 34, 51, 20))
    odraw.rounded_rectangle(
        box,
        radius,
        fill=(*rgb(COLORS[fill]), 255),
        outline=(*rgb(COLORS[outline]), 255),
        width=2,
    )
    image.alpha_composite(overlay) if image.mode == "RGBA" else image.paste(
        Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    )


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[int, int],
    max_width: int,
    *,
    text_font: ImageFont.FreeTypeFont,
    fill: str = "muted",
    line_gap: int = 9,
) -> int:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = word if not current else f"{current} {word}"
        if draw.textlength(trial, font=text_font) <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    x, y = xy
    for line in lines:
        draw.text((x, y), line, font=text_font, fill=COLORS[fill])
        y += text_font.size + line_gap
    return y


def text_center(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    *,
    text_font: ImageFont.FreeTypeFont,
    fill: str = "ink",
) -> None:
    bbox = draw.textbbox((0, 0), text, font=text_font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x0, y0, x1, y1 = box
    draw.text(
        (x0 + (x1 - x0 - width) / 2, y0 + (y1 - y0 - height) / 2 - 2),
        text,
        font=text_font,
        fill=COLORS[fill],
    )


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def delta(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def draw_badge(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    fill: str = "blue_soft",
    color: str = "blue",
) -> tuple[int, int, int, int]:
    x, y = xy
    pad_x = 18
    pad_y = 9
    bbox = draw.textbbox((0, 0), text, font=FONTS["small"])
    w = bbox[2] - bbox[0] + pad_x * 2
    h = bbox[3] - bbox[1] + pad_y * 2
    draw.rounded_rectangle((x, y, x + w, y + h), 18, fill=COLORS[fill])
    draw.text((x + pad_x, y + pad_y - 1), text, font=FONTS["small"], fill=COLORS[color])
    return (x, y, x + w, y + h)


def xmap(value: float, domain: tuple[float, float], span: tuple[int, int]) -> int:
    lo, hi = domain
    x0, x1 = span
    return round(x0 + (value - lo) / (hi - lo) * (x1 - x0))


def draw_axis(
    draw: ImageDraw.ImageDraw,
    span: tuple[int, int],
    y: int,
    domain: tuple[float, float],
    ticks: list[float],
) -> None:
    x0, x1 = span
    draw.line((x0, y, x1, y), fill=COLORS["border"], width=2)
    for tick in ticks:
        x = xmap(tick, domain, span)
        draw.line((x, y - 8, x, y + 8), fill=COLORS["border"], width=2)
        label = f"{tick:.2f}" if abs(tick) >= 0.01 else "0"
        bbox = draw.textbbox((0, 0), label, font=FONTS["tiny"])
        draw.text((x - (bbox[2] - bbox[0]) / 2, y + 14), label, font=FONTS["tiny"], fill=COLORS["muted"])


def generate_overview() -> None:
    claim = read_json("results/final/claim_summary.json")
    metrics = claim["headline_metrics"]
    image = canvas(1600, 840).convert("RGBA")
    draw = ImageDraw.Draw(image)

    draw.text((72, 58), "GLASSBOX AUDIT", font=FONTS["eyebrow"], fill=COLORS["blue"])
    draw.text((72, 94), "Refusal behavior, tested causally", font=FONTS["title"], fill=COLORS["ink"])
    summary = (
        "Layer-27 residual direction in Qwen2.5-1.5B; OR-Bench transfer; partial "
        "Qwen2.5-3B replication. Boundary: no circuit claim, and SAE does not beat "
        "mean/probe baselines."
    )
    draw_wrapped(draw, summary, (72, 166), 1320, text_font=FONTS["body"], fill="muted", line_gap=8)

    cards = [
        ("Target layer", str(metrics["target_layer"]), "Qwen2.5-1.5B residual stream", "blue"),
        ("Mean direction", delta(metrics["mean_direction_harmful_delta"]), "held-out harmful-score delta", "teal"),
        ("SAE feature", delta(metrics["sae_feature_harmful_delta"]), "weaker than mean/probe baselines", "coral"),
        ("Clean-room", "matched", "reproduced within fixed tolerances", "purple"),
    ]
    card_w = 350
    for i, (label, value, note, color) in enumerate(cards):
        x = 72 + i * (card_w + 24)
        shadowed_round_rect(image, (x, 280, x + card_w, 436), radius=24)
        draw.rounded_rectangle((x + 24, 304, x + 64, 344), 14, fill=COLORS[f"{color}_soft"])
        draw.ellipse((x + 38, 318, x + 50, 330), fill=COLORS[color])
        draw.text((x + 82, 302), label, font=FONTS["small"], fill=COLORS["muted"])
        draw.text((x + 24, 344), value, font=FONTS["value"], fill=COLORS[color])
        draw.text((x + 24, 398), note, font=FONTS["small"], fill=COLORS["muted"])

    shadowed_round_rect(image, (72, 490, 1528, 758), radius=30)
    draw.text((108, 522), "Release evidence chain", font=FONTS["h2"], fill=COLORS["ink"])
    steps = [
        ("Train-only discovery", "no held-out peeking"),
        ("Validation selection", "thresholds and scales fixed"),
        ("Held-out causal test", "mean direction strongest"),
        ("External OR-Bench", "transfer without rediscovery"),
        ("Clean-room rerun", "same claim boundary"),
    ]
    node_y = 610
    left = 180
    gap = 260
    for i, (title, note) in enumerate(steps):
        cx = left + i * gap
        if i:
            draw.line((cx - gap + 66, node_y, cx - 66, node_y), fill=COLORS["border"], width=8)
            draw.line((cx - gap + 66, node_y, cx - 66, node_y), fill=COLORS["blue"], width=3)
        draw.ellipse((cx - 42, node_y - 42, cx + 42, node_y + 42), fill=COLORS["blue_soft"], outline=COLORS["blue"], width=3)
        text_center(draw, (cx - 42, node_y - 42, cx + 42, node_y + 42), str(i + 1), text_font=FONTS["h3"], fill="blue")
        title_box = draw.textbbox((0, 0), title, font=FONTS["small"])
        note_box = draw.textbbox((0, 0), note, font=FONTS["tiny"])
        draw.text(
            (cx - (title_box[2] - title_box[0]) / 2, node_y + 58),
            title,
            font=FONTS["small"],
            fill=COLORS["ink"],
        )
        draw.text(
            (cx - (note_box[2] - note_box[0]) / 2, node_y + 88),
            note,
            font=FONTS["tiny"],
            fill=COLORS["muted"],
        )

    image.convert("RGB").save(FIGURES / "readme_audit_overview.png", quality=96, optimize=True)


def method_label(method: str) -> str:
    return {
        "mean_difference": "Mean direction",
        "mean_difference_direction_ablation": "Mean direction",
        "sae_features": "SAE features",
        "sae_feature_ablation": "SAE features",
        "linear_probe": "Linear probe",
        "linear_probe_direction_ablation": "Linear probe",
    }.get(method, method.replace("_", " "))


def generate_causal_effects() -> None:
    audit = read_json("results/final/qwen2_5_1_5b_audit.json")
    external = read_json("results/external-causal/or_bench_qwen15b_1000_summary.json")
    image = canvas(1600, 920).convert("RGBA")
    draw = ImageDraw.Draw(image)
    draw.text((70, 52), "Causal effects: simple directions beat SAE features", font=FONTS["title"], fill=COLORS["ink"])
    draw.text(
        (72, 120),
        "Lower harmful-score delta is stronger suppression. CIs are paired bootstrap intervals where available.",
        font=FONTS["body"],
        fill=COLORS["muted"],
    )

    shadowed_round_rect(image, (70, 180, 990, 820), radius=28)
    draw.text((108, 216), "Held-out Qwen2.5-1.5B audit", font=FONTS["h2"], fill=COLORS["ink"])
    draw.text((108, 260), "Harmful-score delta after ablation", font=FONTS["small"], fill=COLORS["muted"])
    domain = (-0.16, 0.012)
    span = (395, 910)
    zero_x = xmap(0, domain, span)
    draw.line((zero_x, 315, zero_x, 690), fill=COLORS["grid"], width=4)
    rows = [
        ("mean_difference_direction_ablation", audit["mean_harmful_delta"], audit["mean_harmful_ci"], "blue"),
        ("sae_feature_ablation", audit["sae_harmful_delta"], audit["sae_harmful_ci"], "teal"),
        ("linear_probe_direction_ablation", audit["probe_harmful_delta"], [-0.003124, -0.001749], "purple"),
    ]
    for idx, (method, value, ci, color) in enumerate(rows):
        y = 348 + idx * 120
        draw.text((108, y - 20), method_label(method), font=FONTS["h3"], fill=COLORS["ink"])
        draw.text((108, y + 18), f"{delta(value)} harmful-score delta", font=FONTS["small"], fill=COLORS["muted"])
        vx = xmap(value, domain, span)
        lo = xmap(ci[0], domain, span)
        hi = xmap(ci[1], domain, span)
        draw.line((lo, y + 2, hi, y + 2), fill=COLORS["slate"], width=4)
        draw.line((lo, y - 12, lo, y + 16), fill=COLORS["slate"], width=4)
        draw.line((hi, y - 12, hi, y + 16), fill=COLORS["slate"], width=4)
        draw.rounded_rectangle((min(vx, zero_x), y - 16, max(vx, zero_x), y + 20), 10, fill=COLORS[f"{color}_soft"])
        draw.ellipse((vx - 13, y - 13, vx + 13, y + 13), fill=COLORS[color])
    draw_axis(draw, span, 725, domain, [-0.15, -0.10, -0.05, 0.0])

    shadowed_round_rect(image, (1040, 180, 1530, 820), radius=28)
    draw.text((1078, 216), "External OR-Bench transfer", font=FONTS["h2"], fill=COLORS["ink"])
    draw.text((1078, 260), "Frozen controlled-corpus interventions", font=FONTS["small"], fill=COLORS["muted"])
    methods = external["summary"]["main_methods"]
    ext_rows = [
        ("mean_difference_direction_ablation", methods["mean_difference_direction_ablation"], "blue"),
        ("sae_feature_ablation", methods["sae_feature_ablation"], "teal"),
        ("linear_probe_direction_ablation", methods["linear_probe_direction_ablation"], "purple"),
    ]
    y0 = 332
    for idx, (method, values, color) in enumerate(ext_rows):
        y = y0 + idx * 128
        draw.text((1078, y - 34), method_label(method), font=FONTS["h3"], fill=COLORS["ink"])
        score = values["harmful_score_delta"]
        toxic = values["toxic_refusal_rate_delta"]
        bar_x = 1295
        score_w = int(abs(score) / 0.17 * 180)
        toxic_w = int(abs(toxic) / 0.70 * 180)
        draw.rounded_rectangle((bar_x, y - 28, bar_x + 180, y - 8), 10, fill=COLORS["grid"])
        draw.rounded_rectangle((bar_x, y - 28, bar_x + score_w, y - 8), 10, fill=COLORS[color])
        draw.text((1078, y - 2), f"score {delta(score)}", font=FONTS["small"], fill=COLORS["muted"])
        draw.rounded_rectangle((bar_x, y + 28, bar_x + 180, y + 48), 10, fill=COLORS["grid"])
        draw.rounded_rectangle((bar_x, y + 28, bar_x + toxic_w, y + 48), 10, fill=COLORS[f"{color}_soft"])
        draw.text((1078, y + 52), f"toxic refusal rate {pct(toxic)}", font=FONTS["small"], fill=COLORS["muted"])
    draw_badge(draw, (1078, 722), "External result is transfer, not rediscovery", fill="gold_soft", color="gold")

    image.convert("RGB").save(FIGURES / "readme_causal_effects.png", quality=96, optimize=True)


def color_for_delta(value: float) -> str:
    if value < -0.08:
        return "#2563EB"
    if value < -0.02:
        return "#7DD3FC"
    if value < 0:
        return "#D6EAFE"
    return "#F4A69D"


def generate_stability_path() -> None:
    stability = read_json("results/sae-stability/stability_grid.json")
    component = read_json("results/component-path/component_path_summary.json")
    cells = stability["cells"]
    image = canvas(1600, 940).convert("RGBA")
    draw = ImageDraw.Draw(image)
    draw.text((70, 52), "Stability and path evidence keep the claim bounded", font=FONTS["title"], fill=COLORS["ink"])
    draw.text(
        (72, 120),
        "SAE effects are stable but do not pass the held-out SAE-vs-baseline criterion; component analysis points to residual stream, not a full circuit.",
        font=FONTS["body"],
        fill=COLORS["muted"],
    )

    shadowed_round_rect(image, (70, 180, 770, 840), radius=28)
    draw.text((108, 218), "SAE stability grid", font=FONTS["h2"], fill=COLORS["ink"])
    draw_badge(draw, (108, 266), "6/6 cells completed", fill="teal_soft", color="teal")
    draw_badge(draw, (330, 266), "0/6 preregistered H3 passes", fill="coral_soft", color="coral")
    domain = (-0.16, 0.0)
    span = (308, 690)
    zero_x = xmap(0, domain, span)
    mean_reference = sum(c["mean_harmful_delta"] for c in cells) / len(cells)
    mean_x = xmap(mean_reference, domain, span)
    draw.line((mean_x, 330, mean_x, 744), fill=COLORS["blue"], width=4)
    draw.text((mean_x - 78, 754), "mean-direction reference", font=FONTS["tiny"], fill=COLORS["blue"])
    for idx, cell in enumerate(cells):
        y = 348 + idx * 62
        label = f"seed {cell['seed']} x{cell['width']}"
        draw.text((108, y - 10), label, font=FONTS["small"], fill=COLORS["ink"])
        value = cell["sae_harmful_delta"]
        vx = xmap(value, domain, span)
        draw.rounded_rectangle((vx, y - 12, zero_x, y + 14), 9, fill=COLORS["teal_soft"])
        draw.ellipse((vx - 9, y - 9, vx + 9, y + 9), fill=COLORS["teal"])
        draw.text((708, y - 10), delta(value), font=FONTS["tiny"], fill=COLORS["muted"], anchor="ra")
    draw_axis(draw, span, 790, domain, [-0.15, -0.10, -0.05, 0.0])

    shadowed_round_rect(image, (830, 180, 1530, 840), radius=28)
    draw.text((868, 218), "Component/path analysis", font=FONTS["h2"], fill=COLORS["ink"])
    draw.text((868, 262), "Harmful-score delta by layer and component", font=FONTS["small"], fill=COLORS["muted"])
    artifact = component["artifacts"][0]
    rows = artifact["component_outputs"]
    layers = artifact["layers"]
    components = ["residual", "attention", "mlp"]
    grid_x = 1006
    grid_y = 340
    cell_w = 150
    cell_h = 86
    for j, name in enumerate(components):
        text_center(draw, (grid_x + j * cell_w, grid_y - 50, grid_x + (j + 1) * cell_w, grid_y - 10), name, text_font=FONTS["small"], fill="muted")
    for i, layer in enumerate(layers):
        draw.text((882, grid_y + i * cell_h + 28), f"layer {layer}", font=FONTS["small"], fill=COLORS["ink"])
        for j, comp in enumerate(components):
            row = next(r for r in rows if r["layer"] == layer and r["component"] == comp)
            x0 = grid_x + j * cell_w
            y0 = grid_y + i * cell_h
            draw.rounded_rectangle(
                (x0 + 8, y0 + 8, x0 + cell_w - 8, y0 + cell_h - 8),
                18,
                fill=color_for_delta(row["harmful_delta"]),
                outline=COLORS["panel"],
                width=3,
            )
            text_center(
                draw,
                (x0 + 8, y0 + 8, x0 + cell_w - 8, y0 + cell_h - 8),
                delta(row["harmful_delta"]),
                text_font=FONTS["small"],
                fill="ink",
            )
    draw_badge(draw, (868, 720), "Residual stream dominates", fill="blue_soft", color="blue")
    draw_badge(draw, (1168, 720), "No circuit claim", fill="gold_soft", color="gold")

    image.convert("RGB").save(FIGURES / "readme_stability_path.png", quality=96, optimize=True)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    generate_overview()
    generate_causal_effects()
    generate_stability_path()


if __name__ == "__main__":
    main()
