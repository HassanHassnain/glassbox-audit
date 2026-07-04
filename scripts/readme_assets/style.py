"""Shared visual identity for all glassbox-audit README assets.

Every image and GIF frame in docs/assets is drawn through this module so the
whole set reads as one family: same palette, same fonts, same panel and chip
language, same background texture.

Palette story:
  - deep ink-navy glass panels on a dot grid            (the "glass box")
  - cyan   = mean-difference direction (the winner)
  - violet = SAE features
  - slate  = linear probe
  - emerald / amber / rose = supported / partial / failed verdicts
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ----------------------------------------------------------------------------- palette
BG_TOP = "#0A0F1C"
BG_BOTTOM = "#0D1424"
PANEL = "#101A2E"
PANEL_SOFT = "#0E1728"
HAIRLINE = "#22304F"
HAIRLINE_SOFT = "#1A2540"
DOT = "#182238"

TEXT = "#E9EFFA"
MUTED = "#93A1BE"
FAINT = "#5D6C8C"

CYAN = "#5FE1F2"      # mean-difference direction / primary accent
VIOLET = "#A78BFA"    # SAE features
SLATE = "#7E8EAC"     # linear probe
EMERALD = "#3DDC97"   # supported / pass
AMBER = "#FFC24B"     # partial / caution
ROSE = "#FF6B84"      # failed / rejected
GRAY = "#4A5878"      # random controls

METHOD_COLORS = {
    "mean_difference": CYAN,
    "sae_features": VIOLET,
    "linear_probe": SLATE,
    "random_controls": GRAY,
}

METHOD_LABELS = {
    "mean_difference": "MEAN-DIFF DIRECTION",
    "sae_features": "SAE FEATURES",
    "linear_probe": "LINEAR PROBE",
    "random_controls": "RANDOM CONTROLS",
}

# ----------------------------------------------------------------------------- fonts
FONT_DIR = Path(__file__).parent / "fonts"

_SG = "SpaceGrotesk-{w}.ttf"
_JB = "JetBrainsMono-{w}.ttf"


def sg(size: int, weight: str = "Bold") -> ImageFont.FreeTypeFont:
    """Space Grotesk — display / headings."""
    return ImageFont.truetype(str(FONT_DIR / _SG.format(w=weight)), size)


def jb(size: int, weight: str = "Regular") -> ImageFont.FreeTypeFont:
    """JetBrains Mono — data, labels, terminal."""
    return ImageFont.truetype(str(FONT_DIR / _JB.format(w=weight)), size)


# ----------------------------------------------------------------------------- helpers
def hex_rgba(color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    color = color.lstrip("#")
    return (int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16), alpha)


def canvas(width: int, height: int, dots: bool = True, glow: bool = True) -> Image.Image:
    """Dark ink canvas with vertical gradient, corner glows, and a dot grid."""
    img = Image.new("RGB", (width, height), BG_TOP)
    top = hex_rgba(BG_TOP)
    bottom = hex_rgba(BG_BOTTOM)
    grad = Image.new("RGB", (1, height))
    for y in range(height):
        t = y / max(1, height - 1)
        grad.putpixel(
            (0, y),
            tuple(int(top[i] + (bottom[i] - top[i]) * t) for i in range(3)),
        )
    img = grad.resize((width, height))

    if glow:
        layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        gd = ImageDraw.Draw(layer)
        r = int(max(width, height) * 0.55)
        gd.ellipse(
            [-r // 2, -r // 2, r // 2, r // 2], fill=hex_rgba(CYAN, 14)
        )
        gd.ellipse(
            [width - r // 2, height - r // 2, width + r // 2, height + r // 2],
            fill=hex_rgba(VIOLET, 12),
        )
        layer = layer.filter(ImageFilter.GaussianBlur(radius=r // 3))
        img = Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB")

    if dots:
        d = ImageDraw.Draw(img)
        step, radius = 56, 2
        for y in range(step, height, step):
            for x in range(step, width, step):
                d.ellipse([x - radius, y - radius, x + radius, y + radius], fill=DOT)
    return img


def glass_panel(
    img: Image.Image,
    box: tuple[int, int, int, int],
    radius: int = 26,
    fill: str = PANEL,
    fill_alpha: int = 235,
    stroke: str = HAIRLINE,
    highlight: bool = True,
) -> None:
    """Rounded translucent panel with hairline stroke and a top glass highlight."""
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.rounded_rectangle(box, radius=radius, fill=hex_rgba(fill, fill_alpha))
    d.rounded_rectangle(box, radius=radius, outline=hex_rgba(stroke, 255), width=2)
    if highlight:
        x0, y0, x1, _ = box
        d.line(
            [x0 + radius, y0 + 2, x1 - radius, y0 + 2],
            fill=(255, 255, 255, 16),
            width=2,
        )
    img.paste(Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB"), (0, 0))


def text_tracked(
    d: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill,
    tracking: int = 0,
) -> int:
    """Draw text with manual letter-spacing; returns end x."""
    x, y = xy
    for ch in text:
        d.text((x, y), ch, font=font, fill=fill)
        x += d.textlength(ch, font=font) + tracking
    return int(x)


def measure_tracked(
    d: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, tracking: int = 0
) -> int:
    return int(sum(d.textlength(ch, font=font) for ch in text) + tracking * max(0, len(text) - 1))


def chip(
    img: Image.Image,
    xy: tuple[int, int],
    label: str,
    color: str,
    font: ImageFont.FreeTypeFont | None = None,
    pad_x: int = 22,
    pad_y: int = 11,
    tracking: int = 2,
    dot: bool = True,
    fill_alpha: int = 26,
) -> tuple[int, int, int, int]:
    """Status pill: colored hairline, tinted fill, dot + mono uppercase label."""
    font = font or jb(22, "Medium")
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    text = label.upper()
    tw = measure_tracked(d, text, font, tracking)
    asc, desc = font.getmetrics()
    th = asc + desc
    dot_w = th - 10 if dot else 0
    x0, y0 = xy
    x1 = x0 + pad_x * 2 + dot_w + (10 if dot else 0) + tw
    y1 = y0 + pad_y * 2 + th
    r = (y1 - y0) // 2
    d.rounded_rectangle([x0, y0, x1, y1], radius=r, fill=hex_rgba(color, fill_alpha))
    d.rounded_rectangle([x0, y0, x1, y1], radius=r, outline=hex_rgba(color, 160), width=2)
    cx = x0 + pad_x
    if dot:
        cy = (y0 + y1) // 2
        rr = (th - 10) // 2
        d.ellipse([cx, cy - rr, cx + 2 * rr, cy + rr], fill=hex_rgba(color, 255))
        cx += 2 * rr + 10
    text_tracked(d, (cx, y0 + pad_y - 1), text, font, hex_rgba(color, 255), tracking)
    img.paste(Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB"), (0, 0))
    return (x0, y0, x1, y1)


def tinted_box(
    img: Image.Image,
    box: tuple[int, int, int, int],
    color: str,
    radius: int = 18,
    fill_alpha: int = 22,
    stroke_alpha: int = 130,
) -> None:
    """Translucent tinted rounded box — the glass language for inner elements."""
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.rounded_rectangle(box, radius=radius, fill=hex_rgba(color, fill_alpha))
    d.rounded_rectangle(box, radius=radius, outline=hex_rgba(color, stroke_alpha), width=2)
    img.paste(Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB"), (0, 0))


def draw_check(d: ImageDraw.ImageDraw, cx: int, cy: int, r: int, color, width: int = 6) -> None:
    d.line([cx - r, cy, cx - r * 0.25, cy + r * 0.7], fill=color, width=width, joint="curve")
    d.line([cx - r * 0.25, cy + r * 0.7, cx + r, cy - r * 0.6], fill=color, width=width, joint="curve")


def draw_cross(d: ImageDraw.ImageDraw, cx: int, cy: int, r: int, color, width: int = 6) -> None:
    d.line([cx - r, cy - r, cx + r, cy + r], fill=color, width=width)
    d.line([cx - r, cy + r, cx + r, cy - r], fill=color, width=width)


def footer(img: Image.Image, left: str, right: str = "glassbox-audit") -> None:
    """Bottom caption strip used on every figure."""
    d = ImageDraw.Draw(img)
    w, h = img.size
    f = jb(20, "Regular")
    d.line([56, h - 74, w - 56, h - 74], fill=HAIRLINE_SOFT, width=2)
    d.text((56, h - 58), left, font=f, fill=FAINT)
    tw = d.textlength(right, font=f)
    d.text((w - 56 - tw, h - 58), right, font=f, fill=FAINT)


def glow_line(
    img: Image.Image,
    p0: tuple[int, int],
    p1: tuple[int, int],
    color: str,
    width: int = 4,
    glow: int = 10,
) -> None:
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.line([p0, p1], fill=hex_rgba(color, 110), width=width + glow)
    layer = layer.filter(ImageFilter.GaussianBlur(radius=glow // 2))
    d = ImageDraw.Draw(layer)
    d.line([p0, p1], fill=hex_rgba(color, 255), width=width)
    img.paste(Image.alpha_composite(img.convert("RGBA"), layer).convert("RGB"), (0, 0))


def save(img: Image.Image, path: str | Path, scale: float = 0.5) -> None:
    """Assets are drawn at 2x and downsampled for crisp antialiasing."""
    out = img.resize(
        (int(img.width * scale), int(img.height * scale)), Image.LANCZOS
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path)
    print(f"wrote {path} ({out.width}x{out.height})")


# ----------------------------------------------------------------------------- cube mark
def cube_mark(size: int = 400, arrow: bool = True) -> Image.Image:
    """The glassbox logo mark: a wireframe glass cube with a causal direction
    passing straight through it, and sparse feature dots floating inside."""
    import math
    import random

    s = size
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    cx, cy = s * 0.5, s * 0.52
    r = s * 0.36  # circumradius of the hexagonal silhouette

    def pt(angle_deg: float, radius: float = 1.0) -> tuple[float, float]:
        a = math.radians(angle_deg)
        return (cx + radius * r * math.cos(a), cy - radius * r * math.sin(a))

    # classic isometric cube: hexagon corners + near vertex at the center
    A = pt(90)    # top
    B = pt(30)    # upper right
    C = pt(-30)   # lower right
    D = pt(-90)   # bottom
    E = pt(210)   # lower left
    F = pt(150)   # upper left
    M = (cx, cy)  # near vertex

    top = [A, B, M, F]
    right = [B, C, D, M]
    left = [F, M, D, E]

    glow = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    gd = ImageDraw.Draw(glow)
    d = ImageDraw.Draw(img)

    d.polygon(top, fill=hex_rgba(CYAN, 30))
    d.polygon(left, fill=hex_rgba(VIOLET, 24))
    d.polygon(right, fill=(255, 255, 255, 12))

    edge = hex_rgba("#9BE8F5", 240)
    w_edge = max(4, s // 130)
    d.line([A, B, C, D, E, F, A], fill=edge, width=w_edge, joint="curve")
    for corner in (B, F, D):
        d.line([M, corner], fill=edge, width=w_edge, joint="curve")

    # sparse feature dots inside the right/left faces (SAE latents)
    rng = random.Random(1415)
    faces = [right, left, top]
    for i in range(10):
        face = faces[i % 3]
        # random point inside a quad via bilinear interpolation
        u, v = rng.uniform(0.18, 0.82), rng.uniform(0.18, 0.82)
        p01 = [face[0][k] + (face[1][k] - face[0][k]) * u for k in (0, 1)]
        p32 = [face[3][k] + (face[2][k] - face[3][k]) * u for k in (0, 1)]
        px = p01[0] + (p32[0] - p01[0]) * v
        py = p01[1] + (p32[1] - p01[1]) * v
        rr = rng.choice([4, 5, 6]) * s / 400
        gd.ellipse([px - rr * 2.4, py - rr * 2.4, px + rr * 2.4, py + rr * 2.4],
                   fill=hex_rgba(VIOLET, 110))
        d.ellipse([px - rr, py - rr, px + rr, py + rr], fill=hex_rgba(VIOLET, 235))

    if arrow:
        y0 = int(cy - 0.30 * r)
        x_in, x_out = int(cx - 1.42 * r), int(cx + 1.42 * r)
        wl = max(5, s // 110)
        head = int(s * 0.075)
        gd.line([(x_in, y0), (x_out, y0)], fill=hex_rgba(CYAN, 130), width=wl * 3)
        d.line([(x_in, y0), (x_out - head + 4, y0)], fill=hex_rgba(CYAN, 255), width=wl)
        d.polygon(
            [(x_out, y0), (x_out - head, y0 - head // 2), (x_out - head, y0 + head // 2)],
            fill=hex_rgba(CYAN, 255),
        )

    glow = glow.filter(ImageFilter.GaussianBlur(radius=max(5, s // 90)))
    return Image.alpha_composite(glow, img)
