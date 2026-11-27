#!/usr/bin/env python3
"""Draw a PNG chord-shape sheet from shorthand tab lines.

Input format (one chord per line):
    [3f1s 0f2s 0f3s 0f4s 2f5s 3f6s]wd:C_Major
    [xf1s 3f2s 2f3s 0f4s 1f5s 0f6s]wd:F_Major d:optional note

`f1s` means fret on string 1. In the shorthand, string 1 is the high A string
and string 6 is the low A string. The drawn chart is reversed left → right:
low A, D, G, C, E, high A.

Usage:
    python3 draw_tabs_to_image.py my-tabs.txt
    python3 draw_tabs_to_image.py my-tabs.txt -o chords.png --title "My Chords"
    python3 draw_tabs_to_image.py my-tabs.txt --cols 2 --no-watermark
"""

from pathlib import Path
import argparse
import math
import re

from PIL import Image, ImageDraw, ImageFont

FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_SERIF = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"

# Palette from site/src/index.css
BG = (17, 20, 28)
PANEL = (24, 29, 40)
TEXT = (226, 232, 240)
MUTED = (136, 146, 176)
BORDER = (45, 55, 72)
GRID = (90, 100, 120)
ACCENT = (34, 211, 238)
SOFT = (148, 163, 184)

WATERMARK_TEXT = "Guitalele Tutorials"
WATERMARK_COLOR = (148, 163, 184)
WATERMARK_OPACITY = 14
WATERMARK_ANGLE = 30
WATERMARK_SIZE_RATIO = 0.055


def font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size)


def parse_tokens(tokens):
    frets = []
    for token in tokens.split():
        match = re.match(r"([0-9oOxX])f(\d+)s$", token, re.I)
        if not match:
            return None
        raw = match.group(1).lower()
        if raw == "x":
            frets.append(None)
        elif raw == "o":
            frets.append(0)
        else:
            frets.append(int(raw))
    return frets if len(frets) == 6 else None


def parse_score(path):
    title = None
    chords = []

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        if title is None:
            score_match = re.match(r"Score:\s*(.+)$", stripped, re.I)
            if score_match:
                title = score_match.group(1).strip()
                continue

        for match in re.finditer(r"\[([^\]]+)\]\s*wd:(.+?)(?:\s+d:|$)", stripped, re.I):
            token_text, label = match.groups()
            frets = parse_tokens(token_text)
            if frets is not None:
                chords.append((label.strip().replace("_", " "), frets))

    if title is None:
        title = path.stem.replace("_", " ").title()
    return title, chords


def draw_chord_grid(draw, x, y, frets, scale=1.0):
    string_gap = int(38 * scale)
    fret_h = int(34 * scale)
    grid_w = string_gap * 5
    grid_h = fret_h * 5
    dot_r = int(7 * scale)

    draw.line([(x, y), (x + grid_w, y)], fill=TEXT, width=max(2, int(4 * scale)))
    for i in range(1, 6):
        yy = y + i * fret_h
        draw.line([(x, yy), (x + grid_w, yy)], fill=GRID, width=max(1, int(2 * scale)))
    for i in range(6):
        xx = x + i * string_gap
        draw.line([(xx, y), (xx, y + grid_h)], fill=GRID, width=max(1, int(2 * scale)))

    for i, fret in enumerate(frets):
        cx = x + i * string_gap
        cy = y - int(18 * scale)
        if fret == 0:
            draw.ellipse(
                [cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r],
                outline=TEXT,
                width=max(1, int(2 * scale)),
            )
        elif fret is None:
            d = int(7 * scale)
            draw.line([(cx - d, cy - d), (cx + d, cy + d)], fill=MUTED, width=max(1, int(2 * scale)))
            draw.line([(cx + d, cy - d), (cx - d, cy + d)], fill=MUTED, width=max(1, int(2 * scale)))

    for i, fret in enumerate(frets):
        if not fret:
            continue
        cx = x + i * string_gap
        cy = y + (fret - 0.5) * fret_h
        draw.ellipse([cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r], fill=ACCENT)

    label_font = font(max(11, int(14 * scale)))
    for i, label in enumerate(["low A", "D", "G", "C", "E", "high A"]):
        cx = x + i * string_gap
        draw.text((cx, y + grid_h + int(10 * scale)), label, font=label_font, fill=MUTED, anchor="mt")


def apply_watermark(image):
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    watermark_font = ImageFont.truetype(FONT_SERIF, max(20, int(image.width * WATERMARK_SIZE_RATIO)))
    draw.text(
        (image.width / 2, image.height / 2),
        WATERMARK_TEXT,
        font=watermark_font,
        fill=(*WATERMARK_COLOR, WATERMARK_OPACITY),
        anchor="mm",
    )
    overlay = overlay.rotate(WATERMARK_ANGLE, resample=Image.Resampling.BICUBIC, expand=False)
    return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")


def draw_sheet(title, chords, output, cols=3, watermark=True):
    margin = 60
    cell_w = 500
    cell_h = 320
    gap = 20
    top = 190

    rows = math.ceil(len(chords) / cols)
    width = margin * 2 + cols * cell_w + (cols - 1) * gap
    height = top + rows * cell_h + margin

    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)

    draw.text((width / 2, 42), title, font=font(38, True), fill=TEXT, anchor="mt")
    draw.text(
        (width / 2, 100),
        "Left → right: low A, D, G, C, E, high A",
        font=font(20),
        fill=MUTED,
        anchor="mt",
    )
    draw.text(
        (width / 2, 138),
        "O = open string     X = muted string     Dot = finger down",
        font=font(18),
        fill=SOFT,
        anchor="mt",
    )

    for index, (label, frets) in enumerate(chords):
        col, row = index % cols, index // cols
        cell_x = margin + col * (cell_w + gap)
        cell_y = top + row * cell_h

        draw.rounded_rectangle(
            [cell_x, cell_y, cell_x + cell_w, cell_y + cell_h - gap],
            radius=14,
            outline=BORDER,
            width=2,
            fill=PANEL,
        )
        draw.text(
            (cell_x + cell_w / 2, cell_y + 20),
            label,
            font=font(24, True),
            fill=TEXT,
            anchor="mt",
        )
        draw_chord_grid(draw, cell_x + 72, cell_y + 90, list(reversed(frets)))

    if watermark:
        image = apply_watermark(image)

    image.save(output)
    print(f"Wrote {output} ({len(chords)} chords)")


def main():
    parser = argparse.ArgumentParser(description="Draw chord diagrams from shorthand tab lines.")
    parser.add_argument("input", type=Path, help="Input .txt tab file")
    parser.add_argument("-o", "--output", type=Path, help="Output PNG path")
    parser.add_argument("--title", help="Override the title from the Score: header")
    parser.add_argument("--cols", type=int, default=3, help="Number of chord columns")
    parser.add_argument("--no-watermark", action="store_true", help="Skip the watermark")
    args = parser.parse_args()

    title, chords = parse_score(args.input)
    if args.title:
        title = args.title
    if not chords:
        parser.error("No chord lines found. Expected lines like [3f1s ...]wd:C_Major")

    output = args.output or args.input.with_suffix(".png")
    draw_sheet(title, chords, output, cols=max(1, args.cols), watermark=not args.no_watermark)


if __name__ == "__main__":
    main()
