"""Stamp a faint diagonal 'Guitalele Tutorials' watermark on blog images.

Usage:
    python3 watermark_images.py blogs/guitalele-tabs blogs/guitalele-improvisation-tweaks

Each edited PNG keeps an untouched <name>.orig.png beside it, so re-running the
script re-stamps from the clean original instead of stacking watermarks.
Those backups are gitignored.
"""

from pathlib import Path
import shutil
import sys

from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageStat

TEXT = "Guitalele Tutorials"
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
ANGLE = 30          # degrees counter-clockwise
OPACITY = 14        # 0 = invisible, 255 = solid
DARK_BG_COLOR = (148, 163, 184)    # light slate, close to the dark diagrams
LIGHT_BG_COLOR = (175, 183, 198)   # subtle slate, for the hand-drawn white images
LIGHT_BG_THRESHOLD = 180
SIZE_RATIO = 0.055  # font size as a fraction of image width
ROOT = Path(__file__).resolve().parent


def watermark(src: Path) -> None:
    backup = src.with_name(src.stem + ".orig.png")

    if backup.exists():
        base = Image.open(backup).convert("RGBA")
    else:
        shutil.copy2(src, backup)
        base = Image.open(src).convert("RGBA")

    brightness = ImageStat.Stat(ImageOps.grayscale(base.convert("RGB"))).mean[0]
    color = LIGHT_BG_COLOR if brightness > LIGHT_BG_THRESHOLD else DARK_BG_COLOR

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.truetype(FONT_PATH, max(20, int(base.width * SIZE_RATIO)))

    draw.text(
        (base.width / 2, base.height / 2),
        TEXT,
        font=font,
        fill=(*color, OPACITY),
        anchor="mm",
    )

    overlay = overlay.rotate(ANGLE, resample=Image.Resampling.BICUBIC, expand=False)
    Image.alpha_composite(base, overlay).save(src)
    print(f"watermarked: {src.relative_to(ROOT)}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Pass at least one image folder.", file=sys.stderr)
        sys.exit(1)

    for raw in sys.argv[1:]:
        folder = Path(raw)
        if not folder.is_absolute():
            folder = ROOT / folder
        for image in sorted(folder.glob("*.png")):
            if not image.name.endswith(".orig.png"):
                watermark(image)


if __name__ == "__main__":
    main()
