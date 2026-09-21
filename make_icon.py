"""Renders the robot emoji into a multi-resolution .ico for the app shortcut."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent / "app_icon.ico"
FONT_PATH = r"C:\Windows\Fonts\seguiemj.ttf"
SIZE = 256

font = ImageFont.truetype(FONT_PATH, size=int(SIZE * 0.85))
img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
draw = ImageDraw.Draw(img)

bbox = draw.textbbox((0, 0), "\U0001F916", font=font, embedded_color=True)
w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
pos = ((SIZE - w) / 2 - bbox[0], (SIZE - h) / 2 - bbox[1])
draw.text(pos, "\U0001F916", font=font, embedded_color=True)

img.save(OUT, sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
print(f"Saved {OUT}")
