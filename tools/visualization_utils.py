from PIL import ImageDraw, ImageFont


TEXT_FONT = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 15)
TEXT_FILL = (255, 0, 255)
BAKCGROUND_OUTLINE, BAKCGROUND_FILL = (0, 0, 0), (0, 0, 0)


def draw_text(image, text, xy):
    """Draw text on image with guaranteed good contrast."""

    draw = ImageDraw.Draw(image)
    bounding_box = draw.textbbox(xy, text, TEXT_FONT)
    draw.rectangle(bounding_box, outline=BAKCGROUND_OUTLINE, fill=BAKCGROUND_FILL)
    draw.text(xy, text, font=TEXT_FONT, fill=TEXT_FILL)
