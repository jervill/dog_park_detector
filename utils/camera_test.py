#!/usr/bin/env python3

from picamera import PiCamera
from PIL import Image, ImageDraw
from time import sleep

# Bounding boxes of interesting locations:
LOCATIONS = {
    'court_one': (0, 220,170, 380),
    'court_two': (170, 220, 408, 380),
    'dog_park': (540, 195, 820, 355),
}

def draw_rectangle(draw, x0, y0, x1, y1, border, fill=None, outline=None):
    assert border % 2 == 1
    for i in range(-border // 2, border // 2 + 1):
        draw.rectangle((x0 + i, y0 + i, x1 - i, y1 - i), fill=fill, outline=outline)

with PiCamera(sensor_mode=4, resolution=(820, 616), framerate=30) as camera:
    # Draw bounding boxes around locations
    # Load the arbitrarily sized image
    img = Image.new('RGB', (820, 616))
    draw = ImageDraw.Draw(img)

    for location in LOCATIONS.values():
        x1, y1, x2, y2 = location
        draw_rectangle(draw, x1, y1, x2, y2, 3, outline='white')

    # Create an image padded to the required size with
    # mode 'RGB'
    pad = Image.new('RGB', (
        ((img.size[0] + 31) // 32) * 32,
        ((img.size[1] + 15) // 16) * 16,
        ))
    # Paste the original image into the padded one
    pad.paste(img, (0, 0))

    # Add the overlay with the padded image as the source,
    # but the original image's dimensions
    camera.add_overlay(pad.tobytes(), alpha=64, layer=3, size=img.size)

    camera.start_preview()

    # Wait indefinitely until the user terminates the script
    while True:
        sleep(1)