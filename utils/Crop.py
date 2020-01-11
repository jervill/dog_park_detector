#!/usr/bin/env python3

# Script to crop images to the right dimensions.

# Dimensions from original CropImages.py script.
# vb1 = input.crop(box=(0,240,170,400))
# vb2 = input.crop(box=(188, 240, 408,400))
# dog = input.crop(box=(559, 228, 820,388))

import os
import argparse
from PIL import Image, ImageDraw

def draw_rectangle(draw, x, y, width, height, border, fill=None, outline=None):
    assert border % 2 == 1
    for i in range(-border // 2, border // 2 + 1):
        draw.rectangle((x + i, y + i, x + width - i, y + height - i), fill=fill, outline=outline)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', required=True,
        help='Path to the directory that holds the images.')
    parser.add_argument('--filename_start', required=True,
        help='Only process files that begin with this.')
    parser.add_argument('--x', type=int, default=0, help='Leftmost starting position')
    parser.add_argument('--y', type=int, default=0, help='Topmost starting position')
    parser.add_argument('--height', type=int, default=160, help='Height')
    parser.add_argument('--width', type=int, default=160, help='Width')
    parser.add_argument('--dry_run', action='store_true', default=False,
        help='Instead of cropping, show the bounding box of what will be cropped')
    parser.add_argument('--prepend_name', default='',
        help='text to prepend to the output filename.')
    parser.add_argument('--output_directory', default=None,
        help='directory in which to save the output.')
    args = parser.parse_args()

    output_directory = args.directory if args.output_directory == None else args.output_directory

    print(f'Going to work on \'{args.directory}\'')
    print(f'Looking for files that match \'{args.directory}/{args.filename_start}\'')
    print('\n')

    with os.scandir(args.directory) as dir:
        number_matched = 0

        for file in dir:
            if file.name.startswith(args.filename_start) and file.name.endswith('jpeg') and file.is_file():
                print(f'Found {file.name}')
                number_matched += 1

        if number_matched == 0:
            print('Nothing matched.')
        else:
            print(f'Processing {number_matched} files.')

    print('\n')
    print(f'Checking to see that the output directory \'{output_directory}\' exists')
    os.listdir(output_directory)
    print('\n')

    with os.scandir(args.directory) as dir:

        x = args.x
        y = args.y
        width = args.width
        height = args.height
        test_loop = True

        for file in dir:

            if file.name.startswith(args.filename_start) and file.name.endswith('jpeg') and file.is_file():
                with Image.open(f'{args.directory}/{file.name}') as image:
                    max_width, max_height = image.size

                    # Update the height and width if they go beyond the image size
                    width = width if x + width < max_width else max_width - x
                    height = height if y + height < max_height else max_height - y

                    # For the first image in the list, show a preview.
                    while test_loop:
                        annotated_image = image.copy()
                        draw = ImageDraw.Draw(annotated_image)
                        draw_rectangle(draw, x, y, width, height, 3, outline='green')

                        annotated_image.show()
                        print('This is how the image will be cropped.')
                        print('\n')

                        user_input = input('Press enter to adjust preview or type "PROCEED" to crop at these dimensions: ')
                        if user_input == 'PROCEED':
                            print('\n')
                            test_loop = False

                        if test_loop:
                            x = input(f'Enter a new x value (default to {x}):') or x
                            y = input(f'Enter a new y value (default to {y}):') or y
                            width = input(f'Enter a new width (default to {width}):') or width
                            height = input(f'Enter a new height (default to {height}):') or height

                            x = int(x)
                            y = int(y)
                            width = int(width)
                            height = int(height)

                            # Update the height and width if they go beyond the image size.
                            if x + width > max_width or x + height > max_height:
                                width = width if x + width < max_width else max_width - x
                                height = height if y + height < max_height else max_height - y

                                print(f'Width adjusted to {width}')
                                print(f'Height adjusted to {height}')


                    cropped = image.crop((x, y, x + width, y + height))
                    cropped_width, cropped_height = cropped.size

                    # Resize if the image is smaller than 160 x 160
                    if cropped_height < 160 or cropped_width < 160:
                        resize_x = 160 if cropped_width < 160 else cropped_width
                        resize_y = 160 if cropped_height < 160 else cropped_height

                        new_dimensions = (resize_x, resize_y)
                        cropped = cropped.resize(new_dimensions)

                    output_filename = f'{output_directory}/{args.prepend_name}{file.name}'
                    print(f'Cropping {args.directory}/{file.name}')
                    cropped.save(output_filename)

    print('\n')
    print(f'{number_matched} images cropped to the following dimensions:')
    print(f'  x: {x}')
    print(f'  y: {y}')
    print(f'  width: {width}')
    print(f'  height: {height}')
                    

if __name__ == '__main__':
    main()