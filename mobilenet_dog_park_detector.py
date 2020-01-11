#!/usr/bin/env python3
#
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Script to run generic MobileNet based classification model."""
import os
import argparse
import time
import json
import random
from contextlib import ExitStack

from io import BytesIO
from PIL import Image, ImageDraw

from picamera import PiCamera, Color

from aiy.vision import inference
from aiy.vision.models import utils

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

def commit_data_to_long_term(interval, filename):
    def get_average(list):
        accumulator = 0
        for value in list:
            accumulator += value

        return round(accumulator / len(list),2)

    def reset_data():
        return {
            'time': [],
            'dog park': {
                'high activity': [],
                'low activity': [],
                'no activity': [],
            }
        }

    # Map labels to an int
    label_int_map = {
        'high activity' : 2,
        'low activity' : 1,
        'no activity' : 0,
    }

    # Save a file with an empty data structure.
    data = reset_data()
    with open(filename, 'w') as file:
        file.write(json.dumps({
            'results': []
        }))

    while True:
        processed_result = yield

        # Add the result to the data object
        data['time'].append(int(time.time()))
        for label, prob in processed_result:
            data['dog park'][label].append(prob)

        elapsed_time = data['time'][-1] - data['time'][0]

        # If we've hit the time interval, record to long term.
        if elapsed_time > interval:
            with open(filename, 'r+') as file:
                data_over_time = json.load(file)
                
                # [time, value]
                datapoint = [data['time'][-1],0]
                max_value = 0
                for key, value in data['dog park'].items():
                    average = get_average(value)

                    if average > max_value:
                        datapoint[1] = label_int_map[key]
                    
                data_over_time['results'].append(datapoint)
                data = reset_data()

                file.seek(0)
                file.write(json.dumps(data_over_time))


def read_labels(label_path):
    with open(label_path) as label_file:
        return [label.strip() for label in label_file.readlines()]


def get_message(result):
    result_as_string = [' %s (%.2f)' % (label, prob) for label, prob in result]
    if result:
        return 'Detecting:\n %s' % '\n '.join(result_as_string)


def process(result, labels, tensor_name):
    """Processes inference result and returns labels sorted by confidence."""
    # MobileNet based classification model returns one result vector.
    assert len(result.tensors) == 1
    tensor = result.tensors[tensor_name]
    probs, shape = tensor.data, tensor.shape
    assert shape.depth == len(labels)
    pairs = [pair for pair in enumerate(probs)]
    pairs = sorted(pairs, key=lambda pair: pair[1], reverse=True)
    labeled_pairs = [(labels[index], prob) for index, prob in pairs]
    return labeled_pairs


def get_cropped_images(camera):
    while True:
        stream = BytesIO()
        # Take picture
        camera.capture(stream, format='jpeg', use_video_port=True)
        # "Rewind" the stream to the beginning so we can read its content
        stream.seek(0)
        image = Image.open(stream)

        # Crop picture and return it
        dog_park = image.crop(LOCATIONS['dog_park'])
        court_one = image.crop(LOCATIONS['court_one'])
        court_two = image.crop(LOCATIONS['court_two'])
        yield (dog_park, court_one, court_two)


def _make_filename(image_folder, name, label, extension='jpeg'):
    subdirectory = '{label}/'.format(label=label) if label else ''
    path = '%s/Dog/%s%s.%s'
    filename = os.path.expanduser(path % (image_folder, subdirectory, name, extension))
    return filename


# TODO: Hardcode the arguments that never change anyway (like input height, width, layer)
# TODO: Add argument for each model and layer. Put into List, then run each if possible.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True,
        help='Path to converted model file that can run on VisionKit.')
    parser.add_argument('--label_path', required=True,
        help='Path to label file that corresponds to the model.')
    parser.add_argument('--input_height', type=int, required=True, help='Input height.')
    parser.add_argument('--input_width', type=int, required=True, help='Input width.')
    parser.add_argument('--input_layer', required=True, help='Name of input layer.')
    parser.add_argument('--output_layer', required=True, help='Name of output layer.')
    parser.add_argument('--num_frames', type=int, default=None,
        help='Sets the number of frames to run for, otherwise runs forever.')
    parser.add_argument('--input_mean', type=float, default=128.0, help='Input mean.')
    parser.add_argument('--input_std', type=float, default=128.0, help='Input std.')
    parser.add_argument('--input_depth', type=int, default=3, help='Input depth.')
    parser.add_argument('--preview', action='store_true', default=False,
        help='Enables camera preview in addition to printing result to terminal.')
    parser.add_argument('--show_fps', action='store_true', default=False,
        help='Shows end to end FPS.')
    parser.add_argument('--time_interval', type=int, default=10,
        help='Time interval at which to store data in seconds.')
    parser.add_argument('--gather_data', action='store_true', default=False,
        help='Also save images according to the assigned category.')
    parser.add_argument('--image_folder', default='/home/pi/Pictures/Data',
        help='Folder to save captured images')
    args = parser.parse_args()

    model = inference.ModelDescriptor(
        name='mobilenet_based_classifier',
        input_shape=(1, args.input_height, args.input_width, args.input_depth),
        input_normalizer=(args.input_mean, args.input_std),
        compute_graph=utils.load_compute_graph(args.model_path))
    labels = read_labels(args.label_path)

    # Check that the folder exists
    if args.gather_data:
        expected_subfolders = ['Dog', 'VB1', 'VB2']
        subfolders = os.listdir(args.image_folder)
        for folder in expected_subfolders:
            assert folder in subfolders

    with ExitStack() as stack:
        camera = stack.enter_context(PiCamera(sensor_mode=4, resolution=(820, 616), framerate=30))
        # TODO: Load the volleyball models too
        image_inference = stack.enter_context(inference.ImageInference(model))

        if args.preview:
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

        data_filename = _make_filename(args.image_folder, 'data', None, 'json')
        data_generator = commit_data_to_long_term(args.time_interval, data_filename)
        data_generator.send(None)

        # Capture one picture of entire scene each time it's started again.
        time.sleep(2)
        date = time.strftime('%Y-%m-%d')
        scene_filename = _make_filename(args.image_folder, date, None)
        camera.capture(scene_filename)

        # Draw bounding box on image showing the crop locations
        with Image.open(scene_filename) as scene:
            draw = ImageDraw.Draw(scene)

            for location in LOCATIONS.values():
                x1, y1, x2, y2 = location
                draw_rectangle(draw, x1, y1, x2, y2, 3, outline='white')

            scene.save(scene_filename)        

        # TODO: For each inference model, crop and process a different thing.
        # Constantly get cropped images
        for cropped_images in get_cropped_images(camera):
            cropped_image = cropped_images[0]

            # then run image_inference on them.
            result = image_inference.run(cropped_image)
            processed_result = process(result, labels, args.output_layer)
            data_generator.send(processed_result)
            message = get_message(processed_result)

            # Print the message
            print(message)

            timestamp = time.strftime('%Y-%m-%d_%H.%M.%S')
            print(timestamp + '\n')

            if args.gather_data:
                # Gather 1% data on 'no activity' since it's biased against that.
                # Gather 0.1% of all images.
                if(
                    (processed_result[0][0] == 'no activity' and random.random() > 0.99) or
                    (random.random() > 0.999)
                ):
                    filename = _make_filename(args.image_folder, timestamp, processed_result[0][0])
                    cropped_image.save(filename)

            if args.preview:
                camera.annotate_foreground = Color('black')
                camera.annotate_background = Color('white')
                # PiCamera text annotation only supports ascii.
                camera.annotate_text = '\n %s' % message.encode(
                    'ascii', 'backslashreplace').decode('ascii')

        if args.preview:
            camera.stop_preview()


if __name__ == '__main__':
    main()
