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
from aiy.vision.streaming.server import StreamingServer
from aiy.vision.streaming import svg

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

def svg_overlay(faces, frame_size, joy_score):
    width, height = frame_size
    doc = svg.Svg(width=width, height=height)

    for face in faces:
        x, y, w, h = face.bounding_box
        doc.add(svg.Rect(x=int(x), y=int(y), width=int(w), height=int(h), rx=10, ry=10,
                         fill_opacity=0.3 * face.face_score,
                         style='fill:red;stroke:white;stroke-width:4px'))

        doc.add(svg.Text('Joy: %.2f' % face.joy_score, x=x, y=y - 10,
                         fill='red', font_size=30))

    doc.add(svg.Text('Faces: %d Avg. joy: %.2f' % (len(faces), joy_score),
            x=10, y=50, fill='red', font_size=40))
    return str(doc)

def commit_data_to_long_term(interval, filename):
    def get_average(list):
        accumulator = 0
        for value in list:
            accumulator += value

        return round(accumulator / len(list),2)

    def reset_data():
        return {
            'time': [],
            'dog_park': {
                'high activity': [],
                'low activity': [],
                'no activity': [],
            },
            'court_one': {
                'high activity': [],
                'low activity': [],
                'no activity': [],
            },
            'court_two': {
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
    if not os.path.isfile(filename):
        with open(filename, 'w') as file:
            file.write(json.dumps({
                'dog_park': {},
                'court_one': {},
                'court_two': {},
            }))

    while True:
        location_name, processed_result = yield

        # Add the result to the data object
        data['time'].append(int(time.time()))
        for label, prob in processed_result:
            data[location_name][label].append(prob)

        elapsed_time = data['time'][-1] - data['time'][0]

        # If we've hit the time interval, record to long term.
        if elapsed_time > interval:
            with open(filename, 'r+') as file:
                data_over_time = json.load(file)
                
                # [location][time] = value
                logged_time = data['time'][-1]

                datapoint = 0

                max_value = 0                    
                for key, value in data[location_name].items():
                    average = get_average(value)

                    if average > max_value:
                        datapoint = label_int_map[key]

                    # Reset values
                    data[location_name][key] = []

                print(time.strftime('%Y-%m-%d_%H.%M.%S'))
                print(location_name + ':')
                print('  Got activity score of ' + str(datapoint))
                print('\n')

                data_over_time[location_name][logged_time] = datapoint

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

        yield {
            'dog_park': dog_park,
            'court_one': court_one,
            'court_two': court_two,
        }


def _make_filename(image_folder, name, label, extension='jpeg'):
    subdirectory = '{label}/'.format(label=label) if label else ''
    path = '%s/%s%s.%s'
    filename = os.path.expanduser(path % (image_folder, subdirectory, name, extension))
    return filename


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dog_park_model_path',
        help='Path to the model file for the dog park.')
    parser.add_argument('--vb1_model_path',
        help='Path to the model file for volley ball court 1.')
    parser.add_argument('--vb2_model_path',
        help='Path to the model file for volley ball court 1.')
    parser.add_argument('--label_path', required=True,
        help='Path to label file that corresponds to the model.')
    parser.add_argument('--input_mean', type=float, default=128.0, help='Input mean.')
    parser.add_argument('--input_std', type=float, default=128.0, help='Input std.')
    parser.add_argument('--input_depth', type=int, default=3, help='Input depth.')
    parser.add_argument('--enable_streaming', default=False, action='store_true',
                        help='Enable streaming server')
    parser.add_argument('--streaming_bitrate', type=int, default=1000000,
                        help='Streaming server video bitrate (kbps)')
    parser.add_argument('--mdns_name', default='',
                        help='Streaming server mDNS name')
    parser.add_argument('--preview', action='store_true', default=False,
        help='Enables camera preview in addition to printing result to terminal.')
    parser.add_argument('--time_interval', type=int, default=10,
        help='Time interval at which to store data in seconds.')
    parser.add_argument('--gather_data', action='store_true', default=False,
        help='Also save images according to the assigned category.')
    parser.add_argument('--image_folder', default='/home/pi/Pictures/Data',
        help='Folder to save captured images')
    args = parser.parse_args()

    labels = read_labels(args.label_path)

    # At least one model needs to be passed in.
    assert args.dog_park_model_path or args.vb1_model_path or args.vb2_model_path

    # Check that the folder exists
    if args.gather_data:
        expected_subfolders = ['dog_park', 'court_one', 'court_two']
        subfolders = os.listdir(args.image_folder)
        for folder in expected_subfolders:
            assert folder in subfolders

    with ExitStack() as stack:

        dog_park = {
            'location_name': 'dog_park',
            'path': args.dog_park_model_path,
        } if args.dog_park_model_path else None
        vb1 = {
            'location_name': 'court_one',
            'path': args.vb1_model_path,
        } if args.vb1_model_path else None
        vb2 = {
            'location_name': 'court_two',
            'path': args.vb2_model_path,
        } if args.vb2_model_path else None


        # Get the list of models, filter to only the ones that were passed in.
        models = [dog_park, vb1, vb2]
        models = list(filter(lambda model: model, models))

        # Initialize models and add them to the context
        for model in models:
            print('Initializing {model_name}...'.format(model_name=model["location_name"]))
            descriptor = inference.ModelDescriptor(
                name='mobilenet_based_classifier',
                input_shape=(1, 160, 160, args.input_depth),
                input_normalizer=(args.input_mean, args.input_std),
                compute_graph=utils.load_compute_graph(model['path']))

            model['descriptor'] = descriptor
        
        if dog_park:
            dog_park['image_inference'] = stack.enter_context(inference.ImageInference(dog_park['descriptor']))
        if vb1:
            vb1['image_inference'] = stack.enter_context(inference.ImageInference(vb1['descriptor']))
        if vb2:
            vb2['image_inference'] = stack.enter_context(inference.ImageInference(vb2['descriptor']))

        camera = stack.enter_context(PiCamera(sensor_mode=4, resolution=(820, 616), framerate=30))

        server = None
        svg_scale_factor = 1.32
        if args.enable_streaming:
            server = stack.enter_context(StreamingServer(camera, bitrate=args.streaming_bitrate,
                                                         mdns_name=args.mdns_name))

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

        # Constantly get cropped images
        for cropped_images in get_cropped_images(camera):

            svg_doc = None
            if args.enable_streaming:
                width = 820 * svg_scale_factor
                height = 616 * svg_scale_factor
                svg_doc = svg.Svg(width=width, height=height)

                for location in LOCATIONS.values():
                    x, y, x2, y2 = location
                    w = (x2 - x) * svg_scale_factor
                    h = (y2 - y) * svg_scale_factor
                    x = x * svg_scale_factor
                    y = y * svg_scale_factor
                    svg_doc.add(svg.Rect(x=int(x), y=int(y), width=int(w), height=int(h), rx=10, ry=10,
                                    fill_opacity=0.3,
                                    style='fill:none;stroke:white;stroke-width:4px'))

            # For each inference model, crop and process a different thing.
            for model in models:
                location_name = model['location_name']
                image_inference = model['image_inference']

                cropped_image = cropped_images[location_name]

                # then run image_inference on them.
                result = image_inference.run(cropped_image)
                processed_result = process(result, labels, 'final_result')
                data_generator.send((location_name, processed_result))
                message = get_message(processed_result)
                label = processed_result[0][0]

                # Print the message
                # print('\n')
                # print('{location_name}:'.format(location_name=location_name))
                # print(message)

                timestamp = time.strftime('%Y-%m-%d_%H.%M.%S')
                # print(timestamp)
                # print('\n')

                if args.gather_data:
                    # Gather 1% data on 'no activity' since it's biased against that.
                    # Gather 0.1% of all images.
                    if(
                        # (label == 'no activity' and random.random() > 0.99) or
                        # (random.random() > 0.999)
                        (location_name != 'dog_park' and random.random() > 0.99) or
                        (random.random() > 0.999)
                    ):
                        subdir = '{location_name}/{label}'.format(location_name=location_name, label=label)
                        filename = _make_filename(args.image_folder, timestamp, subdir)
                        cropped_image.save(filename)

                if svg_doc:
                    ## Plot points out
                    ## 160 x 80 grid
                    ## 16px width
                    ## 20, 40, 60 for 0, 1, 2
                    lines = message.split('\n')
                    y_correction = len(lines) * 20
                    for line in lines:
                        svg_doc.add(svg.Text(line,
                        x=(LOCATIONS[location_name][0]) * svg_scale_factor,
                        y=(LOCATIONS[location_name][1] - y_correction) * svg_scale_factor,
                        fill='white', font_size=20))

                        y_correction = y_correction - 20

                # TODO: Figure out how to annotate at specific locations.
                # if args.preview:
                #     camera.annotate_foreground = Color('black')
                #     camera.annotate_background = Color('white')
                #     # PiCamera text annotation only supports ascii.
                #     camera.annotate_text = '\n %s' % message.encode(
                #         'ascii', 'backslashreplace').decode('ascii')

            if server:
                server.send_overlay(str(svg_doc))

        if args.preview:
            camera.stop_preview()


if __name__ == '__main__':
    main()
