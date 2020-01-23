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

from wand.image import Image as WandImage
from wand.color import Color

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
SVG_SCALE_FACTOR = 1.32

TIMELAPSE_DELAY = 120

def draw_rectangle(draw, x0, y0, x1, y1, border, fill=None, outline=None):
    assert border % 2 == 1
    for i in range(-border // 2, border // 2 + 1):
        draw.rectangle((x0 + i, y0 + i, x1 - i, y1 - i), fill=fill, outline=outline)

def plot_svg_chart(svg_doc, location, data):
    x, y, x2, y2 = LOCATIONS[location]
    plotted_data = sorted(list(data[location].keys()))[-20:]

    for pos, time in enumerate(plotted_data):
        start_x = (x + 8 * pos) * SVG_SCALE_FACTOR
        width = 4 * SVG_SCALE_FACTOR
        start_y = (y - 20 * data[location][time])
        height = (y - start_y) * SVG_SCALE_FACTOR
        start_y = start_y * SVG_SCALE_FACTOR

        color = 'red' if data[location][time] == 2 else 'green'

        svg_doc.add(svg.Rect(x=int(start_x), y=int(start_y), width=int(width), height=int(height), fill_opacity=0.3,
                                    style='fill:'+color+';stroke:'+color+';stroke-width:4px'))

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
        location_name, processed_result, svg_doc = yield

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
                logged_time = str(data['time'][-1])

                datapoint = 0

                max_value = 0

                for key, value in data[location_name].items():
                    average = get_average(value)

                    if average > max_value:
                        datapoint = label_int_map[key]

                    # Reset values
                    data[location_name][key] = []

                print('\n' + time.strftime('%Y-%m-%d_%H.%M.%S'))
                print(location_name + ' activity score:' + str(datapoint))

                data_over_time[location_name][logged_time] = datapoint

                ## Plot most recent data on svg
                plot_svg_chart(svg_doc, location_name, data_over_time)

                file.seek(0)
                file.write(json.dumps(data_over_time))
            if location_name == 'court_two':
                data = reset_data()
                print('\n')


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


def get_cropped_images(camera, timelapse):
    next_timelapse = int(time.time()) + TIMELAPSE_DELAY

    run_inference_until = {
        'dog_park': False,
        'court_one': False,
        'court_two': False,
    }

    while True:
        stream = BytesIO()
        # Take picture
        camera.capture(stream, format='jpeg', use_video_port=True)
        # "Rewind" the stream to the beginning so we can read its content
        stream.seek(0)
        image_1 = Image.open(stream)

        # Crop picture and return it
        dog_park_1 = image_1.crop(LOCATIONS['dog_park'])
        court_one_1 = image_1.crop(LOCATIONS['court_one'])
        court_two_1 = image_1.crop(LOCATIONS['court_two'])

        # (Image Comparison) Wait one second
        time.sleep(1)

        # (Image Comparison) Take another picture, crop it
        stream.seek(0)
        camera.capture(stream, format='jpeg', use_video_port=True)
        # "Rewind" the stream to the beginning so we can read its content
        stream.seek(0)
        image_2 = Image.open(stream)

        # Crop picture and return it
        dog_park_2 = image_2.crop(LOCATIONS['dog_park'])
        court_one_2 = image_2.crop(LOCATIONS['court_one'])
        court_two_2 = image_2.crop(LOCATIONS['court_two'])

        result = {
            'dog_park': 'dog park',
            'court_one': 'court one',
            'court_two': 'court two',
        }

        # TODO: (Image Comparison) Compare crops
        pics_to_compare = [
            ('dog_park', dog_park_1, dog_park_2),
            ('court_one', court_one_1, court_one_2),
            ('court_two', court_two_1, court_two_2)
        ]

        for key, img1, img2 in pics_to_compare:
            if run_inference_until[key] and run_inference_until[key] > time.time():
                print('Running inference on ' + key + ' at ' + time.strftime('%H.%M.%S'))
                result[key] = img2
            else:
                with BytesIO() as bytes1:
                    with BytesIO() as bytes2:
                        img1.save(bytes1, 'jpeg')
                        img2.save(bytes2, 'jpeg')

                        img1_bytes = bytes1.getvalue()
                        img2_bytes = bytes2.getvalue()
                        
                        with WandImage(blob=img1_bytes) as base:
                            with WandImage(blob=img2_bytes) as change:
                                base.fuzz = base.quantum_range * 0.20
                                _, result_metric = base.compare(change, metric='absolute')

                                if result_metric > 100:
                                    print('Movement in ' + key +  ': ' + str(result_metric))
                                    result[key] = img2
                                    # Run inference for the next 5 minutes
                                    run_inference_until[key] = time.time() + 300
                                    print('Run inference until: ' + str(run_inference_until[key]))
                                else:
                                    result[key] = False
                                    run_inference_until[key] = False
                    


        # If they are not different, mark as False which later is interpreted as "no activity"
        # If they are different, return the cropped image so an inference check can be done on it.
        if timelapse:
            if int(time.time()) > next_timelapse:
                filename = '/home/pi/Pictures/Data/' + time.strftime('%Y-%m-%d_%H.%M.%S' + '.jpeg')
                image_2.save(filename)
                next_timelapse = int(time.time()) + TIMELAPSE_DELAY
        
        yield result


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
    parser.add_argument('--timelapse', action='store_true', default=False,
        help='Also save some timelapses of the entire scene, every 120 seconds.')
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
        for cropped_images in get_cropped_images(camera, args.timelapse):

            svg_doc = None
            if args.enable_streaming:
                width = 820 * SVG_SCALE_FACTOR
                height = 616 * SVG_SCALE_FACTOR
                svg_doc = svg.Svg(width=width, height=height)

                for location in LOCATIONS.values():
                    x, y, x2, y2 = location
                    w = (x2 - x) * SVG_SCALE_FACTOR
                    h = (y2 - y) * SVG_SCALE_FACTOR
                    x = x * SVG_SCALE_FACTOR
                    y = y * SVG_SCALE_FACTOR
                    svg_doc.add(svg.Rect(x=int(x), y=int(y), width=int(w), height=int(h), rx=10, ry=10,
                                    fill_opacity=0.3,
                                    style='fill:none;stroke:white;stroke-width:4px'))

            # For each inference model, crop and process a different thing.
            for model in models:
                location_name = model['location_name']
                image_inference = model['image_inference']

                cropped_image = cropped_images[location_name]
                
                # TODO: (Image Comparison) If False,return no activity.
                if cropped_image:
                    # then run image_inference on them.
                    result = image_inference.run(cropped_image)
                    processed_result = process(result, labels, 'final_result')
                    data_generator.send((location_name, processed_result, svg_doc))
                    message = get_message(processed_result)

                    # Print the message
                    # print('\n')
                    # print('{location_name}:'.format(location_name=location_name))
                    # print(message)
                else:
                    # Fake processed_result
                    processed_result = [('no activity', 1.00),('low activity', 0.00),('high activity', 0.00)]
                    data_generator.send((location_name, processed_result, svg_doc))


                label = processed_result[0][0]
                timestamp = time.strftime('%Y-%m-%d_%H.%M.%S')
                # print(timestamp)
                # print('\n')

                if args.gather_data and cropped_image:
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

                # if svg_doc:
                #     ## Plot points out
                #     ## 160 x 80 grid
                #     ## 16px width
                #     ## 20, 40, 60 for 0, 1, 2
                #     lines = message.split('\n')
                #     y_correction = len(lines) * 20
                #     for line in lines:
                #         svg_doc.add(svg.Text(line,
                #         x=(LOCATIONS[location_name][0]) * SVG_SCALE_FACTOR,
                #         y=(LOCATIONS[location_name][1] - y_correction) * SVG_SCALE_FACTOR,
                #         fill='white', font_size=20))

                #         y_correction = y_correction - 20

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
