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

from io import BytesIO
from PIL import Image

from picamera import PiCamera, Color

from aiy.vision import inference
from aiy.vision.models import utils

# Make this a csv
long_term_data_over_time = {
    'time': [],
    'dog park': {
        'high activity': [],
        'low activity': [],
        'no activity': [],
    }
}

def commit_data_to_long_term(interval, short_term_data={}):
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

    short_term_data = reset_data()

    while True:
        yield short_term_data

        elapsed_time = short_term_data['time'][-1] - short_term_data['time'][0]

        # If we've hit the time interval, record to long term.
        if elapsed_time > interval:
            long_term_data_over_time['time'].append(short_term_data['time'][-1])

            for k, v in short_term_data['dog park'].items():
                average = get_average(v)
                long_term_data_over_time['dog park'][k].append(average)

            short_term_data = reset_data()
            print(json.dumps(long_term_data_over_time))


def handle_data(data, result):
    data['time'].append(int(time.time()))

    for label, prob in result:
        data['dog park'][label].append(prob)

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


def get_cropped_image(camera):
    # Locations of interesting locations:
    court_one_dimensions = (0, 195, 210, 445)
    court_two_dimensions = (148, 195, 448, 445)
    dog_park_dimensions = (520, 183, 820, 433)

    locations = {
        'court_one': court_one_dimensions,
        'court_two': court_two_dimensions,
        'dog_park': dog_park_dimensions,
    }

    while True:
        stream = BytesIO()
        # Take picture
        camera.capture(stream, format='jpeg', use_video_port=True)
        # "Rewind" the stream to the beginning so we can read its content
        stream.seek(0)
        image = Image.open(stream)

        # Crop picture and return it
        yield image.crop(locations['dog_park'])

def _make_filename(image_folder, timestamp, label):
    path = '%s/Dog/%s/%s.%s'
    return os.path.expanduser(path % (image_folder, label, timestamp, 'jpeg'))


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

    with PiCamera(sensor_mode=4, resolution=(820, 616), framerate=30) as camera:
        if args.preview:
            camera.start_preview()

        data_generator = commit_data_to_long_term(args.time_interval)
        data = data_generator.send(None)

        with inference.ImageInference(model) as image_inference:
            # Constantly get cropped images
            for cropped_image in get_cropped_image(camera):
                # then run image_inference on them.
                result = image_inference.run(cropped_image)
                processed_result = process(result, labels, args.output_layer)
                handle_data(data, processed_result)
                data = data_generator.send(data)
                message = get_message(processed_result)

                # Print the message
                print(message)

                timestamp = time.strftime('%Y-%m-%d_%H.%M.%S')
                print(timestamp + '\n')

                if args.gather_data:
                    # Gather all data on 'no activity' since it's biased against that.
                    # Gather 1% of all others.
                    if(
                        (processed_result[0][0] == 'no activity') or
                        (random.random() > 0.99)
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
