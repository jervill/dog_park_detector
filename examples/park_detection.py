#!/usr/bin/env python3
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
"""Object detection library demo.

 - Takes an input image and tries to detect person, dog, or cat.
 - Draws bounding boxes around detected objects.
 - Saves an image with bounding boxes around detected objects.
"""
import argparse

from PIL import Image, ImageDraw

from aiy.vision.inference import ImageInference
from aiy.vision.models import park_detection_model

# Locations of interesting locations:
court_one_dimensions = (0,240,170,400)
court_two_dimensions = (188, 240, 408,400)
dog_park_dimensions = (559, 228, 820,388)

locations = {
    'court_one': court_one_dimensions,
    'court_two': court_two_dimensions,
    'dog_park': dog_park_dimensions,
}

def crop_object(image, dimensions=locations['dog_park']):
    return image.crop(dimensions)

def process(result, labels=['high activity', 'low activity', 'none'], threshold=0.1, top_k=3):
    """Processes inference result and returns labels sorted by confidence."""
    # MobileNet based classification model returns one result vector.
    assert len(result.tensors) == 1
    tensor = result.tensors['final_result']
    probs, shape = tensor.data, tensor.shape
    assert shape.depth == len(labels)
    pairs = [pair for pair in enumerate(probs) if pair[1] > threshold]
    pairs = sorted(pairs, key=lambda pair: pair[1], reverse=True)
    pairs = pairs[0:top_k]
    return [' %s (%.2f)' % (labels[index], prob) for index, prob in pairs]

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', '-i', dest='input', required=True,
                        help='Input image file.')
    parser.add_argument('--threshold', '-t', type=float, default=0.3,
                        help='Detection probability threshold.')
    args = parser.parse_args()

    with ImageInference(park_detection_model.model()) as inference:
        image = Image.open(args.input)
        image_center = crop_object(image, locations['dog_park'])

        result = inference.run(image_center)
        processed_result = process(result)
        print(processed_result)

        # for i, obj in enumerate(objects):
        #     print('Object #%d: %s' % (i, obj))


if __name__ == '__main__':
    main()
