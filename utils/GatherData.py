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
"""Joy detection demo."""
import argparse
import collections
import contextlib
import io
import logging
import math
import os
import queue
import signal
import sys
import threading
import time

from PIL import Image, ImageDraw, ImageFont
from picamera import PiCamera

from aiy.board import Board
from aiy.leds import Color, Leds, Pattern, PrivacyLed
from aiy.toneplayer import TonePlayer
from aiy.vision.inference import CameraInference
from aiy.vision.models import face_detection
from aiy.vision.streaming.server import StreamingServer
from aiy.vision.streaming import svg

logger = logging.getLogger(__name__)

FONT_FILE = '/usr/share/fonts/truetype/freefont/FreeSans.ttf'

@contextlib.contextmanager
def stopwatch(message):
    try:
        logger.info('%s...', message)
        begin = time.monotonic()
        yield
    finally:
        end = time.monotonic()
        logger.info('%s done. (%fs)', message, end - begin)

class Service:

    def __init__(self):
        self._requests = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            request = self._requests.get()
            if request is None:
                self.shutdown()
                break
            self.process(request)
            self._requests.task_done()

    def process(self, request):
        pass

    def shutdown(self):
        pass

    def submit(self, request):
        self._requests.put(request)

    def close(self):
        self._requests.put(None)
        self._thread.join()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.close()


class Photographer(Service):
    """Saves photographs to disk."""

    def __init__(self, format, folder):
        super().__init__()
        assert format in ('jpeg', 'bmp', 'png')

        self._font = ImageFont.truetype(FONT_FILE, size=25)
        self._faces = ([], (0, 0))
        self._format = format
        self._folder = folder

    def _make_filename(self, timestamp, annotated):
        path = '%s/%s_annotated.%s' if annotated else '%s/%s.%s'
        return os.path.expanduser(path % (self._folder, timestamp, self._format))

    def process(self, message):
        if isinstance(message, tuple):
            self._faces = message
            return

        camera = message
        timestamp = time.strftime('%Y-%m-%d_%H.%M.%S')

        stream = io.BytesIO()
        with stopwatch('Taking photo'):
            camera.capture(stream, format=self._format, use_video_port=True)

        filename = self._make_filename(timestamp, annotated=False)
        with stopwatch('Saving original %s' % filename):
            stream.seek(0)
            with open(filename, 'wb') as file:
                file.write(stream.read())

    def update_faces(self, faces):
        self.submit(faces)

    def shoot(self, camera):
        self.submit(camera)


def gather_data(interval, image_format, image_folder,
                 enable_streaming, streaming_bitrate, mdns_name):
    done = threading.Event()
    def stop():
        logger.info('Stopping...')
        done.set()

    signal.signal(signal.SIGINT, lambda signum, frame: stop())
    signal.signal(signal.SIGTERM, lambda signum, frame: stop())

    logger.info('Starting...')

    with contextlib.ExitStack() as stack:
        photographer = stack.enter_context(Photographer(image_format, image_folder))

        # Forced sensor mode, 1640x1232, full FoV. See:
        # https://picamera.readthedocs.io/en/release-1.13/fov.html#sensor-modes
        # This is the resolution inference run on.
        # Use half of that for video streaming (820x616).
        camera = stack.enter_context(PiCamera(sensor_mode=4, resolution=(820, 616)))

        server = None
        if enable_streaming:
            server = stack.enter_context(StreamingServer(camera, bitrate=streaming_bitrate,
                                                         mdns_name=mdns_name))

        def take_photo():
            logger.info('Taking picture.')
            photographer.shoot(camera)

        while True:
            take_photo()
            time.sleep(interval)

            if done.is_set():
                break


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--image_format', default='jpeg',
                        choices=('jpeg', 'bmp', 'png'),
                        help='Format of captured images')
    parser.add_argument('--image_folder', default='~/Pictures/Data',
                        help='Folder to save captured images')
    parser.add_argument('--blink_on_error', default=False, action='store_true',
                        help='Blink red if error occurred')
    parser.add_argument('--enable_streaming', default=False, action='store_true',
                        help='Enable streaming server')
    parser.add_argument('--streaming_bitrate', type=int, default=1000000,
                        help='Streaming server video bitrate (kbps)')
    parser.add_argument('--mdns_name', default='',
                        help='Streaming server mDNS name')
    args = parser.parse_args()

    try:
        gather_data(5, args.image_format, args.image_folder,
                     args.enable_streaming, args.streaming_bitrate, args.mdns_name)
        
    except KeyboardInterrupt:
        sys.exit()
    except Exception:
        logger.exception('Exception while running joy demo.')
        if args.blink_on_error:
            with Leds() as leds:
                leds.pattern = Pattern.blink(100)  # 10 Hz
                leds.update(Leds.rgb_pattern(Color.RED))
                time.sleep(1.0)

    return 0

if __name__ == '__main__':
    sys.exit(main())
