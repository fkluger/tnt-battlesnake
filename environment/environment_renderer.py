import logging
import os

import matplotlib
import numpy as np

LOGGER = logging.getLogger('EnvironmentRenderer')

try:
    if os.environ['DISPLAY']:
        LOGGER.debug('Found display. Using ffmpeg backend.')
except KeyError:
    LOGGER.debug('Did not find display. Using Agg backend.')
    matplotlib.use('Agg')


import matplotlib.pyplot as plt
import matplotlib.animation as animation


class EnvironmentRenderer:

    current_frames = list()
    episodes = list()

    def __init__(self, output_directory):
        self.output_directory = output_directory
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
            LOGGER.info('Created output directory.')

    def add_frame(self, frame):
        self.current_frames.append(np.sum(frame, -1))

    def on_reset(self):
        self.episodes.append(np.copy(self.current_frames))
        self.current_frames.clear()

    def render(self, filename):
        ims = []
        fig = plt.figure()
        for frame in self.episodes[-1]:
            if frame is not None:
                im = plt.imshow(frame, animated=True)
                ims.append([im])
        ani = animation.ArtistAnimation(fig, ims, interval=100, repeat=False, blit=True)
        output_file = f'{self.output_directory}/{filename}'
        ani.save(output_file)
        plt.close()
        LOGGER.info(f'Rendered current episode to {output_file}.')
