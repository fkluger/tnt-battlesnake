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


import matplotlib.animation as animation
import matplotlib.pyplot as plt


class EnvironmentRenderer:

    def __init__(self, output_directory):
        self.current_frames = list()
        self.last_episode = None
        self.output_directory = output_directory
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
            LOGGER.info('Created output directory.')

    def add_frame(self, frame):
        self.current_frames.append(frame)

    def on_reset(self):
        self.last_episode = np.copy(self.current_frames)
        self.current_frames.clear()

    def render(self, filename):
        if self.last_episode is None:
            return
        fig, (ax1) = plt.subplots(1, 1)
        ax1.axis('off')

        frame = self.last_episode[0]
        img1 = ax1.imshow(frame[:, :, 0], animated=True)

        def update(current_frame):
            frame = self.last_episode[current_frame]
            if frame is not None:
                img1.set_data(frame[:, :, 0])
            return [img1]

        ani = animation.FuncAnimation(fig, update, frames=len(self.last_episode), interval=100, repeat=False)
        output_file = f'{self.output_directory}/{filename}'
        plt.close()
        ani.save(output_file)
        LOGGER.info(f'Rendered current episode to {output_file}.')
