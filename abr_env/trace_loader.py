import os
import wget
import numpy as np

import abr_env


def load_chunk_sizes():
    # bytes of video chunk file at different bitrates

    # source video: "Envivio-Dash3" video H.264/MPEG-4 codec
    # at bitrates in {300,750,1200,1850,2850,4300} kbps

    # original video file:
    # https://github.com/hongzimao/pensieve/tree/master/video_server

    # download video size folder if not existed
    video_folder = abr_env.__path__[0] + '/videos/'
    os.makedirs(video_folder, exist_ok=True)
    if not os.path.exists(video_folder + 'video_sizes.npy'):
        wget.download(
            'https://www.dropbox.com/s/hg8k8qq366y3u0d/video_sizes.npy?dl=1',
            out=video_folder + 'video_sizes.npy')

    chunk_sizes = np.load(video_folder + 'video_sizes.npy')

    return chunk_sizes