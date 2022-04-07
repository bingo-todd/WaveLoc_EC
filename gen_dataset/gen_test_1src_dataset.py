import os
import numpy as np

from BasicTools import wav_tools
from BasicTools.easy_parallel import easy_parallel
from BasicTools.GPU_Filter import GPU_Filter
from BasicTools.Iterator import Iterator
from BasicTools.get_file_path import get_file_path

from make_record_path_log2 import make_record_path_log


Work_Space = os.getenv('Work_Space')
brirs_dir = f'{Work_Space}/Data/BRIRs/BRIRs_Surrey/16kHz/align_to_anechoic'
rooms = ['Anechoic', 'Room_A', 'Room_B', 'Room_C', 'Room_D']

TIMIT_dir = f'{Work_Space}/Data/Speech/TIMIT_wav/TIMIT/TEST'


def load_brir(room, azi):
    """load brirs of given room
    Args:
    """
    brirs_path = os.path.join(brirs_dir, f'BRIRs_{room}.npy')
    brirs = np.load(brirs_path)
    return brirs[azi]


def syn_record(src_path, room, azi_i, reverb_path):
    if os.path.exists(reverb_path):
        return
    os.makedirs(os.path.dirname(reverb_path), exist_ok=True)
    src, fs = wav_tools.read(src_path)
    brir = load_brir(room, azi_i)
    gpu_filter = GPU_Filter(0)
    reverb = gpu_filter.brir_filter(src, brir)
    wav_tools.write(reverb, fs, reverb_path, n_bit=24)


def gen_set(set_dir):

    set_dir = os.path.realpath(set_dir)
    os.makedirs(set_dir, exist_ok=True)

    n_azi = 37
    n_wav_per_cond = 10

    # src paths
    src_paths = get_file_path(TIMIT_dir, suffix='.wav', is_absolute=True)

    src_path_logger = open(f'{set_dir}/src_path.txt', 'w')
    tasks = []
    for room in rooms:
        src_path_iterator = Iterator(src_paths, shuffle=True)
        for azi_i in range(n_azi):
            for wav_i in range(n_wav_per_cond):
                src_path = src_path_iterator.next()
                reverb_path = \
                    f'{set_dir}/{room}/reverb/{azi_i:0>2d}_{wav_i}_src1.wav'
                tasks.append([src_path, room, azi_i, reverb_path])
                src_path_logger.write(f'{reverb_path}: {src_path}\n')
                # syn_record(src_path, room, azi_i, reverb_path)
    easy_parallel(func=syn_record, tasks=tasks, n_worker=6,
                  show_progress=True)


if __name__ == '__main__':
    gen_set('../Data/test-1src')

    for room in rooms:
        make_record_path_log(f'../Data/test-1src/{room}/reverb')
