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

# src paths
TIMIT_dir = f'{Work_Space}/Data/Speech/TIMIT_wav/TIMIT/TEST'
src_paths = get_file_path(TIMIT_dir, suffix='.wav', is_absolute=True)
src_path_iterator = Iterator(src_paths, shuffle=True)


def get_src_path():
    src_path = src_path_iterator.next()
    if src_path is None:
        src_path_iterator.reset()
        src_path = src_path_iterator.next()
    return src_path


def load_brir(room, azi):
    """load brirs of given room
    Args:
    """
    brirs_path = os.path.join(brirs_dir, f'BRIRs_{room}.npy')
    brirs = np.load(brirs_path)
    return brirs[azi]


def syn_record(room, src1_path, azi_src1, record1_path,
               src2_path, azi_src2, record2_path,
               src3_path, azi_src3, record3_path):
    if os.path.exists(record1_path) or os.path.exists(record2_path):
        return False
    os.makedirs(os.path.dirname(record1_path), exist_ok=True)
    os.makedirs(os.path.dirname(record2_path), exist_ok=True)

    if src1_path is None or src2_path is None:
        raise Exception()

    if src1_path == src2_path:
        return False

    src1, fs = wav_tools.read(src1_path)
    src2, fs = wav_tools.read(src2_path)
    src2 = wav_tools.set_snr(src2, ref=src1, snr=0)
    src3, fs = wav_tools.read(src3_path)
    src3 = wav_tools.set_snr(src3, ref=src1, snr=0)

    gpu_filter = GPU_Filter(0)
    brir_src1 = load_brir(room, azi_src1)
    record1 = gpu_filter.brir_filter(src1, brir_src1)
    wav_tools.write(record1, fs, record1_path, n_bit=24)

    brir_src2 = load_brir(room, azi_src2)
    record2 = gpu_filter.brir_filter(src2, brir_src2)
    wav_tools.write(record2, fs, record2_path, n_bit=24)

    brir_src3 = load_brir(room, azi_src3)
    record3 = gpu_filter.brir_filter(src3, brir_src3)
    wav_tools.write(record3, fs, record3_path, n_bit=24)


def gen_set(set_dir):

    set_dir = os.path.realpath(set_dir)
    os.makedirs(set_dir, exist_ok=True)

    n_azi = 37
    n_wav_per_cond = 10
    azi_diff_all = [1, 2, 4, 6, 8]

    src_path_logger = open(f'{set_dir}/src_path.txt', 'w')
    tasks = []
    for room in rooms:
        for azi_src2 in range(n_azi):
            for azi_diff_1_2 in azi_diff_all:
                azi_src1 = azi_src2 - azi_diff_1_2
                if azi_src1 < 0:
                    continue
                for azi_diff_2_3 in azi_diff_all:
                    azi_src3 = azi_src2 + azi_diff_2_3
                    if azi_src3 >= n_azi:
                        continue
                    for wav_i in range(n_wav_per_cond):
                        src1_path = get_src_path()
                        record1_path = (f'{set_dir}/{room}/reverb/'
                                        + '_'.join((f'{azi_src1:0>2d}',
                                                    f'{azi_src2:0>2d}',
                                                    f'{azi_src3:0>2d}',
                                                    f'{wav_i}', 'src1'))
                                        + '.wav')
                        src2_path = get_src_path()
                        while src2_path == src1_path:
                            src2_path = get_src_path()
                        record2_path = (f'{set_dir}/{room}/reverb/'
                                        + '_'.join((f'{azi_src1:0>2d}',
                                                    f'{azi_src2:0>2d}',
                                                    f'{azi_src3:0>2d}',
                                                    f'{wav_i}', 'src2'))
                                        + '.wav')
                        src3_path = get_src_path()
                        while src3_path == src1_path or src3_path == src2_path:
                            src3_path = get_src_path()
                        record3_path = (f'{set_dir}/{room}/reverb/'
                                        + '_'.join((f'{azi_src1:0>2d}',
                                                    f'{azi_src2:0>2d}',
                                                    f'{azi_src3:0>2d}',
                                                    f'{wav_i}', 'src3'))
                                        + '.wav')
                        tasks.append([room, src1_path, azi_src1, record1_path,
                                      src2_path, azi_src2, record2_path,
                                      src3_path, azi_src3, record3_path])
                        src_path_logger.write(f'{record1_path}: {src1_path}\n')
                        src_path_logger.write(f'{record2_path}: {src2_path}\n')
                        src_path_logger.write(f'{record3_path}: {src3_path}\n')
    easy_parallel(func=syn_record, tasks=tasks, n_worker=8, show_progress=True)


if __name__ == '__main__':
    gen_set('../Data/test-3src')

    for room in rooms:
        make_record_path_log(f'../Data/test-1src/{room}/reverb')
