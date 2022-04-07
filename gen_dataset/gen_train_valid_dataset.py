import numpy as np
import os
import logging
import time
import pickle
# import matplotlib.pyplot as plt

from BasicTools import wav_tools
from BasicTools.ProcessBar import ProcessBar  # noqa:E402
from BasicTools.GPU_Filter import GPU_Filter  # noqa:E402
from BasicTools.get_file_path import get_file_path  # noqa:E402
from BasicTools.easy_parallel import easy_parallel  # noqa:E402

from make_record_path_log import make_record_path_log

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

TIMIT_dir = os.path.expanduser('~/Work_Space/Data/TIMIT_wav')

n_azi = 37  # -90~90 in step of 5
n_room_per_azi = 24
n_wav_per_cond_all = {'train': 4, 'valid': 1, 'test': 0}

fs = 16e3
frame_len = int(20e-3*fs)
shift_len = int(10e-3*fs)


def _syn_record(src_path, brir_path, record_path, gpu_filter):

    n_run = 0
    while not os.path.exists(brir_path):
        time.sleep(10)
        n_run = n_run + 1
        print(f'{record_path} {n_run}')

    src, fs = wav_tools.read_wav(src_path)
    brir, fs = wav_tools.read_wav(brir_path)
    record = gpu_filter.brir_filter(src, brir)
    wav_tools.write(record, fs, record_path, n_bit=24)


def load_src_paths(set_dir):
    src_paths_path = f'{set_dir}/src_paths.pkl'
    if not os.path.exists(src_paths_path):
        print('find all TIMIT wav')
        n_wav_train = n_azi * n_room_per_azi * n_wav_per_cond_all['train']
        n_wav_valid = n_azi * n_room_per_azi * n_wav_per_cond_all['valid']
        n_wav_test = n_azi * n_room_per_azi * n_wav_per_cond_all['test']

        # train and validate
        TIMIT_train_dir = f'{TIMIT_dir}/TIMIT/TRAIN'
        src_paths = get_file_path(TIMIT_train_dir, suffix='.wav',
                                  is_absolute=True)
        n_src_path = len(src_paths)
        print('train & valid: n_src'
              + f'{n_src_path} n_src_need {n_wav_train+n_wav_valid}')
        if n_wav_train + n_wav_valid > n_src_path:
            src_paths = np.concatenate((
                src_paths,
                np.random.choice(src_paths,
                                 n_wav_train+n_wav_valid-n_src_path)))
        np.random.shuffle(src_paths)
        train_src_paths = src_paths[:n_wav_train]
        valid_src_paths = src_paths[n_wav_train:]

        # test
        TIMIT_test_dir = os.path.join(TIMIT_dir, 'TIMIT/TEST')
        src_paths = get_file_path(TIMIT_test_dir, suffix='.wav',
                                  is_absolute=True)
        np.random.shuffle(src_paths)
        n_src_path = len(src_paths)
        print(f'test: n_src {n_src_path} n_src_need {n_wav_test}')
        if n_wav_test > n_src_path:
            test_src_paths = np.concatenate((
                src_paths,
                np.random.choice(src_paths, n_wav_test-n_src_path)))
        else:
            test_src_paths = src_paths

        os.makedirs(set_dir)
        with open(src_paths_path, 'wb') as file_obj:
            pickle.dump(
                [train_src_paths, valid_src_paths, test_src_paths],
                file_obj)
    else:
        print(f'load from {src_paths_path}')

    with open(src_paths_path, 'rb') as file_obj:
        [train_src_paths,
         valid_src_paths,
         test_src_paths] = pickle.load(file_obj)
    return [train_src_paths, valid_src_paths, test_src_paths]


def syn_record(src_paths, dataset_dir, n_wav_per_cond, n_room_per_azi,
               room_count_init, brir_dir, gpu_id):

    # shuffle src path
    n_wav_need = n_wav_per_cond*n_room_per_azi*n_azi
    n_repeat = np.int(np.ceil(n_wav_need/len(src_paths)))
    if n_repeat > 0:
        src_paths = src_paths * n_repeat
    np.random.shuffle(src_paths)
    src_paths = src_paths[:n_wav_need]

    direct_dir = f'{dataset_dir}/direct'
    os.makedirs(direct_dir, exist_ok=True)
    direct_src_path_file = open(f'{direct_dir}/src_path.txt', mode='x')

    reverb_dir = f'{dataset_dir}/reverb'
    os.makedirs(reverb_dir, exist_ok=True)
    reverb_src_path_file = open(f'{reverb_dir}/src_path.txt', mode='x')

    gpu_filter = GPU_Filter(gpu_id=gpu_id)
    pb = ProcessBar(n_azi*n_room_per_azi*n_wav_per_cond)
    wav_count = 0
    room_count = room_count_init-1
    for azi_i in range(n_azi):
        for room_i in range(n_room_per_azi):
            room_count = room_count + 1
            for i in range(n_wav_per_cond):
                pb.update()
                src_path = src_paths[wav_count]
                wav_count = wav_count+1

                brir_name = f'{azi_i:0>2d}_{room_count:0>4d}'
                record_name = f'{brir_name}_{i}'
                direct_brir_path = f'{brir_dir}/{brir_name}_direct.wav'
                direct_record_path = f'{direct_dir}/{record_name}.wav'

                reverb_brir_path = f'{brir_dir}/{brir_name}.wav'
                reverb_record_path = f'{reverb_dir}/{record_name}.wav'
                if (os.path.exists(direct_brir_path)
                        and os.path.exists(reverb_record_path)):
                    continue

                _syn_record(src_path, direct_brir_path, direct_record_path,
                            gpu_filter)
                direct_src_path_file.write(
                    ': '.join([
                        os.path.realpath(direct_record_path),
                        os.path.realpath(src_path)]))
                direct_src_path_file.write('\n')

                _syn_record(src_path, reverb_brir_path, reverb_record_path,
                            gpu_filter)
                reverb_src_path_file.write(
                    ': '.join([
                        os.path.realpath(reverb_record_path),
                        os.path.realpath(src_path)]))
                reverb_src_path_file.write('\n')

    direct_src_path_file.close()
    reverb_src_path_file.close()


def main():
    brir_dir = '../syn_BRIRs/BRIRs'
    dataset_basedir = '../Data'
    gpu_id = 0  # performance convolution using GPU

    [train_src_paths,
     valid_src_paths,
     test_src_paths] = load_src_paths(dataset_basedir)

    # train set
    room_count_init = 0
    syn_record(src_paths=train_src_paths,
               dataset_dir=f'{dataset_basedir}/train',
               n_wav_per_cond=n_wav_per_cond_all['train'],
               n_room_per_azi=n_room_per_azi,
               room_count_init=room_count_init,
               brir_dir=brir_dir,
               gpu_id=gpu_id)
    make_record_path_log(f'{dataset_basedir}/train/reverb')

    # valid set
    room_count_init = 0
    syn_record(src_paths=train_src_paths,
               dataset_dir=f'{dataset_basedir}/valid',
               n_wav_per_cond=n_wav_per_cond_all['valid'],
               n_room_per_azi=n_room_per_azi,
               room_count_init=room_count_init,
               brir_dir=brir_dir,
               gpu_id=gpu_id)
    make_record_path_log(f'{dataset_basedir}/valid/reverb')


if __name__ == '__main__':
    main()
