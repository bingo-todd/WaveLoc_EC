import os
import numpy as np
from BasicTools import wav_tools


def file_reader(record1_path, is_slice=True, return_label=True):
    frame_len = 320*2
    shift_len = 160
    n_azi = 37

    # record signal
    record1, fs = wav_tools.read(record1_path)

    record2_path = record1_path.replace('src1', 'src2')
    record2, fs = wav_tools.read(record2_path)

    min_len = np.min((record1.shape[0], record2.shape[0]))
    mix = record1[:min_len]+record2[:min_len]

    x = np.expand_dims(
        wav_tools.frame_data(mix, frame_len, shift_len),
        axis=-1)

    if return_label:
        # onehot azi label
        record_path = os.path.expanduser(record1_path)
        *_, record_name = record_path.split('/')
        azi = np.int(record_name[:-4].split('_')[0])
        n_sample = x.shape[0]
        y = np.zeros((n_sample, n_azi), dtype=np.float32)
        y[:, azi] = 1
        return x, y
    else:
        return x
