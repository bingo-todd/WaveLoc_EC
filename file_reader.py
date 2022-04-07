import os
import numpy as np
from BasicTools import wav_tools


def file_reader(reverb_record_path, is_slice=True, return_label=True):
    frame_len = 320*2
    shift_len = 160
    n_azi = 37

    # record signal
    reverb_record, fs = wav_tools.read(reverb_record_path)
    wav_r = np.expand_dims(
                wav_tools.frame_data(reverb_record, frame_len, shift_len),
                axis=-1)

    if return_label:
        # onehot azi label
        reverb_record_path = os.path.expanduser(reverb_record_path)
        *_, test_i, set_type, _, reverb_record_name = \
            reverb_record_path.split('/')
        azi_i = np.int(reverb_record_name[:-4].split('_')[0])
        n_sample = wav_r.shape[0]
        loc_label = np.zeros((n_sample, n_azi), dtype=np.float32)
        loc_label[:, azi_i] = 1
        return wav_r, loc_label
    else:
        return wav_r
