import numpy as np
from BasicTools.get_file_path import get_file_path, get_realpath
from LocTools.add_log import add_log


def make_azi_diff_log(inter_reverb_dir):
    azi_diff_logger = open(f'{inter_reverb_dir}/azi_diff.txt', 'x')

    n_room_per_azi = 24
    inter_reverb_paths = get_file_path(inter_reverb_dir, suffix='.wav',
                                       is_absolute=True)
    for inter_reverb_path in inter_reverb_paths:
        [*basedir, _, set_type,
         snr, _, inter_reverb_name] = inter_reverb_path.split('/')
        inter_azi_i, room_i, wav_i = [int(item) for item in
                                      inter_reverb_name[:-4].split('_')]
        # target
        if room_i >= 888:
            tar_azi_i = np.int(np.floor((room_i-888)/n_room_per_azi))
        else:
            tar_azi_i = np.int(np.floor(room_i/n_room_per_azi))
        inter_reverb_realpath = get_realpath(inter_reverb_path,
                                             root_dir='~/Work_Space')
        azi_i_diff = np.abs(tar_azi_i - inter_azi_i)
        add_log(azi_diff_logger, inter_reverb_realpath, [[azi_i_diff]])
    azi_diff_logger.close()


if __name__ == '__main__':
    import sys
    inter_reverb_dir = sys.argv[1]
    make_azi_diff_log(inter_reverb_dir)
