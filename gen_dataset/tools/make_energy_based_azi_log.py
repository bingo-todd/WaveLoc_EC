import numpy as np
from BasicTools import wav_tools
from BasicTools.parse_file import file2dict
from BasicTools.get_file_path import get_realpath, get_file_path
from LocTools.add_log import add_log


# load source file log
tar_src_path_dict = {}
inter_src_path_dict = {}
# for set_type in ['train', 'valid', 'test']:
for set_type in ['train', 'valid', 'test']:
    tmp_dict = file2dict('../../Data/synroom/tar/'
                         + f'{set_type}/reverb/src_path.txt')
    tar_src_path_dict.update(tmp_dict)

    for snr in [0, 10, 20]:
        tmp_dict = file2dict('../../Data/synroom/inter/'
                             + f'{set_type}/{snr}/reverb/src_path.txt')
        inter_src_path_dict.update(tmp_dict)


def make_azi_energy_besed_log(inter_reverb_dir):
    frame_len = 320
    frame_shift = 160

    n_room_per_azi = 24
    snr_theta = 0

    azi_logger = open(f'{inter_reverb_dir}/energy_based_azi.txt', 'x')
    azi_logger.write(f'# frame_len:{frame_len}  frame_shift: {frame_shift}')

    inter_reverb_paths = get_file_path(inter_reverb_dir, suffix='.wav',
                                       is_absolute=True)
    for inter_reverb_path in inter_reverb_paths:
        # interference
        [*basedir, _, set_type,
         snr, _, inter_reverb_name] = inter_reverb_path.split('/')
        inter_azi_i, room_i, wav_i = [int(item) for item in
                                      inter_reverb_name[:-4].split('_')]

        # target
        if room_i >= 888:
            tar_azi_i = np.int(np.floor((room_i-888)/n_room_per_azi))
        else:
            tar_azi_i = np.int(np.floor(room_i/n_room_per_azi))
        tar_reverb_name = f'{tar_azi_i:0>2d}_{room_i:0>4d}_{wav_i}.wav'
        tar_reverb_path = '/'.join([*basedir, 'tar', set_type, 'reverb',
                                    tar_reverb_name])

        inter_src_path = inter_src_path_dict[get_realpath(inter_reverb_path)]
        inter_src, fs = wav_tools.read_wav(inter_src_path)

        tar_src_path = tar_src_path_dict[get_realpath(tar_reverb_path)]
        tar_src, fs = wav_tools.read_wav(tar_src_path)

        mix_len = min(inter_src.shape[0], tar_src.shape[0])
        snr_frame_all = wav_tools.cal_snr(tar_src[:mix_len],
                                          inter_src[:mix_len],
                                          frame_len=frame_len,
                                          frame_shift=frame_shift)
        n_sample = snr_frame_all.shape[0]
        energy_based_azi = np.zeros((n_sample, 1))
        energy_based_azi[snr_frame_all >= snr_theta] = tar_azi_i
        energy_based_azi[snr_frame_all < snr_theta] = inter_azi_i

        add_log(azi_logger, inter_reverb_path, energy_based_azi)
    azi_logger.close()


if __name__ == '__main__':
    import sys
    inter_reverb_dir = sys.argv[1]
    make_azi_energy_besed_log(inter_reverb_dir)
