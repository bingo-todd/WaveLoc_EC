import os
import sys

from BasicTools.get_file_path import get_file_path


def make_brir_list(record_dir, brir_dir):
    reverb_record_dir = f'{record_dir}/reverb'
    reverb_brir_path_logger = open(f'{reverb_record_dir}/brir.txt', 'w')

    direct_record_dir = f'{record_dir}/direct'
    direct_brir_path_logger = open(f'{direct_record_dir}/brir.txt', 'w')

    reverb_record_paths = get_file_path(reverb_record_dir, suffix='.wav',
                                        is_absolute=True)
    for reverb_record_path in reverb_record_paths:
        reverb_record_name = os.path.basename(reverb_record_path)[:-4]
        azi_i, room_i, *_ = [int(item)
                             for item in reverb_record_name.split('_')]
        reverb_brir_path = f'{brir_dir}/{azi_i:0>2d}_{room_i:0>4d}.wav'
        reverb_brir_realpath = os.path.realpath(reverb_brir_path)

        if not os.path.exists(os.path.expanduser(reverb_brir_realpath)):
            raise Exception()

        reverb_record_realpath = os.path.realpath(reverb_record_path)
        reverb_brir_path_logger.write('{}: {}\n'.format(
            reverb_record_realpath, reverb_brir_realpath))

        direct_brir_path = f'{brir_dir}/{azi_i:0>2d}_{room_i:0>4d}_direct.wav'
        direct_brir_realpath = os.path.realpath(direct_brir_path)
        direct_record_realpath = os.path.realpath(
            f'{direct_record_dir}/{reverb_record_name}.wav')
        direct_brir_path_logger.write('{}: {}\n'.format(
            direct_record_realpath, direct_brir_realpath))

    reverb_brir_path_logger.close()
    direct_brir_path_logger.close()


if __name__ == '__main__':
    record_dir, brir_dir = sys.argv[1:3]
    make_brir_list(record_dir, brir_dir)
