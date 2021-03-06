import sys

from BasicTools.get_file_path import get_file_path


def make_record_path_log(wav_dir):
    record_paths = get_file_path(wav_dir, suffix='.wav', is_absolute=True)
    with open(f'{wav_dir}/record_paths.txt', 'w') as f:
        for record_path in record_paths:
            f.write(f'{record_path}\n')


if __name__ == '__main__':
    wav_dir = sys.argv[1]
    make_record_path_log(wav_dir)
