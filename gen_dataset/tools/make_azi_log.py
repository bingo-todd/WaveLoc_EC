import sys
import os

from BasicTools.get_file_path import get_file_path, get_realpath
from LocTools.add_log import add_log


def main(record_dir):
    record_paths = get_file_path(record_dir, suffix='.wav', is_absolute=True)
    azi_logger = open(f'{record_dir}/azi.txt', 'x')
    for record_path in record_paths:
        record_path_realpath = get_realpath(record_path,
                                            root_dir='~/Work_Space')
        record_name = os.path.basename(record_path)[:-4]
        azi, *_ = [int(item) for item in record_name.split('_')]
        add_log(azi_logger, record_path_realpath, [[azi]])
    azi_logger.close()


if __name__ == '__main__':
    record_dir = sys.argv[1]
    main(record_dir)
