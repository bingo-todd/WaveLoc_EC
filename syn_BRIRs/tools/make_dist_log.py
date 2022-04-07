import configparser
import numpy as np
import os
import sys
from BasicTools.get_file_path import get_file_path


def main():
    brir_dir = sys.argv[1]
    log_path = f'{brir_dir}/dist.txt'
    logger = open(log_path, 'w')
    config = configparser.ConfigParser()
    brir_paths = get_file_path(brir_dir, suffix='.wav', is_absolute=True)
    for brir_path in brir_paths:
        if brir_path.find('direct') > -1:
            continue
        cfg_path = f'{brir_path[:-4]}.cfg'
        config.read(os.path.expanduser(cfg_path))
        source_pos = np.asarray(
            list(
                map(float, config['Source']['pos'].split(','))))
        receiver_pos = np.asarray(
            list(
                map(float, config['Receiver']['pos'].split(','))))
        dist = np.sqrt(np.sum((source_pos-receiver_pos)**2))
        logger.write(f'{brir_path}: {dist:.2f}\n')
    logger.close()


if __name__ == '__main__':
    main()
