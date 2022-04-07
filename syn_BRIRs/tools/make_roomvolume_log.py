import configparser
import os
import sys
from BasicTools.get_file_path import get_file_path


def main():
    brir_dir = sys.argv[1]
    volume_path = f'{brir_dir}/volume.txt'
    logger = open(volume_path, 'w')
    config = configparser.ConfigParser()
    brir_paths = get_file_path(brir_dir, suffix='.wav', is_absolute=True)

    for brir_path in brir_paths:
        if brir_path.find('direct') > -1:
            continue
        cfg_path = f'{brir_path[:-4]}.cfg'
        config.read(os.path.expanduser(cfg_path))
        room_size = list(map(float, config['Room']['size'].split(',')))
        volume = room_size[0] * room_size[1] * room_size[2]
        logger.write(f'{brir_path}: {volume:.2f}\n')
    logger.close()


if __name__ == '__main__':
    main()
