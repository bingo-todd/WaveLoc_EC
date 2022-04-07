import configparser
import os
import sys
from BasicTools.get_file_path import get_file_path


def main(brir_dir):
    rt_path = f'{brir_dir}/RT.txt'
    rt_logger = open(rt_path, 'w')
    config = configparser.ConfigParser()
    config_paths = get_file_path(brir_dir,
                                 suffix='.cfg',
                                 is_absolute=True)
    print(f'n_config: {len(config_paths)}')
    for config_path in config_paths:
        config.read(os.path.expanduser(config_path))
        rt = float(config['Room']['RT60'])
        brir_path = config_path[:-4]+'.wav'
        rt_logger.write(f'{brir_path}: {rt:.2f}\n')
    rt_logger.close()


if __name__ == '__main__':
    brir_dir = sys.argv[1]
    main(brir_dir)
