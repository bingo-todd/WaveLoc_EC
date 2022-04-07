import sys
import shutil
import numpy as np

from BasicTools.get_file_path import get_file_path
from BasicTools.ProcessBar import ProcessBar

from syn_inter_BRIR import parse_config_file


def main():
    brir_dir = sys.argv[1]
    config_paths = get_file_path(brir_dir, suffix='.cfg')
    pb = ProcessBar(len(config_paths))
    for config_path in config_paths:
        pb.update()
        [room_config,
         receiver_config,
         source_config] = parse_config_file(config_path)

        room_size = [np.round(float(item)*100)/100
                     for item in room_config['Room']['size'].split(',')]
        room_config['Room']['size'] = ', '.join(
            [f'{item:.2f}' for item in room_size])

    shutil.copy(config_path, f'{config_path}_origin')
    with open(config_path, 'w') as config_file:
        room_config.write(config_file)
        receiver_config.write(config_file)
        source_config.write(config_file)


if __name__ == '__main__':
    main()
