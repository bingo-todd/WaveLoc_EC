import configparser
import sys


def main(brir_dir):
    rt_path = f'{brir_dir}/room_RT.txt'
    rt_logger = open(rt_path, 'w')

    config = configparser.ConfigParser()
    n_room = 888
    for room_i in range(n_room):
        config_path = f'{brir_dir}/00_{room_i:0>4d}.cfg'
        config.read(config_path)
        rt = float(config['Room']['RT60'])
        rt_logger.write(f'{room_i}: {rt:.2f}\n')
    rt_logger.close()


if __name__ == '__main__':
    brir_dir = sys.argv[1]
    main(brir_dir)
