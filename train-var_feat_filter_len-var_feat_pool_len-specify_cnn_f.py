import configparser
import argparse
import os
import logging

from BasicTools.import_script import import_script

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import tensorflow as tf  # noqa: E402


def train(args):
    # select GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[args.gpu_id], 'GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    WaveLoc_module = import_script(args.model_script_path)
    Model = WaveLoc_module.Model
    file_reader_module = import_script(args.file_reader_script_path)
    file_reader = file_reader_module.file_reader

    frame_len = 320*2
    n_unit_fcn = 128
    epsilon_energy = 1e-4

    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)

    config_path = f'{args.model_dir}/config.cfg'
    if not os.path.exists(config_path):
        config = configparser.ConfigParser()
        config['model'] = {'model_script_path': args.model_script_path,
                           'n_unit_fcn': n_unit_fcn,
                           'n_feat_filter': args.n_feat_filter,
                           'feat_filter_len': args.feat_filter_len,
                           'pool_len': args.pool_len,
                           'cnn_coef': args.cnn_coef,
                           'epsilon_energy': epsilon_energy,
                           'fs': 16000,
                           'n_band': 32,
                           'cf_low': 70,
                           'cf_high': 7000,
                           'frame_len': frame_len,
                           'shift_len': 160,
                           'filter_len': 320,
                           'n_azi': 37}

        config['train'] = {'model_dir': args.model_dir,
                           'batch_size': args.batch_size,
                           'shuffle_size': args.batch_size*10,
                           'init_lr': 0.001,
                           'max_epoch': 100,
                           'is_print_log': True,
                           'train_file_list': args.train_file_list,
                           'valid_file_list': args.valid_file_list}

        with open(config_path, 'w') as config_file:
            if config_file is None:
                raise Exception('fail to create file')
            config.write(config_file)
    log_path = f'{args.model_dir}/log'
    model = Model(
        file_reader, config_path=config_path, log_path=log_path)
    model.train()


def parse_args():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--model_script_path', required=True, type=str,
                        help='')
    parser.add_argument('--file_reader_script_path', type=str,
                        default='file_reader.py', help='')
    parser.add_argument('--train_file_list', type=str, required=True)
    parser.add_argument('--valid_file_list', type=str, required=True)
    parser.add_argument('--n_feat_filter', required=True, type=int, help='')
    parser.add_argument('--cnn_coef', type=int, required=True, help='')
    parser.add_argument('--feat_filter_len', required=True, type=int, help='')
    parser.add_argument('--pool_len', required=True, type=int, help='')
    parser.add_argument('--model_dir', required=True, type=str, help='')
    parser.add_argument('--batch_size', type=int, default=256, help='')
    parser.add_argument('--gpu_id', type=int, default=0, help='')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    train(args)


if __name__ == '__main__':
    main()
