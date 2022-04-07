import argparse
import os
import logging

from BasicTools.import_script import import_script

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)


def evaluate(Model, file_reader, model_dir, file_paths, log_path):
    if not os.path.exists(model_dir):
        raise Exception(f'{model_dir} do not exists')

    config_path = f'{model_dir}/config.cfg'
    model_log_path = f'{model_dir}/log'
    model = Model(file_reader=file_reader,
                  config_path=config_path,
                  log_path=model_log_path)
    model.load_model()

    model_basedir = os.path.dirname(model_dir)
    log_path = f'{model_basedir}/evaluations/{log_path}'
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    model.evaluate(file_paths, log_path=log_path)


def parse_args():
    parser = argparse.ArgumentParser(description='parse argments')
    parser.add_argument('--model_script_path', type=str, required=True)
    parser.add_argument('--file_reader_script_path', type=str, required=True)
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--file_list', type=str, required=True)
    parser.add_argument('--log_path', type=str, required=True)
    parser.add_argument('--gpu_id', type=int, default=0)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_visible_devices(gpus[args.gpu_id], 'GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    #
    Model_module = import_script(args.model_script_path)
    Model = Model_module.Model
    #
    file_reader_module = import_script(args.file_reader_script_path)
    file_reader = file_reader_module.file_reader

    with open(args.file_list, 'r') as f:
        file_paths = [item.strip() for item in f.readlines()]

    evaluate(Model, file_reader, args.model_dir, file_paths, args.log_path)


if __name__ == '__main__':
    main()
