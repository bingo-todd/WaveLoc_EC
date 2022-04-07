from BasicTools.get_file_path import get_file_path

import sys
sys.path.append('../')
from syn_inter_BRIR import parse_config_file  # noqa: E403


def main(train_dir, test_dir):
    train_brir_config_paths = get_file_path(train_dir, suffix='.cfg',
                                            is_absolute=True)
    print(f'train_brir_conifg: {len(train_brir_config_paths)}')
    train_brir_config_dict = {}
    n_repeat = 0
    for config_path in train_brir_config_paths:
        [room_config,
         receiver_config,
         source_config] = parse_config_file(config_path)

        room_info = '{}_{}'.format(room_config['Room']['size'],
                                   room_config['Room']['RT60'])

        if room_info in train_brir_config_dict.keys():
            n_repeat = n_repeat + 1
            # raise Exception(room_info)

        train_brir_config_dict[room_info] = config_path
    print(n_repeat)

    print('test')
    test_brir_config_paths = get_file_path(test_dir,
                                           suffix='.cfg',
                                           is_absolute=True)
    print(f'test_brir_conifg: {len(test_brir_config_paths)}')
    n_repeat = 0
    for config_path in test_brir_config_paths:
        [room_config,
         receiver_config,
         source_config] = parse_config_file(config_path)

        room_info = '{}_{}'.format(room_config['Room']['size'],
                                   room_config['Room']['RT60'])

        if room_info in train_brir_config_dict.keys():
            n_repeat = n_repeat + 1
            print(config_path)
            continue

            print('test')
            print('Receiver')
            print(list(receiver_config['Receiver'].items()))
            print('Source')
            print(list(source_config['Source'].items()))

            [room_config,
             receiver_config,
             source_config] = parse_config_file(
                 train_brir_config_dict[room_info])
            print('train')
            print('Receiver')
            print(list(receiver_config['Receiver'].items()))
            print('Source')
            print(list(source_config['Source'].items()))

            # input()

            # raise Exception(room_info)
    print(f'n_repeat {n_repeat}')


if __name__ == '__main__':
    main('../tar_BRIRs_train', '../tar_BRIRs_test')
