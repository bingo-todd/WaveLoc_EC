import os
import matplotlib.pyplot as plt
import configparser

import RoomSimulator
# from BasicTools import wav_tools
from BasicTools.ProgressBar import ProgressBar


Fs = 16000
n_room_per_azi = 30
n_azi = 37


def main(brir_dir, RT_range):
    os.makedirs(brir_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 2)
    drr_logger = open(f'{brir_dir}/DRR.txt', 'x')

    pb = ProgressBar(n_azi*n_room_per_azi)
    n_worker = 4
    for azi_i in range(n_azi):
        for room_i in range(n_room_per_azi):
            pb.update()
            file_name = f'{azi_i:0>2d}_{room_i:0>4d}'
            reverb_config_path = f'{brir_dir}/{file_name}.cfg'
            reverb_brir_path = f'{brir_dir}/{file_name}.wav'

            config = configparser.ConfigParser()
            config.read(reverb_config_path)

            room_config = configparser.ConfigParser()
            room_config['Room'] = config['Room']
            #
            receiver_config = configparser.ConfigParser()
            receiver_config['Receiver'] = config['Receiver']
            for mic_i in range(2):
                receiver_config[f'Mic_{mic_i}'] = config[f'Mic_{mic_i}']
            #
            source_config = configparser.ConfigParser()
            source_config['Source'] = config['Source']

            roomsim = RoomSimulator.RoomSimulator(
                    room_config=room_config,
                    source_config=source_config,
                    receiver_config=receiver_config,
                    parent_pid=os.getpid())
            roomsim.cal_all_img()
#
            drr = roomsim.cal_DRR(n_worker=n_worker)
            drr_logger.write(f'{os.path.realpath(reverb_brir_path)}: {drr}\n')
    drr_logger.close()


if __name__ == '__main__':
    #
    RT_range = [0, 0.8]
    main('../tar_BRIRs_train',  RT_range)
