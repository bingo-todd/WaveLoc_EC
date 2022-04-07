import os
import matplotlib.pyplot as plt
import configparser
import numpy as np

# modules developed by Tao Song
import RoomSimulator
from BasicTools import wav_tools
from BasicTools.reverb.RT2A import RT2A


# basic settings
Fs = 16000  # sample frequency
n_azi = 37
n_room_per_azi = 30


def gen_rand(min_value, max_value, rand_state):
    """generate random value within the given range"""
    return rand_state.rand()*(max_value-min_value)+min_value


def cos_degree(degree):
    """cos function which take degree as input"""
    return np.cos(degree/180*np.pi)


def sin_degree(degree):
    """sin function which take degree as input"""
    return np.sin(degree/180*np.pi)


def gen_room_settings(RT_range, rand_state):
    max_wall_len = 12
    min_wall_len = 4

    min_A = 0.01
    max_A = 0.85

    min_RT, max_RT = RT_range

    room_height = 4  # room height is fixed as 4 m
    RT = gen_rand(max_RT, min_RT, rand_state)

    # try multiple times when randomly generated settings do not meet
    # requirements
    n_run = 0

    while True:
        room_width = gen_rand(max_wall_len, min_wall_len, rand_state)
        room_length = gen_rand(max_wall_len, min_wall_len, rand_state)
        room_size = [room_width, room_length, room_height]
        A = RT2A(RT, room_size)
        if np.max(A) <= max_A and np.min(A) >= min_A:
            break
        n_run = n_run + 1
        if n_run > 20:
            RT = gen_rand(max_RT, min_RT, rand_state)
        if n_run > 100:
            raise Exception()

    room_size = [room_width, room_length, room_height]
    room_settings = {
            'size': ', '.join([f'{item:.2f}' for item in room_size]),
            'RT60': f'{RT:.2f}',
            'A': '',
            'Fs': Fs,
            'reflect_order': '-1',
            'HP_cutoff': '100'}
    return room_settings


def gen_receiver_settings(room_settings, rand_state):
    room_size = list(map(float, room_settings['size'].split(', ')))
    pos_z = 1.2
    dist_from_wall = 1.5
    pos_x = gen_rand(dist_from_wall, room_size[0]-dist_from_wall,
                     rand_state)
    pos_y = gen_rand(dist_from_wall, room_size[1]-dist_from_wall,
                     rand_state)
    yaw = gen_rand(-180, 180, rand_state)
    head_r = 0.145/2
    receiver_settings = {
            'pos': f'{pos_x:.2f}, {pos_y:.2f}, {pos_z:.2f}',
            'rotate': f'0, 0, {yaw:.2f}',
            'direct_type': 'omnidirectional',
            'n_mic': '2'}
    mic0_settings = {
            'pos': f'0, {head_r}, 0',
            'rotate': '0, 0, 90',
            'direct_type': 'binaural_L'}
    mic1_settings = {
            'pos': f'0, {-head_r}, 0',
            'rotate': '0, 0, -90',
            'direct_type': 'binaural_R'}
    return receiver_settings, mic0_settings, mic1_settings


def gen_sourece_settings(room_settings, receiver_settings, azi,
                         rand_state):

    min_dist = 1

    room_size = list(map(float, room_settings['size'].split(', ')))
    receiver_pos = list(map(float, receiver_settings['pos'].split(', ')))
    receiver_rotate = list(map(float, receiver_settings['rotate'].split(', ')))
    azi_room = ((180 - (receiver_rotate[2] - azi)) % 360) - 180

    dist_from_wall = 0.5
    if np.abs(azi_room) < 90:
        max_dist_x = np.abs(
            (room_size[0]-dist_from_wall-receiver_pos[0])/cos_degree(azi_room))
    elif np.abs(azi_room) > 90:
        max_dist_x = np.abs(
            (receiver_pos[0]-dist_from_wall)/cos_degree(azi_room))
    else:
        max_dist_x = room_size[0]

    if azi_room > 0 and azi_room < 180:
        max_dist_y = np.abs(
            (receiver_pos[1]-dist_from_wall)/sin_degree(azi_room))
    elif azi_room < 0 and azi_room > -180:
        max_dist_y = np.abs(
            (room_size[1]-dist_from_wall-receiver_pos[1])/sin_degree(azi_room))
    else:
        max_dist_y = room_size[1]
    max_dist = min((max_dist_x, max_dist_y))

    if max_dist < min_dist:
        print(f'illegal settings: dist range [{min_dist} {max_dist}]')
        raise Exception()

    dist = gen_rand(min_dist, max_dist, rand_state)
    source_pos = [
            receiver_pos[0]+dist*cos_degree(azi_room),
            receiver_pos[1]-dist*sin_degree(azi_room),
            receiver_pos[2]]

    if (source_pos[0] < dist_from_wall
            or source_pos[0] >= room_size[0]-dist_from_wall):
        print(f'{source_pos[0]}: illegal pos')
        print('\n'*5)
        raise Exception(f'{source_pos[0]}: illegal pos')

    if (source_pos[1] < dist_from_wall
            or source_pos[1] >= room_size[1]-dist_from_wall):
        print(f'{source_pos[1]}: illegal pos')
        print('\n'*5)
        raise Exception(f'{source_pos[0]}: illegal pos')

    source_settings = {
            'pos': ', '.join(map(lambda x: f'{x:.2f}', source_pos)),
            'rotate': '0, 0, 0',
            'directivity': 'omnidirectional'}
    return source_settings


def gen_configs(azi, RT_range, rand_state):
    room_settings = gen_room_settings(RT_range, rand_state)
    [receiver_settings,
     *mic_settings_all] = gen_receiver_settings(room_settings, rand_state)
    source_settings = gen_sourece_settings(room_settings,
                                           receiver_settings,
                                           azi,
                                           rand_state)

    room_config = configparser.ConfigParser()
    room_config['Room'] = room_settings

    receiver_config = configparser.ConfigParser()
    receiver_config['Receiver'] = receiver_settings
    for mic_i, mic_settings in enumerate(mic_settings_all):
        receiver_config[f'Mic_{mic_i}'] = mic_settings

    source_config = configparser.ConfigParser()
    source_config['Source'] = source_settings

    return room_config, receiver_config, source_config


def syn_brir(configs, reverb_config_path, reverb_brir_path, fig_path,
             parallel_type, n_worker):

    if os.path.exists(reverb_config_path):
        print('exists')
        return
    print(reverb_config_path)

    room_config, receiver_config, source_config = configs
    # save configure to file
    with open(reverb_config_path, 'x') as config_file:
        room_config.write(config_file)
        receiver_config.write(config_file)
        source_config.write(config_file)

    roomsim = RoomSimulator.RoomSimulator(
            room_config=room_config,
            source_config=source_config,
            receiver_config=receiver_config,
            parent_pid=os.getpid())
    roomsim.cal_all_img()
#
    direct_brir = roomsim.cal_direct_ir_mic()
    direct_brir_path = f'{reverb_brir_path[:-4]}_direct.wav'
    wav_tools.write(direct_brir, Fs, direct_brir_path)
    #
    brir = roomsim.cal_ir_mic(parallel_type=parallel_type, n_worker=n_worker)
    wav_tools.write(brir, Fs, reverb_brir_path)

    drr = roomsim.cal_DRR(n_worker=n_worker)
    wav_tools.write(brir, Fs, reverb_brir_path)

    if False:
        fig = plt.figure(figsize=[10, 4])
        ax_full = fig.add_subplot(121, projection='3d')
        roomsim.visualize(ax_full)
        ax_full.legend([])
        ax_full.set_title(source_config['Source']['pos'])
        ax_full.view_init(60, -60)
        ax_zoom = fig.add_subplot(122, projection='3d')
        roomsim.visualize(ax_zoom, is_zoom=True)
        ax_zoom.legend([])
        ax_zoom.view_init(60, -60)
        fig.savefig(fig_path, dpi=120)
        plt.close(fig)

    # roomsim.clean_dump()
    return drr


def main(brir_dir, RT_range):
    os.makedirs(brir_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 2)
    drr_logger = open(f'{brir_dir}/drr.txt', 'a')

    # parameters for RoomSimulator
    parallel_type = 2
    n_worker = 8

    for azi_i in range(n_azi):
        azi = 5*azi_i-90
        for room_i in range(n_room_per_azi):
            # pb.update()
            file_name = f'{azi_i:0>2d}_{room_i:0>4d}'
            reverb_config_path = f'{brir_dir}/{file_name}.cfg'
            reverb_brir_path = f'{brir_dir}/{file_name}.wav'
            fig_path = f'{brir_dir}/{file_name}.png'

            if (os.path.exists(reverb_brir_path)
                    and os.path.exists(reverb_config_path)):
                print('continue')
                continue

            rand_state = np.random.RandomState()
            configs = gen_configs(azi, RT_range, rand_state)

            drr = syn_brir(configs, reverb_config_path, reverb_brir_path,
                           fig_path, parallel_type, n_worker)
            drr_logger.write('{}: {}'.format(
                os.path.realpath(reverb_brir_path), drr))
    drr_logger.close()


if __name__ == '__main__':
    #
    brir_dir = 'BRIRs'
    RT_range = [0, 0.8]
    main(brir_dir,  RT_range)
