import os
import sys
import numpy as np
import matplotlib.pyplot as plt

from RoomSimulator import RoomSimulator
from BasicTools.get_file_path import get_file_path
from LocTools.add_log import add_log


def main():
    brir_dir, fig_path = sys.argv[1:3]
    config_paths = get_file_path(brir_dir,
                                 suffix='.cfg',
                                 is_absolute=True,
                                 root_dir='~/Work_Space')
    A_path = f'{brir_dir}/A.txt'
    A_logger = open(A_path, 'w')
    A_all = []
    for config_path in config_paths:
        roomsimulator = RoomSimulator(os.path.expanduser(config_path))
        record_path = config_path[:-4]+'.wav'
        A_all.append(roomsimulator.room.A[0, :])
        add_log(A_logger, record_path, [roomsimulator.room.A[0, :]])
    A_logger.close()
    A_all = np.concatenate(A_all)
    fig, ax = plt.subplots(1, 1)
    values, counts = np.unique(A_all, return_counts=True)
    ax.plot(values, counts)
    fig.savefig(fig_path)


if __name__ == '__main__':
    main()
