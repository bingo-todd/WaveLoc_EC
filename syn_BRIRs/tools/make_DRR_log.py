import os
import numpy as np

from BasicTools import wav_tools
from BasicTools.get_file_path import get_file_path
from LocTools.add_log import add_log


def make_DRR_log(brir_dir):
    reverb_brir_paths = get_file_path(
        brir_dir, suffix='.wav',
        filter_func=lambda x: x.find('direct') < 0,
        is_absolute=True)

    DRR_logger = open(f'{brir_dir}/DRR.txt', 'x')
    for reverb_brir_path in reverb_brir_paths:
        brir_name = os.path.basename(reverb_brir_path)[:-4]
        direct_brir_path = f'{brir_dir}/{brir_name}_direct.wav'

        reverb_brir, fs = wav_tools.read(reverb_brir_path)
        direct_brir, fs = wav_tools.read(direct_brir_path)

        direct = direct_brir
        reverb = reverb_brir - direct_brir
        DRR = 10*np.log10(np.sum(direct**2, axis=0)/np.sum(reverb**2, axis=0))
        add_log(DRR_logger, reverb_brir_path, [DRR])


if __name__ == '__main__':
    import sys
    brir_dir = sys.argv[1]
    make_DRR_log(brir_dir)
