import matplotlib.pyplot as plt
import numpy as np
from BasicTools import wav_tools
from BasicTools.ProcessBar import ProcessBar
from BasicTools.get_file_path import get_file_path


def get_amp_range(record_dir):
    record_paths = get_file_path(record_dir, suffix='.wav', is_absolute=True)
    n_record = len(record_paths)
    max_amp_all = np.zeros(n_record)
    pb = ProcessBar(n_record)
    for record_i, record_path in enumerate(record_paths):
        pb.update()
        wav, fs = wav_tools.read_wav(record_path)
        max_amp_all[record_i] = np.max(np.abs(wav))

    # np.save('amp_range.npy', max_amp_all)
    print(record_paths[np.argmin(max_amp_all)])
    return max_amp_all


def main():
    fig, ax = plt.subplots(1, 1)

    max_amp_all = get_amp_range('/home/st/Work_Space/Data/TIMIT_wav/TIMIT')
    ax.scatter(max_amp_all, max_amp_all+0.02, alpha=0.5, label='TIMIT')
    min_amp = np.min(max_amp_all)
    ax.text(min_amp-0.02, min_amp+0.02, f'{min_amp:.2e}',
            va='center', ha='center')

    max_amp_all = get_amp_range('../../Data/realroom/tar/test/reverb')
    ax.scatter(max_amp_all, max_amp_all-0.02, alpha=0.5, label='tar_test')
    min_amp = np.min(max_amp_all)
    ax.text(min_amp+0.02, min_amp-0.02, f'{min_amp:.2e}',
            va='center', ha='center')

    ax.legend()
    fig.savefig('amp_range_realroom.png')


if __name__ == '__main__':
    main()
