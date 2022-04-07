import numpy as np
import matplotlib.pyplot as plt
from BasicTools.get_file_path import get_file_path
from BasicTools.ProcessBar import ProcessBar
from BasicTools import wav_tools


if True:
    brir_paths = get_file_path('../tar_BRIRs_train',
                               suffix='.wav',
                               filter_func=lambda x: x.find('direct') == -1,
                               is_absolute=True)
    n_brir = len(brir_paths)
    print(f'n_brir: {n_brir}')
    pb = ProcessBar(n_brir)
    max_brir_len = 0
    for brir_path in brir_paths:
        pb.update()
        if 'direct' in brir_path:
            continue
        brir, fs = wav_tools.read_wav(brir_path)
        if max_brir_len < brir.shape[0]:
            max_brir_len = brir.shape[0]

    print('cal amp spectrum')
    pb = ProcessBar(n_brir)
    fft_len = int(max_brir_len/2.)
    brir_amp_spec_all = np.zeros((n_brir, fft_len), dtype=np.float16)
    for brir_i, brir_path in enumerate(brir_paths):
        pb.update()
        brir, fs = wav_tools.read_wav(brir_path)
        L_amp_spec = np.abs(np.fft.fft(brir[:, 0], n=max_brir_len)[:fft_len])
        R_amp_spec = np.abs(np.fft.fft(brir[:, 1], n=max_brir_len)[:fft_len])
        brir_amp_spec_all[brir_i] = np.float16(
                np.max(
                    np.concatenate(
                        (L_amp_spec[:, np.newaxis], R_amp_spec[:, np.newaxis]),
                        axis=1),
                    axis=1))
        # if np.min(brir_amp_spec_all[brir_i][950:1050]) < 1e-2:
        #     print(brir_path)
        #     fig, ax = plt.subplots(2, 1)
        #     ax[0].plot(brir[:, 0])
        #     ax[0].plot(brir[::-1, 1])
        #     ax[1].plot(brir_amp_spec_all[brir_i])
        #     fig.savefig('tmp.png')
        #     #
        #     os.system(f'cp {brir_path[:-4]}.cfg ./')
        #     raise Exception()

    np.save('brir_amp_spec_all.npy', brir_amp_spec_all)

brir_amp_spec_all = np.load('brir_amp_spec_all.npy')
brir_amp_spec_mean = np.mean(brir_amp_spec_all, axis=0)
brir_amp_spec_std = np.std(brir_amp_spec_all, axis=0)

fs = 16000
fig, ax = plt.subplots(2, 1)
ax[0].plot(np.arange(fft_len)/max_brir_len*fs,
           np.min(brir_amp_spec_all, axis=0),
           label='min')
ax[0].plot(np.arange(fft_len)/max_brir_len*fs,
           brir_amp_spec_mean,
           label='mean')
ax[0].plot(np.arange(fft_len)/max_brir_len*fs,
           np.max(brir_amp_spec_all, axis=0),
           label='max')
ax[0].legend()
ax[1].plot(np.arange(fft_len)/max_brir_len*fs, brir_amp_spec_std)
# ax[1].set_xlim([0, 10])
fig.savefig('brir_amp_spec.png')
