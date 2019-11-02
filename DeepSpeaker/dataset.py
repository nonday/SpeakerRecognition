import os
import glob

import torch.utils.data as data
import torch

import python_speech_features as psf
import scipy.io.wavfile as wav
import numpy as np

def normalize_frames(m, eps=2e-12):
    return (m - np.mean(m, axis=0)) / (np.std(m, axis=0) + eps)

def get_fbank(wav_path, delta=0, ndim=64):
    '''
    :param wav_path:
    :param delta:
    :return: (nchannel,nframes,ndim)
    '''

    (rate, sig) = wav.read(wav_path)
    features = []

    filter_banks, energies = psf.fbank(sig, samplerate=rate, nfilt=ndim, winlen=0.025, winstep=0.01)
    fbank0 = normalize_frames(filter_banks)

    features.append(fbank0)

    if delta == 1:
        delta_1 = psf.delta(filter_banks, N=1)
        fbank1 = normalize_frames(delta_1)
        features.append(fbank1)

    if delta == 2:
        delta_1 = psf.delta(filter_banks, N=1)
        delta_2 = psf.delta(delta_1, N=1)

        fbank1 = normalize_frames(delta_1)
        fbank2 = normalize_frames(delta_2)

        features.append(fbank1)
        features.append(fbank2)

    return np.array(features)


class Dataset(data.Dataset):

    def __init__(self, path = 'data/',nframes = 160):
        self.labels,self.fpaths= self.get_data(path)
        self.num_frames = nframes

    def get_data(self,base_path):
        spk_list = []
        for spk in os.listdir(base_path):
            spk_path = os.path.join(base_path, spk)
            if os.path.isdir(spk_path):
                spk_list.append(spk)
        spk_list.sort()

        spk_ids = []
        spk_fpaths = []
        for id in range(len(spk_list)):
            spk_files = glob.glob(os.path.join(base_path, spk_list[id], '*.wav'))
            spk_id = [id for _ in range(len(spk_files))]

            spk_ids += spk_id
            spk_fpaths += spk_files

        return spk_ids,spk_fpaths


    def __getitem__(self, index):
        feat = get_fbank(self.fpaths[index], delta=2)
        chn, nframes, nfft = feat.shape
        while nframes < self.num_frames:
            feat = np.concatenate((feat, feat), axis=1)
            chn, nframes, nfft = feat.shape

        X = feat
        Y = self.labels[index]
        nframes, freq = X.shape[1], X.shape[2]
        rand_frame = np.random.randint(0, nframes - self.num_frames)
        X = X[:, rand_frame:rand_frame + self.num_frames, :]

        data = torch.Tensor(X)
        label = torch.LongTensor([Y])
        return data, label

    def __len__(self):
        return len(self.labels)

