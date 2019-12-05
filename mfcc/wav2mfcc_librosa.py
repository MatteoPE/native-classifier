import numpy as np
from librosa.feature import mfcc
from python_speech_features import delta


class Wav2MfccLibrosa:

    def __init__(self, sample_rate, num_features, wnd_step, wnd_len):
        self.sample_rate = sample_rate
        self.num_features = num_features
        self.wnd_step = wnd_step
        self.wnd_len = wnd_len

    def mfcc_delta_feature_extraction(self, data):
        vectorized_data = []
        hop_length = int(self.wnd_step * self.sample_rate)
        win_length = int(self.wnd_len * self.sample_rate)
        for d in data:
            mfcc_librosa = mfcc(d, sr=self.sample_rate, n_mfcc=self.num_features, hop_length=hop_length,
                                win_length=win_length)
            # mfcc_librosa = mfcc(d, sr=self.sample_rate, n_mfcc=self.num_features)
            mfcc_f = np.transpose(mfcc_librosa)
            delta_f = delta(mfcc_f, 8)
            v_d = np.append(mfcc_f, delta_f, axis=1)
            vectorized_data.append(v_d)
        return vectorized_data
