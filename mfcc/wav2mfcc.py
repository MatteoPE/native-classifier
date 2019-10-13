import os
import numpy as np
import scipy.io.wavfile as wav
from python_speech_features import mfcc
from python_speech_features import delta


class Wav2Mfcc:

    def __init__(self, data_path, wav_data_size, sample_rate, wnd_len, wnd_step, num_features, num_wnd):
        self.data_path = data_path
        self.wav_data_size = wav_data_size
        self.sample_rate = sample_rate
        self.wnd_len = wnd_len
        self.wnd_step = wnd_step
        self.num_features = num_features
        self.num_wnd = num_wnd

    def load_data(self):
        X_data = []
        y_data = []
        for subdir, dirs, files in os.walk(self.data_path):
            print(subdir)
            for file in files:
                # print(os.path.join(subdir, file), subdir.replace((data_path + "/"), ""))
                # ./dataset_png/AEFYG1/AEFYG1037.png AEFYG1
                sample_rate, wav_data = wav.read(os.path.join(subdir, file))
                # e.g. 6099
                curr_data_size = wav_data.shape[0]
                # e.g. (6099-6000)/2 = 99/2 = 49
                start_idx = int((curr_data_size - self.wav_data_size) / 2)
                # e.g. 49+6000 = 6049
                end_idx = start_idx + self.wav_data_size
                X_data.append(wav_data[start_idx:end_idx])
                #target = subdir.replace((self.data_path + "/"), "")
                # true for native ('Y'), false for non-native ('N')
                #y_data.append(target[3] == 'Y')
        return X_data #, y_data

    def mfcc_feature_extraction(self, data):
        vectorized_data = []
        for d in data:
            v_d = mfcc(d, samplerate=self.sample_rate, winlen=self.wnd_len, winstep=self.wnd_step,
                       numcep=self.num_features)
            #         v_d = np.reshape(v_d, (int(num_wnd*num_features), ))
            vectorized_data.append(v_d)
        return vectorized_data

    def mfcc_delta_feature_extraction(self, data):
        vectorized_data = []
        for d in data:
            mfcc_f = mfcc(d, samplerate=self.sample_rate, winlen=self.wnd_len, winstep=self.wnd_step,
                          numcep=self.num_features)
            delta_f = delta(mfcc_f, 8)
            v_d = np.append(mfcc_f, delta_f, axis=1)
            #         v_d = np.reshape(v_d, (int(num_wnd*num_features*2), ))
            vectorized_data.append(v_d)
        return vectorized_data

    def mfcc_delta_deltadelta_feature_extraction(self, data):
        vectorized_data = []
        for d in data:
            mfcc_f = mfcc(d, samplerate=self.sample_rate, winlen=self.wnd_len, winstep=self.wnd_step,
                          numcep=self.num_features)
            delta_f = delta(mfcc_f, 8)
            deltadelta_f = delta(delta_f, 8)
            v_d = np.append(mfcc_f, delta_f, axis=1)
            v_d = np.append(v_d, deltadelta_f, axis=1)
            #         v_d = np.reshape(v_d, (int(num_wnd*num_features*3), ))
            vectorized_data.append(v_d)
        return vectorized_data
