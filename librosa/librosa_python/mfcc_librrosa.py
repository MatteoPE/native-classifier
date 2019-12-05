import os
import librosa

sample_rate = None
wav_data_size = None
wnd_step = None
wnd_len = None
num_wnd = None
num_features = None


class Librosa_Mfcc:

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

        wav_data, samplerate = librosa.load(self.data_path, self.sample_rate)
        # e.g. 6099

        curr_data_size = wav_data.shape[0]
        # e.g. (6099-6000)/2 = 99/2 = 49

        start_idx = int((curr_data_size - wav_data_size) / 2)
        # e.g. 49+6000 = 6049

        end_idx = start_idx + wav_data_size
        X_data.append(wav_data[start_idx:end_idx])

        return X_data

    def get_mfcc(self):
        data = self.load_data();
        mfcc = librosa.feature.mfcc(y=data, sr=self.sample_rate, n_fft=2048, hop_length=self.wnd_step)
        #, win_length=self.wnd_len
        return mfcc


if __name__ == '__main__':

    sample_rate = 16000
    wav_data_size = 6000
    duration = 1 / sample_rate * wav_data_size

    wnd_len = 0.025
    wnd_step = 0.010
    num_wnd = 1 + ((duration - wnd_len) / wnd_step)
    num_features = 26
    #obj_librosa_mfcc = Librosa_Mfcc('AAMNG1001.wav',wav_data_size,sample_rate,wnd_len,wnd_step,num_features,num_wnd);
    #mfcc = obj_librosa_mfcc.get_mfcc();
    wav_data, samplerate = librosa.load('AAMNG1001.wav', sample_rate)
    mfcc = librosa.feature.mfcc(y=wav_data, sr=samplerate, n_fft=2048, hop_length=512)

    print(mfcc);