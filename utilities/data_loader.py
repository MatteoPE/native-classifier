import os
import numpy as np
import scipy.io.wavfile as wav
import librosa

class DataLoader:

    def __init__(self, data_path, wav_data_size, target_id=3, target_char='Y'):
        self.data_path = data_path
        self.wav_data_size = wav_data_size
        self.target_id = target_id
        self.target_char = target_char

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
                target = subdir.replace((self.data_path + "/"), "")
                # true for native ('Y'), false for non-native ('N')
                y_data.append(target[self.target_id] == self.target_char)
        return X_data, y_data

class DataLoaderLibrosa:

    def __init__(self, data_path, wav_data_size, sample_rate, target_id=3, target_char='Y'):
        self.data_path = data_path
        self.wav_data_size = wav_data_size
        self.sample_rate = sample_rate
        self.target_id = target_id
        self.target_char = target_char

    def load_data(self):
        X_data = []
        y_data = []
        for subdir, dirs, files in os.walk(self.data_path):
            print(subdir)
            for file in files:
                # print(os.path.join(subdir, file), subdir.replace((data_path + "/"), ""))
                # ./dataset_png/AEFYG1/AEFYG1037.png AEFYG1
                wav_data, sample_rate = librosa.load(os.path.join(subdir, file), sr=self.sample_rate)
                # e.g. 6099
                curr_data_size = wav_data.shape[0]
                # e.g. (6099-6000)/2 = 99/2 = 49
                start_idx = int((curr_data_size - self.wav_data_size) / 2)
                # e.g. 49+6000 = 6049
                end_idx = start_idx + self.wav_data_size
                X_data.append(wav_data[start_idx:end_idx])
                target = subdir.replace((self.data_path + "/"), "")
                # true for native ('Y'), false for non-native ('N')
                y_data.append(target[self.target_id] == self.target_char)
        return X_data, y_data
