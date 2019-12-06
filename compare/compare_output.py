import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
from librosa.feature import mfcc


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def load_java_data(path):
    print("load java data...")
    data = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            x = np.loadtxt(os.path.join(subdir, file))
            # print(x.shape)
            data.append(x)
    print("...java data loaded")
    return data


def load_librosa_data(java_path, wav_path):
    print("generating librosa data...")
    sample_rate = 16000
    librosa_data = []
    java_data = []
    for subdir, dirs, files in os.walk(wav_path):
        for file in files:
            wav_data, sample_rate = librosa.load(os.path.join(subdir, file), sr=sample_rate)
            mfcc_data = mfcc(wav_data, sr=sample_rate).flatten('F')
            librosa_data.append(mfcc_data)

            folder = os.path.basename(subdir)
            txt_path = os.path.join(java_path, folder, file).replace(".wav", ".txt")
            txt_data = np.loadtxt(txt_path)
            java_data.append(txt_data)
            if mfcc_data.shape != txt_data.shape:
                print(file)
    print("...librosa data generated")
    return librosa_data, java_data


if __name__ == "__main__":
    java_path = os.path.join(os.path.dirname(os.getcwd()), "dataset_java_feature")
    wav_path = os.path.join(os.path.dirname(os.getcwd()), "dataset_wav")
    print(java_path)
    print(wav_path)
    librosa_data, java_data = load_librosa_data(java_path, wav_path)

    rmse_array = []
    for i in range(len(librosa_data[0])):
        rmse_val = rmse(np.array(librosa_data[i]), np.array(java_data[i]))
        rmse_array.append(rmse_val)
    n_bins = 50
    plt.hist(rmse_array, bins=n_bins,
             range=(np.min(rmse_array),0.000007),
             density=True,
             cumulative=True)
    plt.xlabel("Root-Mean-Square Error")
    plt.ylabel("Density")
    plt.show()
