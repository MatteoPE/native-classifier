import numpy as np
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from mfcc.wav2mfcc import Wav2Mfcc
from model.native_classifier import NativeClassifierLSTM

if __name__ == '__main__':

    sample_rate = 16000
    wav_data_size = 6000
    duration = 1 / sample_rate * wav_data_size

    data_path = "./dataset_wav"

    wnd_len = 0.025
    wnd_step = 0.010
    num_wnd = 1 + ((duration - wnd_len) / wnd_step)
    num_features = 26

    feature_extraction = Wav2Mfcc(data_path, wav_data_size, sample_rate, wnd_len, wnd_step, num_features, num_wnd)

    X_data_wav, y_data = feature_extraction.load_data()

    X_data_mfcc = feature_extraction.mfcc_delta_feature_extraction(X_data_wav)

    y_data = [1 * val for val in y_data]

    X_test = np.array(X_data_mfcc[:250] + X_data_mfcc[500:750])
    y_test = y_data[:250] + y_data[500:750]
    X_train = np.array(X_data_mfcc[250:500] + X_data_mfcc[750:])
    y_train = y_data[250:500] + y_data[750:]

    scalers = {}
    for i in range(X_train.shape[2]):
        scalers[i] = StandardScaler()
        X_train[:, :, i] = scalers[i].fit_transform(X_train[:, :, i])

    n_input = X_train[0].shape
    n_hidden = 64
    n_output = 2

    model = NativeClassifierLSTM(n_input, n_hidden, n_output)

    X_train_NN = X_train
    X_test_NN = X_test

    for i in range(X_test_NN.shape[2]):
        X_test_NN[:, :, i] = scalers[i].transform(X_test_NN[:, :, i])
        X_train_NN[:, :, i] = scalers[i].transform(X_train_NN[:, :, i])

    y_train_NN = to_categorical(y_train)
    y_test_NN = to_categorical(y_test)

    X_val_NN, X_test_NN, y_val_NN, y_test_NN = train_test_split(X_test_NN, y_test_NN, test_size=0.5, random_state=42,
                                                                stratify=y_test_NN)

    batch_size = 32
    num_epochs = 20

    model.fit(X_train_NN, y_train_NN, X_val_NN, y_val_NN, batch_size, num_epochs)