import os
import numpy as np
import scipy.io.wavfile as wav

from model.autoencoder import SimpleAutoencoder
from utilities.data_loader import DataLoader

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils, to_categorical
from keras.layers import LSTM, RepeatVector, TimeDistributed


def train_autoencoder(data_path):

    sample_rate = 16000  # in bit
    wav_data_size = 6000  # in bit
    duration = 1 / sample_rate * wav_data_size  # in sec

    wnd_len = 0.025  # in sec

    wnd_len_in_bit = int(sample_rate*wnd_len)  # in bit
    num_wnd = int(duration/wnd_len)  # in bit

    data_loader = DataLoader(data_path, wav_data_size)

    data = data_loader.load_data()

    data = np.divide(data, 32768)

    X_data = np.reshape(data, (len(data), num_wnd, wnd_len_in_bit))

    input_shape = (num_wnd, wnd_len_in_bit)

    model = Sequential()
    model.add(LSTM(380, activation="relu",
                   input_shape=input_shape, return_sequences=True))
    model.add(LSTM(350, activation="relu", return_sequences=True))
    model.add(LSTM(300, activation="relu", return_sequences=False))
    model.add(RepeatVector(num_wnd))
    model.add(LSTM(300, activation="relu", return_sequences=True))
    model.add(LSTM(350, activation="relu", return_sequences=True))
    model.add(LSTM(380, activation="relu", return_sequences=True))
    model.add(TimeDistributed(Dense(wnd_len_in_bit)))
    model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])
    model.summary()

    model.fit(X_data, X_data, epochs=100, batch_size=8)


if __name__ == '__main__':

    data_path = os.path.join(
        (os.path.dirname(os.path.dirname(os.getcwd()))), "dataset_wav")

    train_autoencoder(data_path)
