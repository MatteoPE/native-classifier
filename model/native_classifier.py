from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils, to_categorical
from keras.layers import LSTM

from time import time
from keras.callbacks import TensorBoard


class NativeClassifierLSTM:

    def __init__(self, n_input, n_hidden, n_output):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.model = self.build()

    def build(self):
        model = Sequential()

        model.add(LSTM(self.n_hidden, input_shape=self.n_input, return_sequences=True, dropout=0.2,
                       recurrent_dropout=0.2))
        model.add(LSTM(self.n_hidden, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))

        model.add(LSTM(self.n_hidden * 2, dropout=0.3, recurrent_dropout=0.3))

        model.add(Dense(self.n_output))
        model.add(Activation('softmax'))

        # tensorboard = TensorBoard(log_dir="log/tensorboard_log")

        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    def fit(self, X_train, y_train, X_val, y_val, batch_size, num_epochs, save=True):
        history = self.model.fit(X_train, y_train,
                                 batch_size=batch_size, epochs=num_epochs,
                                 verbose=1,
                                 validation_data=(X_val, y_val))

        if save:
            self.model.save('model.h5')
