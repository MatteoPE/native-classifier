from keras import Input, Model
from keras.layers import Dense


class SimpleAutoencoder:

    def __init__(self, input_size):
        self.input_size = input_size
        self.model, _ = self.build()

    def build(self):
        hidden1 = int(self.input_size / 2)
        hidden2 = int(hidden1 / 2)

        input = Input(shape=(self.input_size,))

        encoded = Dense(units=hidden1 * 2, activation='relu')(input)
        # encoded = Dense(units=hidden2, activation='relu')(encoded)
        # decoded = Dense(units=hidden1, activation='relu')(encoded)
        # decoded = Dense(units=self.input_size, activation='linear')(decoded)
        decoded = Dense(units=self.input_size, activation='linear')(encoded)

        autoencoder = Model(input, decoded)
        encoder = Model(input, encoded)

        autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

        return autoencoder, encoder

    def _custom_loss(self, y_true, y_pred):
        # TODO: Loss function depends on the accuracy of both native and gender classifier
        # do something with y_true and y_pred
        # Returns
        # Tensor with one scalar loss entry per sample
        return [0]

    def fit(self, X_train, batch_size, num_epochs, validation_split, save=True):
        history = self.model.fit(X_train, X_train,
                                 batch_size=batch_size, epochs=num_epochs,
                                 verbose=1,
                                 validation_split=validation_split)

        if save:
            self.model.save('autoencoder.h5')

    def predict(self, X):
        return self.model.predict(X)


class LSTMAutoencoder:

    def __init__(self, input_size):
        self.input_size = input_size
        self.model, _ = self.build()

    def build(self):
        hidden1 = int(self.input_size / 2)
        hidden2 = int(hidden1 / 2)

        input = Input(shape=(self.input_size,))

        encoded = Dense(units=hidden1 * 2, activation='relu')(input)
        # encoded = Dense(units=hidden2, activation='relu')(encoded)
        # decoded = Dense(units=hidden1, activation='relu')(encoded)
        # decoded = Dense(units=self.input_size, activation='linear')(decoded)
        decoded = Dense(units=self.input_size, activation='linear')(encoded)

        autoencoder = Model(input, decoded)
        encoder = Model(input, encoded)

        autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

        return autoencoder, encoder

    def _custom_loss(self, y_true, y_pred):
        # TODO: Loss function depends on the accuracy of both native and gender classifier
        # do something with y_true and y_pred
        # Returns
        # Tensor with one scalar loss entry per sample
        return [0]

    def fit(self, X_train, batch_size, num_epochs, validation_split, save=True):
        history = self.model.fit(X_train, X_train,
                                 batch_size=batch_size, epochs=num_epochs,
                                 verbose=1,
                                 validation_split=validation_split)

        if save:
            self.model.save('autoencoder.h5')

    def predict(self, X):
        return self.model.predict(X)
