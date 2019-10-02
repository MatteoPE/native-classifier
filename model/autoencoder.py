from keras import Input, Model
from keras.layers import Dense


class SimpleAutoencoder:

    def __init__(self, input_size):
        self.input_size = input_size
        self.model, _ = self.build()

    def build(self):
        input = Input(shape=(self.input_size,))

        encoded = Dense(units=128, activation='relu')(input)
        encoded = Dense(units=64, activation='relu')(encoded)
        encoded = Dense(units=32, activation='relu')(encoded)
        decoded = Dense(units=64, activation='relu')(encoded)
        decoded = Dense(units=128, activation='relu')(decoded)
        decoded = Dense(units=self.input_size, activation='sigmoid')(decoded)

        autoencoder = Model(input, decoded)
        encoder = Model(input, encoded)

        autoencoder.compile(optimizer='adam', loss=self._custom_loss, metrics=['accuracy'])

        return autoencoder, encoder

    def _custom_loss(self, y_true, y_pred):
        # TODO: Loss function depends on the accuracy of both native and gender classifier
        # do something with y_true and y_pred
        # Returns
        # Tensor with one scalar loss entry per sample
        return [0]

    def fit(self, X_train, y_train, X_val, y_val, batch_size, num_epochs, save=True):
        history = self.model.fit(X_train, y_train,
                                 batch_size=batch_size, epochs=num_epochs,
                                 verbose=1,
                                 validation_data=(X_val, y_val))

        if save:
            self.model.save('model.h5')

    def predict(self, X):
        return self.model.predict(X)