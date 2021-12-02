from read_data import read_federated_data
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, Input

def main():
    dsname = "femnist"
    test_model()
    
    clients, train_data, test_data = read_federated_data(dsname)

    print(clients)
    print("Input shape of first client:",np.shape(train_data[clients[0]]['x']))
    print("Labels of first client:", train_data[clients[0]]['y'])
    

def test_model():
    def _construct_client_model(lr):
        model = Sequential()
        model.add(Input(shape=(25, 300)))
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(Bidirectional(LSTM(32)))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        # Output layer
        model.add(Dense(1))

        opt = tf.keras.optimizers.SGD(learning_rate=lr)
        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        model.compile(opt, loss_fn, metrics=['accuracy'])
        return model

    model = _construct_client_model(lr=0.3)

main()