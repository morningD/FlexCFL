import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Reshape, Conv2D, MaxPool2D, Flatten, Dropout
from tensorflow.keras import Sequential

def _construct_client_model(lr):
    model = Sequential()
    # Input Layer
    model.add(Input(shape=(784,)))
    # Reshape Layer
    model.add(Reshape((28, 28, 1)))
    # Conv1 Layer
    model.add(Conv2D(32, 3, padding='same', activation='relu'))
    # MaxPool1 Layer
    model.add(MaxPool2D(pool_size=(2, 2)))
    # Conv2 Layer
    model.add(Conv2D(64, 3, padding='same', activation='relu'))
    # MaxPool2 Layer
    model.add(MaxPool2D(pool_size=(2, 2)))
    # Flatten Layer
    model.add(Flatten())
    # Dropout Layer
    model.add(Dropout(0.5))
    # Output Layer
    model.add(Dense(10, 'softmax'))
    
    opt = tf.keras.optimizers.SGD(learning_rate=lr)
    #opt = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(opt, loss_fn, metrics=['accuracy'])
    return model

def construct_model(trainer_type, lr):
    if trainer_type == 'fedavg':
        return _construct_client_model(lr)
    else:
        return _construct_client_model(lr)
