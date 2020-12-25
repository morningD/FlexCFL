import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Reshape, Conv2D, MaxPool2D, Flatten
from tensorflow.keras import Sequential

def _construct_client_model(lr):
    model = Sequential()
    # Input Layer
    model.add(Input(shape=(784,)))
    # Reshape Layer
    model.add(Reshape((28, 28, 1)))
    # Conv Layer
    model.add(Conv2D(24, 5, padding='same', activation='relu'))
    # MaxPool Layer
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    # Flatten Layer
    model.add(Flatten())
    # Dense Layer
    model.add(Dense(256, 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    # Output Layer
    model.add(Dense(10, 'softmax', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    
    opt = tf.keras.optimizers.SGD(learning_rate=lr)
    #opt = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(opt, loss_fn, metrics=['accuracy'])
    return model

def construct_model(trainer_type, lr):
    if trainer_type == 'fedavg':
        return _construct_client_model(lr)
    else:
        return