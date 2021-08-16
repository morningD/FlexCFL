import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Reshape, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dropout
from tensorflow.keras import Sequential

def _construct_client_model(lr):
    model = Sequential()
    # Input Layer
    model.add(Input(shape=(784,)))
    # Reshape Layer
    model.add(Reshape((28, 28, 1)))

    # Conv1 Layer
    model.add(Conv2D(32, 3, activation='relu'))
    # BN1 Layer
    #model.add(BatchNormalization())
    # Conv2 Layer
    model.add(Conv2D(32, 3, activation='relu'))
    # BN2 Layer
    #model.add(BatchNormalization())
    # Conv3 Layer
    model.add(Conv2D(32, 5, strides=2, padding='same', activation='relu'))
    # BN3 Layer
    #model.add(BatchNormalization())
    # Droupout1
    model.add(Dropout(0.4))

    # Conv4 Layer
    model.add(Conv2D(64, 3, activation='relu'))
    # BN4 Layer
    #model.add(BatchNormalization())
    # Conv5 Layer
    model.add(Conv2D(64, 3, activation='relu'))
    # BN5 Layer
    #model.add(BatchNormalization())
    # Conv6 Layer
    model.add(Conv2D(64, 5, strides=2, padding='same', activation='relu'))
    # BN6 Layer
    #model.add(BatchNormalization())
    # Droupout2
    model.add(Dropout(0.4))

    # Flatten Layer
    model.add(Flatten())
    # Dense Layer
    model.add(Dense(128, 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    # BN7 Layer
    #model.add(BatchNormalization())
    # Droupout3
    model.add(Dropout(0.4))
    # Output Layer
    model.add(Dense(10, 'softmax',  kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    
    opt = tf.keras.optimizers.SGD(learning_rate=lr)
    #opt = tf.keras.optimizers.Adam(learning_rate=lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(opt, loss_fn, metrics=['accuracy'])
    return model

def construct_model(trainer_type, lr=0.003):
    if trainer_type == 'fedavg':
        return _construct_client_model(lr)
    else:
        return _construct_client_model(lr)