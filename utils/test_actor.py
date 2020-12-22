from collections import Counter

from tensorflow.python.keras import metrics
from utils.read_data import read_federated_data
from flearn.actor import Actor
from flearn.server import Server
from flearn.client import Client
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input
from tensorflow.keras import Model, Sequential
import numpy as np

print(__name__)

def construct_model():
    model = Sequential()
    # Input Layer
    model.add(Input(shape=(784,)))
    # Hidden Layer
    #model.add(Dense(128, 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(Dense(16, 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))) #debug
    # Output Layer
    model.add(Dense(10, 'softmax', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    
    opt = tf.keras.optimizers.SGD(learning_rate=0.03)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(opt, loss_fn, metrics=['accuracy'])

    return model

def main():
    workspace_path = '/home/lab/workspace/SplitGP/data'
    clients, train_data, test_data = read_federated_data('mnist')
    id = 'f_00000'
    #print(clients[0])
    #print(len(train_data[id]['x'][0]))

    model = construct_model()

    #ator = Actor(id, 'client', train_data[id], test_data[id], model)
    #num_samples, updates = ator.solve_inner()
    client = Client(id, train_data[id], test_data[id], model=model)
    num_samples, acc, loss, updates = client.train()
    num, test_acc, test_loss = client.test()
    print('num_samples', num)
    print(test_acc, test_loss)

main()
