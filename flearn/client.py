import numpy as np
import tensorflow as tf
from flearn.actor import Actor

'''
Define the client of federated learning framework
'''

class Client(Actor):
    def __init__(self, id, config, train_data={'x':[],'y':[]}, test_data={'x':[],'y':[]}, uplink=[], model=None):
        actor_type = 'client'
        super(Client, self).__init__(id, actor_type, train_data, test_data, model)
        if len(uplink) > 0:
            self.add_uplink(uplink)
        self.clustering = False # Is the client join the clustering proceudre.
        self.difference = [] # tuple of (group, diff) # Record the discrepancy between group and client
        
        # transfer client config to self
        for key, val in config.items(): 
            setattr(self, key, val)

        self.check_trainable()
        self.check_testable()     

    # The client is the end point of FL framework 
    def has_downlink(self):
        return False

    def check_trainable(self):
        if self.train_data['y'].shape[0] > 0:
            self.trainable = True
            # The train size of client is the size of the local training dataset
            self.train_size = self.train_data['y'].shape[0]
        return self.trainable
    
    def check_testable(self):
        if self.test_data['y'].shape[0] > 0:
            self.testable = True
            self.test_size = self.test_data['y'].shape[0]
        return self.testable

    def train(self):
        ''' 
        Train on local training dataset.
        Params:
            num_epoch: The number of local epoch
        Return: 
            num_sampes: number of training samples
            acc = training accuracy of last local epoch
            loss = mean training loss of last local epoch
            updates = update of weights
        '''
        self.check_trainable()
        num_samples, acc, loss, update = self.solve_inner(self.local_epochs, self.batch_size)
        return num_samples, acc[-1], loss[-1], update

    def test(self):
        '''
        Test on local test dataset
        Return:
            num_samples: number of testing samples
            acc = test accuracy
            loss = test loss
        '''
        self.check_testable()
        return self.test_locally()