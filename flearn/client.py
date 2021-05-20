import numpy as np
import tensorflow as tf
from flearn.actor import Actor
from utils.trainer_utils import process_grad

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

    # The client is trainable if it has local dataset
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
            None
        Return: 
            num_sampes: number of training samples
            acc = training accuracy of last local epoch
            loss = mean training loss of last local epoch
            updates = update of weights
        '''
        self.check_trainable()
        num_samples, acc, loss, soln, update = self.solve_inner(self.local_epochs, self.batch_size)
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

    def pretrain(self, iterations=50):

        num_samples, acc, loss, soln, update = self.solve_iters(iterations, self.batch_size)
        #num_samples, acc, loss, soln, update = self.solve_inner(1, self.batch_size)

        return num_samples, acc[-1], loss[-1], soln, update

    def update_difference(self):
        def _calculate_l2_distance(m1, m2):
            v1, v2 = process_grad(m1), process_grad(m2)
            l2d = np.sum((v1-v2)**2)**0.5
            return l2d

        self.difference.clear()
        # Only calcuate the discrepancy between this client and first uplink    
        diff = _calculate_l2_distance(self.local_soln, self.uplink[0].latest_params)
        self.difference.append(diff)
        return