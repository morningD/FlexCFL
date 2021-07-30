import numpy as np
import tensorflow as tf
from flearn.actor import Actor
from utils.trainer_utils import process_grad, calculate_cosine_dissimilarity

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
        self.discrepancy = 0 # The discrepancy between this client and its first uplink node
        # The cosine dissimilarity between this client and its first uplink node
        # Cosine Dissimilarity, definition: (1-cosine) / 2
        self.cosine_dissimilarity = 0 
        self.label_array = None

        # transfer client config to self
        for key, val in config.items(): 
            setattr(self, key, val)

        self.max_temp = self.temperature # Save the max temperature
        self.original_train_data = {'x': np.copy(self.train_data['x']), 'y': np.copy(self.train_data['y'])}

        self.refresh()   

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
        return num_samples, acc[-1], loss[-1], soln, update

    def test(self, from_uplink=False):
        '''
        Test on local test dataset
        Argument: from_uplink indicates the evalutation is based on the model of first uplink node.
        if from_uplink=False, the test is based on its latest_params.
        Return:
            num_samples: number of testing samples
            acc = test accuracy
            loss = test loss
        '''
        self.check_testable()
        if from_uplink == True:
            if len(self.uplink) == 0: 
                print(f'Warning: Node {self.id} does not have an uplink model for testing.')
                return 0, 0, 0
            
            # Temporarily set client's latest_params to first uplink's latest_params
            backup_params = self.latest_params
            self.latest_params = self.uplink[0].latest_params
            test_result = self.test_locally()
            # Reset client's latest_params
            self.latest_params = backup_params
        else:
            # Test on client's params
            test_result = self.test_locally()
        return test_result

    ''' Pretrain this client based on model_params,
        Note: the latest_params and latest_updates will not be modified.
        The latest_soln and latest_gradient wil be update.
    '''
    def pretrain(self, model_params, iterations=20):
        backup_params = self.latest_params
        self.latest_params = model_params
        num_samples, acc, loss, soln, update = self.solve_iters(iterations, self.batch_size, pretrain=True)
        #num_samples, acc, loss, soln, update = self.solve_inner(1, self.batch_size, pretrain=True)
        
        # Restore latest_params after training
        self.latest_params = backup_params

        return num_samples, acc[-1], loss[-1], soln, update

    # Update the discrepancy and cosine dissimilarity
    def update_difference(self):
        def _calculate_l2_distance(m1, m2):
            v1, v2 = process_grad(m1), process_grad(m2)
            l2d = np.sum((v1-v2)**2)**0.5
            return l2d

        # Only calcuate the discrepancy between this client and first uplink    
        # we use self.local_soln istead of self.latest_params, more safe?
        self.discrepancy = _calculate_l2_distance(self.local_soln, self.uplink[0].latest_params)
        self.cosine_dissimilarity = calculate_cosine_dissimilarity(self.local_gradient, self.uplink[0].latest_updates)
        return

    def refresh(self):
        self.check_trainable()
        self.check_testable()

        self.label_array = np.intersect1d(self.train_data['y'], self.test_data['y'])
        return