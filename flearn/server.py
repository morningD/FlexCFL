import numpy as np
from flearn.actor import Actor
import tensorflow as tf

'''
Define the server of federated learning framework
'''

class Server(Actor):
    def __init__(self, train_data={'x':[],'y':[]}, test_data={'x':[],'y':[]}, downlink=[], model=None):
        actor_type = 'server'
        id = 0
        super(Server, self).__init__(id, actor_type, train_data, test_data, model)
        if len(downlink) > 0:
            self.add_downlink(downlink)
        # We need refresh model attribute like 
        # trainable, train_size, testable, test_size after modify the downlink  
        self.refresh()

    # The server is the top node of FL framework
    def has_uplink(self):
        return False

    # The server is trainable if it's downlink nodes are trainable
    def check_trainable(self):
        '''
        Check the server whether can be trained and refresh the train size
        '''
        if self.has_downlink():
            self.train_size = 0
            self.trainable = False
            for node in self.downlink:
                if node.check_trainable() == True:
                    self.trainable = True
                    # Refresh the train size of server,
                    # It is the sum of the train size of all trainable donwlink nodes
                    self.train_size += node.train_size
        return self.trainable

    def check_testable(self):
        if self.has_downlink():
            self.test_size = 0
            self.testable = False
            for nodes in self.downlink:
                if nodes.check_testable() == True:
                    self.testable = True
                    self.test_size += nodes.test_size
        return self.testable

    def refresh(self):
        '''
        The server should be refreshed after add/delete downlink
        '''
        self.check_trainable()
        self.check_testable()

    def add_downlink(self, nodes):
        super(Server, self).add_downlink(nodes)
        # Refresh the server
        self.refresh()

    def delete_downlink(self, nodes):
        super(Server, self).delete_downlink(nodes)
        # Refresh the server
        self.refresh()

    def train(self, selected_nodes):
        '''
        Train on downlink actors like groups and clients
        Params:
            selected_nodes: Train the selected clients.
        Return:
            results: 
                list of list of training results ->[[result1], [result2], [result3], ...]
        '''
        results = []
        
        if self.downlink[0].actor_type == 'client': # i.e. FedAvg
            # Check the trainable of selected clients
            trainable, valid_nodes = self.check_selected_trainable(selected_nodes)
            if trainable == True:
                for node in valid_nodes:
                    num_samples, train_acc, train_loss, soln, update = node.train()
                    results.append([node, num_samples, train_acc, train_loss, update])
        
        elif self.downlink[0].actor_type == 'group': # i.e. FedGroup
            # Check the trainable of all groups
            trainable, valid_nodes = self.check_selected_trainable(self.downlink)
            if trainable == True:
                for group in valid_nodes:
                    # The server will not boardcast the model
                    group_num_samples, group_train_acc, group_train_loss, soln, group_update = group.train(selected_nodes)
                    results.append([group, group_num_samples, group_train_acc, group_train_loss, group_update])
        
        if results == []:
            print('ERROR: This server has not training clients/groups with training data/clients')
            return
        return results
            

    def test(self, selected_nodes):
        '''
        Test on selected nodes
        '''
        testable, valid_nodes = self.check_selected_testable(selected_nodes)
        if testable == True:
            results = []
            for node in valid_nodes:
                num_samples, test_acc, test_loss = node.test()
                results.append([node, num_samples, test_acc, test_loss])
            return results
        else:
            print('ERROR: This server has not test clients/groups with testing data/clients')
            return