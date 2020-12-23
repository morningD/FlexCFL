import numpy as np
from flearn.actor import Actor

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

    def check_selected_trainable(self, selected_nodes):
        ''' 
        Check The selected nodes whether can be trained 
        '''
        nodes_trainable = False
        for node in selected_nodes:
            if node in self.downlink:
                if node.check_trainable() == True:
                    nodes_trainable = True
                    break
        return nodes_trainable

    def train(self, selected_nodes):
        '''
        Train on downlink actors like groups and clients
        Return:
            results: 
                list of list of training results ->[[result1], [result2], [result3], ...]
        '''
        if self.check_selected_trainable(selected_nodes) == True:
            results = []
            for node in selected_nodes:
                num_samples, train_acc, train_loss, updates = node.train()
                results.append([node, num_samples, train_acc, train_loss, updates])
            return results
        else:
            print('ERROR: This server has not training clients/groups with training data/clients')
            return

    def test(self):
        '''
        Test on all downlink actors
        '''
        self.check_testable() # Redundant check
        if self.testable == True:
            results = []
            for node in self.downlink:
                num_samples, test_acc, test_loss = node.test()
                results.append([node, num_samples, test_acc, test_loss])
        else:
            print('ERROR: This server has not test clients/groups with testing data/clients')
            return