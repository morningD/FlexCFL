from flearn.actor import Actor
import numpy as np
from math import floor

'''
Define the group of federated learning framework, 
<Group> is similar with <Server>
'''

class Group(Actor):
    def __init__(self, id, config, train_data={'x':[],'y':[]}, test_data={'x':[],'y':[]}, uplink=[], model=None):
        actor_type = 'group'
        super(Group, self).__init__(id, actor_type, train_data, test_data, model=model)
        if len(uplink) > 0:
            self.add_uplink(uplink)

        # transfer client config to self
        for key, val in config.items(): 
            setattr(self, key, val)

        self.discrepancy = 0 # The mean discrepancy between this group and sublink node
        self.cosine_dissimilarity = 0 # The mean cosine dissimilarity between this group and sublink node

        self.opt_updates = None
        self.aggregation_strategy = 'fedavg'

    # The group is trainable if it's downlink nodes are trainable
    def check_trainable(self):
        '''
        Check the group whether can be trained and refresh the train size
        '''
        self.trainable = False
        if self.has_downlink():
            self.train_size = 0
            for node in self.downlink:
                if node.check_trainable() == True:
                    self.trainable = True
                    # Refresh the train size of server,
                    # It is the sum of the train size of all trainable donwlink nodes
                    self.train_size += node.train_size
        return self.trainable

    def check_testable(self):
        self.testable = False
        if self.has_downlink():
            self.test_size = 0
            for nodes in self.downlink:
                if nodes.check_testable() == True:
                    self.testable = True
                    self.test_size += nodes.test_size
        return self.testable

    def refresh(self):
        '''
        The Group should be refreshed after add/delete/clear downlink
        '''
        self.check_trainable()
        self.check_testable()
        # Refresh the local test set
        if self.eval_locally == True:
            if self.downlink:
                group_test_data = {'x':[], 'y':[]}
                for c in self.downlink:
                    group_test_data['x'].append(c.test_data['x'])
                    group_test_data['y'].append(c.test_data['y'])
                self.test_data['x'] = np.vstack(group_test_data['x'])
                self.test_data['y'] = np.hstack(group_test_data['y'])
            else:
                self.test_data = {'x':[], 'y':[]}
        return

    def add_downlink(self, nodes):
        super(Group, self).add_downlink(nodes)
        self.refresh()

    def delete_downlink(self, nodes):
        super(Group, self).delete_downlink(nodes)
        self.refresh()

    def clear_downlink(self):
        super(Group, self).clear_downlink()
        self.refresh()
    
    ''' The aggregatation algorithm of group is fedavg
    '''
    def federated_averaging_aggregate(self, updates, nks):
        return self.weighted_aggregate(updates, nks)

    """ Aggregate client updates according to their sample size and temperatrues """
    def federated_averaging_aggregate_with_temperature(self, updates, nks, temps, max_temp):
        if len(temps) == 0: 
            return [np.zeros_like(ws) for ws in self.latest_params]
        else:
            temp_nks, epsilon = [], 1e-5
            for nk, temp in zip(nks, temps):
                if temp == None:
                    temp_nks.append(nk)
                else:
                    # Prevent divided by 0
                    #print('debug: group.py:100', temp, max_temp, nk)
                    temp_nks.append(floor((max(temp, 0) / (max_temp+epsilon)) * nk))
            return self.federated_averaging_aggregate(updates, temp_nks)

    def weighted_aggregate(self, updates, weights):
        # Aggregate the updates according their weights
        epsilon = 1e-5 # Prevent divided by 0
        normalws = np.array(weights, dtype=float) / (np.sum(weights, dtype=np.float) + epsilon)
        num_layers = len(updates[0])
        agg_updates = []
        for la in range(num_layers):
            agg_updates.append(np.sum([up[la]*pro for up, pro in zip(updates, normalws)], axis=0))

        return agg_updates # -> list

    def _calculate_weighted_metric(metrics, nks):
            normalws = np.array(nks) / np.sum(nks, dtype=np.float)
            metric = np.sum(metrics*normalws)
            return metric

    '''
        The train procedure of group contains the aggreagation of clients' update
    '''
    def train(self, selected_nodes=None):
        '''
        Train on selected downlink clients and aggregate these updates,
        Default train on all downlink client.
        Return:
            results: 
                list of list of training results ->[[result1], [result2], [result3], ...]
        '''
        # Group may be empty, we skip the training if this group is empty
        if len(self.downlink) == 0:
            print(f"Warning: Group {self.id} is empty.")
            return 0, 0, 0, None

        # Default train on all downlink client if there has any selected nodes.
        if not selected_nodes: selected_nodes = self.downlink

        # Check the trainable of selected nodes, those nodes not in the group's downlink are invalid.
        trainable, valid_nodes = self.check_selected_trainable(selected_nodes)
        
        # 0, Begin training
        if trainable == True:
            train_results = []
            group_params = self.latest_params

            # 1, Broadcast group's model to client
            for node in valid_nodes:
                # Calculate the latest updates of clients
                node.latest_updates = [(w1-w0) for w0, w1 in zip(node.latest_params, group_params)]
                node.latest_params = group_params
            
            # 2, Train the neural model of client and save the results
            for node in valid_nodes:
                num_samples, train_acc, train_loss, soln, update = node.train()
                train_results.append([node, num_samples, train_acc, train_loss, update])
            
            # 3, Aggregate the clients using FedAvg
            nks = [rest[1] for rest in train_results] # -> list
            updates = [rest[4] for rest in train_results] # -> list
            temps = [rest[0].temperature for rest in train_results]
            max_temp = train_results[0][0].max_temp
            if self.aggregation_strategy == 'temp' and max_temp is not None:
                agg_updates = self.federated_averaging_aggregate_with_temperature(updates, nks, temps, max_temp)
            if self.aggregation_strategy == 'fedavg':
                agg_updates = self.federated_averaging_aggregate(updates, nks)
            if self.aggregation_strategy == 'avg':
                agg_updates = self.federated_averaging_aggregate(updates, [1.0*len(nks)])

            # 4, Refresh the latest parameter and update of group, the global model instance will not change.
            self.fresh_latest_params_updates(agg_updates)
            
            """
            # (Optional) Refresh the latest_parameter and update of all downlink clients if this group using consensus policy
            # Otherwise， only update the nodes of this round.
            target_refresh_nodes = self.downlink if self.consensus == True else valid_nodes
            for node in target_refresh_nodes:
                node.latest_params = self.latest_params
                node.latest_updates = agg_updates
            """

            # 5, Summary the train result of group then return
            group_num_samples = np.sum(nks, dtype=np.float)
            group_train_acc = np.average([rest[2] for rest in train_results], weights=nks)
            group_train_loss = np.average([rest[3] for rest in train_results], weights=nks)

            return group_num_samples, group_train_acc, group_train_loss, self.latest_params, agg_updates

        elif self.allow_empty == True:
            group_num_samples, group_train_acc, group_train_loss, update = 0, 0, 0, None
            return group_num_samples, group_train_acc, group_train_loss, self.latest_params, update
        else:
            print(f'ERROR: Group {self.id} has not any valid training clients with training data which is invalid.')
            return

    '''
    Test all clients in the downlink
    '''
    def test(self):
        if len(self.downlink) == 0:
            print(f"Warning: Group {self.id} is empty.")
            return 0, 0, 0
        
        testable, valid_nodes = self.check_selected_testable(self.downlink)
        if testable == False:
            print(f'Warning: Group {self.id} has not test data.')
            return 0, 0, 0

        if self.eval_locally == False:
            # Test on all clients
            test_results = [node.test() for node in valid_nodes]
            # Summary the test result
            nks = [rest[0] for rest in test_results]
            group_num_samples = np.sum(nks, dtype=np.float)
            group_test_acc = np.average([rest[1] for rest in test_results], weights=nks)
            group_test_loss = np.average([rest[2] for rest in test_results], weights=nks)
        else:
            # Test on local test set (Faster)
            group_num_samples, group_test_acc, group_test_loss = self.test_locally()
        
        return group_num_samples, group_test_acc, group_test_loss