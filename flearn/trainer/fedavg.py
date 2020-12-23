import numpy as np
import importlib
import tensorflow as tf
import random
import time
from utils.read_data import read_federated_data
from utils.trainer_utils import TrainConfig
#from flearn.model.mlp import construct_model
from flearn.server import Server
from flearn.client import Client

class FedAvg(object):
    def __init__(self, train_config):
        # Transfer trainer config to self, we save the configurations by this trick
        for key, val in train_config.trainer_config.items(): 
            setattr(self, key, val)
        # Get the config of client
        self.client_config = train_config.client_config
        # Evaluate model on all clients or on this server
        self.eval_locally = True

        # Set the random set
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Construct the actors
        self.clients = None
        self.construct_actors()

    def construct_actors(self):
        # 1, Read dataset
        clients, train_data, test_data = read_federated_data(self.dataset)

        # 2, Get model loader according to dataset and model name and construct the model
        # Set the model loader according to the dataset and model name
        model_path = 'flearn.model.%s.%s' % (self.dataset, self.model)
        self.model_loader = importlib.import_module(model_path).construct_model
        # Construct the model
        client_model = self.model_loader('fedavg', self.client_config['learning_rate'])

        # 3, Construct server
        self.server = Server(model=client_model)

        # 4, Construct clients and set their uplink
        self.clients = [Client(id, self.client_config, train_data[id], test_data[id], 
                        uplink=[self.server], model=client_model) for id in clients]

        # 5, Set the downlink of server
        self.server.add_downlink(self.clients)

        # 6*, We can evaluate model on server to speed the testing,
        # We need construct a total test dataset of server
        if self.eval_locally == True:
            server_test_data = {'x':[], 'y':[]}
            for c in clients:
                server_test_data['x'].append(test_data[c]['x'])
                server_test_data['y'].append(test_data[c]['y'])
            self.server.test_data['x'] = np.vstack(server_test_data['x'])
            self.server.test_data['y'] = np.hstack(server_test_data['y'])

    def train(self):
        for round in range(self.num_rounds):
            # 0, Init time record
            train_time, test_time, agg_time = 0, 0, 0

            # 1, Random select clients
            selected_clients = self.select_clients(round)
            
            # 2, Train selected clients
            start_time = time.time()
            train_results = self.server.train(selected_clients)
            train_time = time.time() - start_time
            if train_results == None:
                continue
            
            # 3, Get model updates (list) and number of samples (list) of clients
            nks = [rest[1] for rest in train_results] # -> list
            updates = [rest[4] for rest in train_results] # -> list
            
            # 4, Aggregate these client acoording to number of samples (FedAvg)
            start_time = time.time()
            agg_updates = self.federated_averaging_aggregate(updates, nks)
            agg_time = time.time() - start_time
            
            # 5, Apply update to the global model. All clients and sever share
            # the same model instance, so we just apply update to server and refresh
            # the latest_params and lastest_updates for all clients.
            self.server.apply_update(agg_updates)
            for c in self.server.downlink:
                c.latest_params = self.server.latest_params
                c.latest_updates = agg_updates

            # 6, Test the model every eval_every round
            if round % self.eval_every == 0:
                start_time = time.time()
                
                if self.eval_locally == False:
                    # Test model on all clients,
                    test_results = self.server.test()
                else:
                    # OR Test model on the server (Faster)
                    test_samples, test_acc, test_loss = self.server.test_locally()
                    test_results = [[self.server, test_samples, test_acc, test_loss]]

                test_time = time.time() - start_time
                # Summary this test
                self.summary_results(round, test_results=test_results)

            # 7, Summary this round of training
            self.summary_results(round, train_results=train_results)

            # 8, Print the train, aggregate, test time
            print(f'Round: {round}, Training time: {train_time}, Test time: {test_time}, Aggregate time: {agg_time}.')
    
    def select_clients(self, round, num_clients=20):
        '''selects num_clients clients weighted by number of samples from possible_clients
        
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))
        
        Return:
            list of selected clients objects
        '''

        num_clients = min(num_clients, len(self.clients))
        random.seed(round+self.seed)  # make sure for each comparison, we are selecting the same clients each round
        selected_clients = random.sample(self.clients, num_clients)
        random.seed(self.seed) # Restore the seed
        return selected_clients
    
    def federated_averaging_aggregate(self, updates, nks):
        return self.weighted_aggregate(updates, nks)


    def weighted_aggregate(self, updates, weights):
        # Aggregate the updates according their weights
        normalws = np.array(weights, dtype=float) / np.sum(weights, dtype=np.float)
        num_clients = len(updates)
        num_layers = len(updates[0])
        # Shape=(num_clients, num_layers, num_params)
        # np_updates = np.array(updates, dtype=float).reshape(num_clients, num_layers, -1)
        agg_updates = []
        for la in range(num_layers):
            agg_updates.append(np.sum([up[la]*pro for up, pro in zip(updates, normalws)], axis=0))
        
        # np_agg_updates = np.sum(np_updates*normalws, axis=0) #-> shape=(num_layers, num_params)
        # Convert numpy array to list of array format (keras weights format)
        #agg_updates = [np_agg_updates[i] for i in range(num_layers)]

        return agg_updates # -> list

    def summary_results(self, round, train_results=None, test_results=None):
        def _calculate_weighted_metric(metrics, nks):
            normalws = np.array(nks) / np.sum(nks, dtype=np.float)
            metric = np.sum(metrics*normalws)
            return metric

        if train_results:
            nks = [rest[1] for rest in train_results]
            train_accs = [rest[2] for rest in train_results]
            train_losses = [rest[3] for rest in train_results]
            weighted_train_acc = _calculate_weighted_metric(train_accs, nks)
            weighted_train_loss = _calculate_weighted_metric(train_losses, nks)
            print(f'Round {round}, Train ACC: {weighted_train_acc}, Train Loss: {weighted_train_loss}')
            return weighted_train_acc, weighted_train_loss
        if test_results:
            nks = [rest[1] for rest in test_results]
            test_accs = [rest[2] for rest in test_results]
            test_losses = [rest[3] for rest in test_results]
            weighted_test_acc = _calculate_weighted_metric(test_accs, nks)
            weighted_test_loss = _calculate_weighted_metric(test_losses, nks)
            print(f'Round {round}, Test ACC: {weighted_test_acc}, Test Loss: {weighted_test_loss}')
            return weighted_test_acc, weighted_test_loss