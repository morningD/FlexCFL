import numpy as np
import importlib
import tensorflow as tf
import random
import time
from termcolor import colored

from utils.read_data import read_federated_data
from utils.trainer_utils import TrainConfig
#from flearn.model.mlp import construct_model
from flearn.server import Server
from flearn.client import Client
from utils.export_result import ResultWriter

class FedAvg(object):
    def __init__(self, train_config):
        # Transfer trainer config to self, we save the configurations by this trick
        for key, val in train_config.trainer_config.items(): 
            setattr(self, key, val)
        self.trainer_type = train_config.trainer_type
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

        # Create results writer
        self.writer = ResultWriter(train_config)

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
        for comm_round in range(self.num_rounds):

            # 0, Init time record
            train_time, test_time, agg_time = 0, 0, 0

            # 1, Random select clients
            selected_clients = self.select_clients(comm_round)
            #selected_clients = self.clients[:20] # DEBUG, only use first 20 clients to train
            
            # 2, The server boardcasts the model to clients
            for c in selected_clients:
                c.latest_params = self.server.latest_params

            # 3, Train selected clients
            start_time = time.time()
            train_results = self.server.train(selected_clients)
            train_time = round(time.time() - start_time, 3)
            if train_results == None:
                continue
            
            # 4, Summary this round of training
            num_train_clients, weighted_train_acc, weighted_train_loss = self.summary_results(comm_round, train_results=train_results)

            # 5, Get model updates (list) and number of samples (list) of clients
            nks = [rest[1] for rest in train_results] # -> list
            updates = [rest[4] for rest in train_results] # -> list
            
            # 6, Aggregate these client acoording to number of samples (FedAvg)
            start_time = time.time()
            agg_updates = self.federated_averaging_aggregate(updates, nks)
            agg_time = round(time.time() - start_time, 3)
            
            # 7, Apply update to the global model. All clients and sever share
            # the same model instance, so we just apply update to server and refresh
            # the latest_params and lastest_updates for selected clients. 
            # Calculate the discrepancy between the global model and client model 
            self.server.apply_update(agg_updates)
            for c in selected_clients:
                c.latest_params = self.server.latest_params
                c.latest_updates = agg_updates
                c.update_difference()

            # 8, Test the model every eval_every round and the last round
            if comm_round % self.eval_every == 0 or comm_round == self.num_rounds-1:
                start_time = time.time()
                
                if self.eval_locally == False:
                    # Test model on all clients,
                    #test_results = self.server.test()
                    test_results = self.server.test(selected_clients)
                else:
                    # OR Test model on the server (Faster)
                    test_samples, test_acc, test_loss = self.server.test_locally()
                    test_results = [[self.server, test_samples, test_acc, test_loss]]

                test_time = round(time.time() - start_time, 3)
                # Summary this test
                _, weighted_test_acc, weighted_test_loss = self.summary_results(comm_round, test_results=test_results)
                # Write this evalution result ot file
                self.writer.write_row(comm_round, [weighted_test_acc, weighted_train_acc, weighted_train_loss, \
                    num_train_clients, self.calculate_mean_discrepancy(selected_clients)])

            # 9, Print the train, aggregate, test time
            print(f'Round: {comm_round}, Training time: {train_time}, Test time: {test_time}, Aggregate time: {agg_time}')
    
    def select_clients(self, comm_round, num_clients=20):
        '''selects num_clients clients weighted by number of samples from possible_clients
        
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))
        
        Return:
            list of selected clients objects
        '''

        num_clients = min(num_clients, len(self.clients))
        random.seed(comm_round+self.seed)  # make sure for each comparison, we are selecting the same clients each round
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

    def summary_results(self, comm_round, train_results=None, test_results=None):

        if train_results:
            nks = [rest[1] for rest in train_results]
            num_clients = len(nks)
            train_accs = [rest[2] for rest in train_results]
            train_losses = [rest[3] for rest in train_results]
            weighted_train_acc = np.average(train_accs, weights=nks)
            weighted_train_loss = np.average(train_losses, weights=nks)
            print(colored(f'Round {comm_round}, Train ACC: {round(weighted_train_acc, 4)},\
                Train Loss: {round(weighted_train_loss, 4)}', 'blue', attrs=['reverse']))
            return num_clients, weighted_train_acc, weighted_train_loss
        if test_results:
            nks = [rest[1] for rest in test_results]
            num_clients = len(nks)
            test_accs = [rest[2] for rest in test_results]
            test_losses = [rest[3] for rest in test_results]
            weighted_test_acc = np.average(test_accs, weights=nks)
            weighted_test_loss = np.average(test_losses, weights=nks)
            print(colored(f'Round {comm_round}, Test ACC: {round(weighted_test_acc, 4)},\
                Test Loss: {round(weighted_test_loss, 4)}', 'red', attrs=['reverse']))
            return num_clients, weighted_test_acc, weighted_test_loss

    def train_locally(self, num_epoch=20, batch_size=10):
        """
            We can train and test model on server for comparsion or debugging reseason
        """
        # 1, We collect all data into server
        print("Collect data.....")
        server_test_data = {'x':[], 'y':[]}
        server_train_data = {'x':[], 'y':[]}
        for c in self.clients:
            server_test_data['x'].append(c.test_data['x'])
            server_test_data['y'].append(c.test_data['y'])
            server_train_data['x'].append(c.train_data['x'])
            server_train_data['y'].append(c.train_data['y'])
        self.server.test_data['x'] = np.vstack(server_test_data['x'])
        self.server.test_data['y'] = np.hstack(server_test_data['y'])
        self.server.train_data['x'] = np.vstack(server_train_data['x'])
        self.server.train_data['y'] = np.hstack(server_train_data['y'])

        self.server.model.summary()

        # 2, Server train locally
        train_size, train_acc, train_loss, update = self.server.solve_inner(num_epoch, batch_size)
        # 3, Server Apply update
        self.server.apply_update(update)
        # 4, Server test locally
        test_size, test_acc, test_loss = self.server.test_locally()

        # 5, Print result, we show the accuracy and loss of all training epochs
        print(f"Train size: {train_size} Train ACC: {[round(acc, 4) for acc in train_acc]} \
             Train Loss: {[round(loss, 4) for loss in train_loss]}")
        print(colored(f"Test size: {test_size}, Test ACC: {round(test_acc, 4)}, \
            Test Loss: {round(test_loss, 4)}", 'red', attrs=['reverse']))

    def calculate_mean_discrepancy(self, clients):
        discrepancy = [c.difference for c in clients]
        return np.mean(discrepancy)