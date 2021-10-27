import numpy as np
import importlib
import tensorflow as tf
import random
import time
from termcolor import colored
from math import ceil

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
        model_path = 'flearn.model.%s.%s' % (self.dataset.split('_')[0], self.model)
        self.model_loader = importlib.import_module(model_path).construct_model
        # Construct the model
        client_model = self.model_loader('fedavg', self.client_config['learning_rate'])
        # *Print the summary of model
        client_model.summary()

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
            
            # * Change the clients's data distribution
            self.data_distribution_shift(comm_round, self.clients, self.shift_type, self.swap_p)

            # 2, The server boardcasts the model to clients
            for c in selected_clients:
                # The selected client calculate the latest_update (This may be many rounds apart)
                c.latest_updates = [(w1-w0) for w0, w1 in zip(c.latest_params, self.server.latest_params)]
                # Broadcast the gloabl model to selected clients
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
            # Set the training model to the new server model, however this step is not important
            self.server.set_params(self.server.latest_params)

            for c in selected_clients:
                ''' The latest_params and updates will be refreshed next time they are selected.
                c.latest_params = self.server.latest_params
                c.latest_updates = agg_updates
                '''
                c.update_difference() # Based on latest_soln and latest_gradient

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

    def train_locally(self, num_epoch=30, batch_size=10):
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

        #self.server.model.summary()

        # 2, Server train locally
        train_size, train_acc, train_loss, _, update = self.server.solve_inner(num_epoch, batch_size)
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
        discrepancy = [c.discrepancy for c in clients]
        return np.mean(discrepancy)

    """ This function will randomly swap all clients' data with probability swap_p
    """
    def swap_data(self, clients, swap_p, scope='all'):

        # Swap the data of all clients with probability swap_p
        clients_size = len(clients)
        # Randomly swap two clients' dataset
        if swap_p > 0 and swap_p < 1:
            # Shuffle the client index
            shuffle_idx = np.random.permutation(clients_size)
            swap_flag = np.random.choice([0,1], int(clients_size/2), p=[1-swap_p, swap_p]) # Half size of clients_size
            for idx in np.nonzero(swap_flag)[0]:
                # Swap clients' data with index are idx and -(idx+1)
                cidx1, cidx2 = shuffle_idx[idx], shuffle_idx[-(idx+1)]
                c1, c2 = clients[cidx1], clients[cidx2]
                g1, g2 = c1.uplink[0], c2.uplink[0]
                
                # Swap train data and test data
                if scope == 'all':
                    c1.distribution_shift, c2.distribution_shift = True, True
                    c1.train_data, c2.train_data = c2.train_data, c1.train_data
                    c1.test_data, c2.test_data = c2.test_data, c1.test_data
                    print(colored(f"Swap C-{c1.id}@G{g1.id} and C-{c2.id}@G{g2.id} data", 'cyan', attrs=['reverse']))

                if scope == 'part':
                    if len(c1.label_array) == 0 or len(c2.label_array) == 0: return
                    c1_diff, c2_diff = np.setdiff1d(c1.label_array, c2.label_array, True), \
                        np.setdiff1d(c2.label_array, c1.label_array, True)
                    if c1_diff.size == 0 or c2_diff.size == 0: return
                    c1_swap_label, c2_swap_label = np.random.choice(c1_diff, 1)[0], np.random.choice(c2_diff, 1)[0]
                    c1.distribution_shift, c2.distribution_shift = True, True
                    '''
                    print('Debug', np.unique(c1.train_data['y']), np.unique(c1.test_data['y']))
                    print('Debug', np.unique(c2.train_data['y']), np.unique(c2.test_data['y']))
                    print(c1_swap_label, c2_swap_label)
                    '''

                    for c1_data, c2_data in zip([c1.train_data, c1.test_data], [c2.train_data, c2.test_data]):
                        label_idx1 = np.where(c1_data['y'] == c1_swap_label)[0]
                        label_idx2 = np.where(c2_data['y'] == c2_swap_label)[0]
                        c1_swap_x, c2_swap_x = c1_data['x'][label_idx1], c2_data['x'][label_idx2]
                        c1_swap_y, c2_swap_y = c1_data['y'][label_idx1], c2_data['y'][label_idx2]
                        
                        # Swap the feature
                        c1_data['x'] = np.delete(c1_data['x'], label_idx1, axis=0)
                        c1_data['x'] = np.vstack([c1_data['x'], c2_swap_x])
                        c2_data['x'] = np.delete(c2_data['x'], label_idx2, axis=0)
                        c2_data['x'] = np.vstack([c2_data['x'], c1_swap_x])

                        # Swap the label
                        c1_data['y'] = np.delete(c1_data['y'], label_idx1)
                        c1_data['y'] = np.hstack([c1_data['y'], c2_swap_y])
                        c2_data['y'] = np.delete(c2_data['y'], label_idx2)
                        c2_data['y'] = np.hstack([c2_data['y'], c1_swap_y])

                        # Shuffle the data
                        random_idx1, random_idx2 = np.arange(c1_data['y'].shape[0]), np.arange(c2_data['y'].shape[0])
                        np.random.shuffle(random_idx1), np.random.shuffle(random_idx2)
                        c1_data['x'], c1_data['y'] = c1_data['x'][random_idx1], c1_data['y'][random_idx1]
                        c2_data['x'], c2_data['y'] = c2_data['x'][random_idx2], c2_data['y'][random_idx2]

                    print(colored(f"Swap C-{c1.id}@G{g1.id}-L{int(c1_swap_label)} and C-{c2.id}@G{g2.id}-L{int(c2_swap_label)} data", \
                        'cyan', attrs=['reverse']))

                # Refresh client and group
                _,_,_ = c1.refresh(), c2.refresh(), g1.refresh()
                if g2 != g1: g2.refresh()
        return

    def increase_data(self, round, clients):
        processing_round = [0, 50, 100, 150]
        rate = [1/4, 1/2, 3/4, 1.0]
        
        if round == 0:
            self.shuffle_index_dict = {}
            # Shuffle the train data
            for c in clients:
                cidx = np.arange(c.train_data['y'].size)
                np.random.shuffle(cidx)
                self.shuffle_index_dict[c] = cidx
        
        if round in processing_round:
            release_rate = rate[processing_round.index(round)]
            print('>Round {:3d}, {:.1%} training data release.'.format(round, release_rate))
            for c in clients:
                # Calculate new train size
                train_size = ceil(c.train_data['y'].size * release_rate)
                release_index = self.shuffle_index_dict[c][:train_size]
                c.train_data['x'] = c.original_train_data['x'][release_index]
                c.train_data['y'] = c.original_train_data['y'][release_index]

                c.refresh()
                if c.has_uplink(): c.uplink[0].refresh()
        return

    def data_distribution_shift(self, round, clients, shift_type=None, swap_p=0):
        if shift_type == None:
            return

        if shift_type == 'increment':
            self.increase_data(round, clients)
        else:       
            if len(clients) == 0: return
            self.swap_data(clients, swap_p, shift_type)
        return