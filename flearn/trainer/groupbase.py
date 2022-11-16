import numpy as np
import importlib
import tensorflow as tf
import random
import time
from termcolor import colored

from utils.read_data import read_federated_data
from utils.trainer_utils import TrainConfig
#from flearn.model.mlp import construct_model
from utils.trainer_utils import process_grad, calculate_cosine_dissimilarity
from flearn.server import Server
from flearn.client import Client
from flearn.group import Group
from collections import Counter
from utils.export_result import ResultWriter
from math import ceil

class GroupBase(object):
    def __init__(self, train_config):
        # Transfer trainer config to self, we save the configurations by this trick
        for key, val in train_config.trainer_config.items():
            setattr(self, key, val)
        self.trainer_type = train_config.trainer_type
        # Get the config of client
        self.client_config = train_config.client_config
        # Get the config of group
        self.group_config = train_config.group_config
        if self.eval_locally == True:
            self.group_config.update({'eval_locally': True})
        else:
            self.group_config.update({'eval_locally': False})

        # Set the random set
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Construct the actors
        self.clients = []
        self.groups = []
        self.construct_actors()

        # Create results writer
        self.writer = ResultWriter(train_config)

        # Store the initial model params
        self.init_params = self.server.get_params()

    def construct_actors(self):
        # 1, Read dataset
        clients, train_data, test_data = read_federated_data(self.dataset)

        # 2, Get model loader according to dataset and model name and construct the model
        # Set the model loader according to the dataset and model name
        model_path = 'flearn.model.%s.%s' % (self.dataset.split('_')[0], self.model)
        self.model_loader = importlib.import_module(model_path).construct_model
        # Construct the model
        client_model = self.model_loader(self.trainer_type, self.client_config['learning_rate'])

        # 3, Construct server
        self.server = Server(model=client_model)

        # 4, Construct Groups and set their uplink
        for id in range(self.num_group):
            # We need create the empty datasets for each group
            empty_train_data, empty_test_data = {'x':[],'y':[]}, {'x':[],'y':[]}
            self.groups.append(Group(id, self.group_config, empty_train_data, empty_test_data,
                                [self.server], client_model))

        # 5, Construct clients (don't set their uplink)
        self.clients = [Client(id, self.client_config, train_data[id], test_data[id], 
                        model=client_model) for id in clients]

        # 6, Set the server's downlink to groups
        self.server.add_downlink(self.groups)

        # 7*, We evaluate the auxiliary global model on server
        # To speed the testing, we need construct a local test dataset for server
        if self.eval_global_model == True:
            server_test_data = {'x':[], 'y':[]}
            for c in clients:
                server_test_data['x'].append(test_data[c]['x'])
                server_test_data['y'].append(test_data[c]['y'])
            self.server.test_data['x'] = np.vstack(server_test_data['x'])
            self.server.test_data['y'] = np.hstack(server_test_data['y'])

        # DEBUG: Randomly schedule all clients
        #self.randomly_schedule_clients()
    

    '''
    The iter-group aggregation
    Note: The latest updates of group will be accumulated from last fedavg training
    '''
    def inter_group_aggregation(self, train_results, agg_lr=0.0):
        group_num = len(train_results)
        groups = [rest[0] for rest in train_results]
        gsolns = [g.latest_params for g in groups]
        # Calculate the scale of group models
        gscale = [0]*group_num
        for i, gsoln in enumerate(gsolns):
            for v in gsoln:
                gscale[i] += np.sum(v.astype(np.float64)**2)
            gscale[i] = gscale[i]**0.5
        # Aggregate the models of each group separately
        for idx, g in enumerate(groups):
            base = [0]*len(gsolns[idx])
            weights = [agg_lr*(1.0/scale) for scale in gscale]
            weights[idx] = 1 # The weight of the main group is 1
            total_weights = sum(weights)
            for j, gsoln in enumerate(gsolns):
                for k, v in enumerate(gsoln):
                    base[k] += weights[j]*v.astype(np.float64)
            averaged_soln = [v / total_weights for v in base]
            # Note: The latest_update accumulated from last fedavg training
            inter_aggregation_update = [w1-w0 for w0, w1 in zip(g.latest_params, averaged_soln)]
            g.latest_updates = [up0+up1 for up0, up1 in zip(g.latest_updates, inter_aggregation_update)]
            g.latest_params = averaged_soln
        return

    def train(self):
        for comm_round in range(self.num_rounds):

            print(f'---------- Round {comm_round} ----------')
            # 0, Init time record
            train_time, test_time, agg_time = 0, 0, 0

            # 1, Random select clients
            selected_clients = self.select_clients(comm_round, self.clients_per_round)
            #selected_clients = self.clients[:20] # DEBUG, only use first 20 clients to train
            
            # * Change the clients's data distribution
            self.data_distribution_shift(comm_round, self.clients, self.shift_type, self.swap_p)

            # 2, Schedule clients (for example: reassign) or cold start clients, need selected clients only
            schedule_results = self.schedule_clients(comm_round, selected_clients, self.groups)

            # 3, Schedule groups (for example: recluster), need all clients
            self.schedule_groups(comm_round, self.clients, self.groups)

            # 4, Train selected clients
            start_time = time.time()
            train_results = self.server.train(selected_clients)
            train_time = round(time.time() - start_time, 3)
            if train_results == None:
                continue

            # *, Print the grouping information of this round
            gids = [c.uplink[0].id for c in selected_clients]
            count = Counter(gids)
            for id in sorted(count):
                print(f'Round {comm_round}, Group {id} has {count[id]} client.')

            # 5, Inter-group aggregation according to the group learning rate
            if self.group_agg_lr > 0:
                start_time = time.time()
                self.inter_group_aggregation(train_results, self.group_agg_lr)
                agg_time = round(time.time() - start_time, 3)

            # 6, update the discrepancy and dissmilarity between group and client
            diffs = self.refresh_discrepancy_and_dissmilarity(selected_clients)

            # 7, schedule clients after training
            self.schedule_clients_after_training(comm_round, selected_clients, self.groups)

            # 7, Summary this round of training
            train_summary = self.summary_results(comm_round, train_results=train_results)

            # 8, Update the auxiliary global model. Simply average group models without weights
            # The empty group will not be aggregated
            self.update_auxiliary_global_model([rest[0] for rest in train_results])
            # Set the training model to the new server model, however this step is not important
            self.server.set_params(self.server.latest_params)

            # 9, Test the model (Last round training) every eval_every round and last round
            if comm_round % self.eval_every == 0 or comm_round == self.num_rounds:
                start_time = time.time()
                
                # Test model on all groups
                test_results = self.server.test(self.server.downlink)
                # Summary this test
                test_summary = self.summary_results(comm_round, test_results=test_results)
                
                if self.eval_global_model == True:
                    # Test model on the server auxiliary model
                    test_samples, test_acc, test_loss = self.server.test_locally()
                    test_results = [[self.server, test_samples, test_acc, test_loss]]
                    # Summary this test
                    self.summary_results(comm_round, test_results=test_results)

                test_time = round(time.time() - start_time, 3)
                # Write the training result and test result to file
                # Note: Only write the complete test accuracy after all client cold start
                self.writer.write_summary(comm_round, train_summary, test_summary, diffs, schedule_results)

            # 10, Print the train, aggregate, test time
            print(f'Round: {comm_round}, Training time: {train_time}, Test time: {test_time}, \
                Inter-Group Aggregate time: {agg_time}')
    
    def select_clients(self, comm_round, num_clients=20):
        '''selects num_clients clients weighted by number of samples from possible_clients
            For the consideration of test comparability, we first select the client by round robin, and then select by randomly
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))
        
        Return:
            list of selected clients objects
        '''

        num_clients = min(num_clients, len(self.clients))
        # Round robin
        if comm_round < len(self.clients) / num_clients:
            head = comm_round * num_clients
            if head + num_clients <= len(self.clients):
                selected_clients = self.clients[head: head+num_clients]
            else:
                selected_clients = self.clients[head:] + self.clients[:head+num_clients-len(self.clients)]
        # Random selecte clients
        else:
            random.seed(comm_round+self.seed)  # make sure for each comparison, we are selecting the same clients each round
            selected_clients = random.sample(self.clients, num_clients)
            random.seed(self.seed) # Restore the seed
        return selected_clients
    
    def federated_averaging_aggregate(self, updates, nks):
        return self.weighted_aggregate(updates, nks)

    def simply_averaging_aggregate(self, params_list):
        weights = [1.0] * len(params_list)
        return self.weighted_aggregate(params_list, weights)

    def weighted_aggregate(self, updates, weights):
        # Aggregate the updates according their weights
        normalws = np.array(weights, dtype=float) / np.sum(weights, dtype=np.float)
        num_clients = len(updates)
        num_layers = len(updates[0])
        agg_updates = []
        for la in range(num_layers):
            agg_updates.append(np.sum([up[la]*pro for up, pro in zip(updates, normalws)], axis=0))

        return agg_updates # -> list

    '''
    Summary the train results or test results
    '''
    def summary_results(self, comm_round, train_results=None, test_results=None):

        partial_test_acc = False
        ty2 = ''
        if train_results:
            results = train_results
            ty, cor = 'Train', 'blue'
        elif test_results:
            results = test_results
            ty, cor = 'Test', 'red'
            if results[0][0].actor_type == 'server':
                ty, cor = 'Auxiliary Model Test', 'green'
            if comm_round < len(self.clients) / min(self.clients_per_round, len(self.clients)):
                ty2 += '(Partial)'
                # We do not write the partial test accuracy
                partial_test_acc = True
            else:
                ty2 += '(Complete)'
        else:
            return

        nks = [rest[1] for rest in results]
        num_sublink = len(nks) # Groups or clients
        accs = [rest[2] for rest in results]
        losses = [rest[3] for rest in results]
        print('groupbase.py', ty, 'NKS:', nks)
        weighted_acc = np.average(accs, weights=nks)
        weighted_loss = np.average(losses, weights=nks)
        print(colored(f'Round {comm_round}, {ty+ty2} ACC: {round(weighted_acc, 4)},\
            {ty+ty2} Loss: {round(weighted_loss, 4)}', cor, attrs=['reverse']))

        summary = {'Total': (num_sublink, weighted_acc, weighted_loss)}
        # Clear partial test result on summary
        if partial_test_acc == True: summary = {'Total': (None, None, None)}

        # Record group accuracy and loss
        if results[0][0].actor_type == 'group':
            groups = [rest[0] for rest in results]
            for idx, g in enumerate(groups):
                if partial_test_acc == True: accs[idx] 
                summary[f'G{g.id}'] = (accs[idx], losses[idx], nks[idx]) # accuracy, loss, number of samples
                print(f'Round {comm_round}, Group: {g.id}, {ty} ACC: {round(accs[idx], 4)},\
                    {ty} Loss: {round(losses[idx], 4)}')

                # Clear partial group test result on summary
                if partial_test_acc == True: summary[f'G{g.id}'] = (None, None, None)

        return summary

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
        train_size, train_acc, train_loss, soln, update = self.server.solve_inner(num_epoch, batch_size)
        # 3, Server Apply update
        self.server.apply_update(update)
        # 4, Server test locally
        test_size, test_acc, test_loss = self.server.test_locally()

        # 5, Print result, we show the accuracy and loss of all training epochs
        print(f"Train size: {train_size} Train ACC: {[round(acc, 4) for acc in train_acc]} \
             Train Loss: {[round(loss, 4) for loss in train_loss]}")
        print(colored(f"Test size: {test_size}, Test ACC: {round(test_acc, 4)}, \
            Test Loss: {round(test_loss, 4)}", 'red', attrs=['reverse']))

        return

    def schedule_clients(self, round, clients, groups):
        """ Randomly schedule all clients to gorups
            Rewrite this function if need
        """
        for client in clients:
            if client.has_uplink() == False:
                assigned_group = random.choice(groups)
                client.set_uplink([assigned_group])
                assigned_group.add_downlink(client)
        return

    def schedule_clients_after_training(self, comm_round, clients, groups):
        return

    def schedule_groups(self, round, clients, groups):
        """rewrite this function if need
        """
        return

    # Refresh the difference value (discrepancy) and cosine dissimilarity cosine of clients and gorups and
    # return the discrepancy (w/0 cosine dissimilarity) information for summary
    def refresh_discrepancy_and_dissmilarity(self, clients):
        def _calculate_mean_diffs(clients):
            discrepancy = [c.discrepancy for c in clients]
            dissimilarity = [c.cosine_dissimilarity for c in clients]
            return np.mean(discrepancy), np.mean(dissimilarity)

        # Call the discrepancy update function of clients
        for c in clients: c.update_difference()

        diffs = {}
        diffs['Total'] = _calculate_mean_diffs(clients)[0] # Return discrepancy
        groups = set([c.uplink[0] for c in clients])
        for g in groups:
            gc = [c for c in clients if c.uplink[0] == g]
            g.discrepancy, g.cosine_dissimilarity = _calculate_mean_diffs(gc)
            # i.e. { 'G1': (numer of group clients, discrepancy) }
            diffs[f'G{g.id}'] = (len(gc), g.discrepancy)
        return diffs

    """ Average the groups' model and get the new auxiliary global model
    """
    def update_auxiliary_global_model(self, groups):
        prev_server_params = self.server.latest_params
        new_server_params = self.simply_averaging_aggregate([g.latest_params for g in groups])
        self.server.latest_updates = [(new-prev) for prev, new in zip(prev_server_params, new_server_params)]
        self.server.latest_params = new_server_params
        return

    """ This function will randomly swap <warm> clients' data with probability swap_p
    """
    def swap_data(self, clients, swap_p, scope='all'):

        # Swap the data of warm clients with probability swap_p
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
                    if g2!= g1:
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
            warm_clients = [c for c in clients if c.has_uplink() == True]
            if len(warm_clients) == 0: return
            self.swap_data(warm_clients, swap_p, shift_type)
        return
