import numpy as np
from tqdm import trange, tqdm
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from .fedbase import BaseFedarated
from flearn.optimizer.pgd import PerturbedGradientDescent
from flearn.utils.tf_utils import process_grad, process_sparse_grad
from flearn.models.client import Client
from flearn.utils.model_utils import Metrics
from flearn.models.group import Group
import random

""" This Server class is customized for Group Prox """
class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using Group prox to Train')
        self.group_list = [] # list of Group() instance
        self.group_ids = [] # list of group id
        self.num_group = params['num_group']
        self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])
        super(Server, self).__init__(params, learner, dataset)
        self.latest_model = self.client_model.get_params() # The global model
        self.create_groups()
        self.group_min_clients = params['min_clients']

    """
    initialize the Group() instants
    """
    def create_groups(self):
        self.group_list = [Group(gid, self.client_model) for gid in range(self.num_group)] # 0,1,...,num_group
        self.group_ids = [g.get_group_id() for g in self.group_list]
        self.group_cold_start() # init the lastest_model of all groups

    """ measure the difference between client_model and group_model """
    def measure_difference(self, client_model, group_model):
        flat_client_model = process_grad(client_model)
        flat_group_model = process_grad(group_model)
        # Strategy #1: angles (cosine) between two vectors
        diff = np.dot(flat_client_model, flat_group_model) / (
            np.sqrt(np.sum(flat_client_model**2)) * np.sqrt(np.sum(flat_group_model**2)))
        diff = 1.0 - ((diff + 1.0) / 2.0) # scale to [0, 1] then flip
        # Strategy #2: Euclidean distance between two vectors
        # diff = np.sum((client_model - group_model)**2)
        return diff

    def client_cold_start(self, client):
        if client.group is not None:
            print("Warning: Client already has a group: {:2d}.".format(client.group))
        client_model = self.pre_train_client(client)
        diff_list = [] # tuple of (group, diff)
        for g in self.group_list:
            diff_g = self.measure_difference(client_model, g.latest_model)
            diff_list.append((g, diff_g)) # w/o sort
        
        # update(Init) the diff list of client
        client.update_difference(diff_list)

        #print("client:", client.id, "diff_list:", diff_list)
        assign_group = self.group_list[np.argmin([tup[1] for tup in diff_list])]
        # Only set the group attr of client, do not actually add clients to the group
        client.set_group(assign_group)
        
    """ Deal with the group cold start problem """
    def group_cold_start(self):
        # Strategy #1: random pre-train num_group clients
        selected_clients = random.choices(self.clients, k=self.num_group)
        for c, g in zip(selected_clients, self.group_list):
            g.latest_model = self.pre_train_client(c)
        # Strategy #2: Pre-train, then clustering the directions of clients' weights

        return

    """ Allocate selected clients to each group """
    """
    def allocate_selected_clients(self, selected_clients):
        # Clear all group
        for g in self.group_list: g.clear_clients()

        # Calculate the number clients in each group
        selected_clients_num = len(selected_clients)
        group_num = len(self.group_list)
        per_group_num = np.array([selected_clients_num // group_num] * group_num)
        remain = selected_clients_num - sum(per_group_num)
        random_groups = random.sample(range(group_num), remain)
        per_group_num[random_groups] += 1 # plus the remain

        for g, max in zip(self.group_list, per_group_num):
            g.max_clients = max

        # Calculate the diff between ALL clients and ALL groups
        diff_tuples = []
        for g in self.group_list:
            for c in selected_clients:
                # !!!!!!!!!!!!!!!!  pretrain NEED
                diff = self.measure_difference(c.get_params(), g.latest_model)
                diff_tuples.append((g, diff))

        # Sort ascending by the diff
        diff_tuples = sorted(diff_tuples, key=lambda tup: tup[1])

        for (g, diff) in diff_tuples:
            if not g.is_full():
                g.add_client
        return
    """

    def measure_group_diffs(self):
        diffs = []
        for g in self.group_list:
            diff = self.measure_difference(self.group_list[0].latest_model, g.latest_model)
            diffs.append(diff)
        return diffs

    """ Pre-train the client 1 epoch and return weights """
    def pre_train_client(self, client):
        start_model = client.get_params() # Backup the start model
        soln, stat = client.solve_inner() # Pre-train the client only one epoch
        ws = soln[1] # weights of model
        self.client_model.set_params(start_model) # Recovery the model
        return ws

    def group_test(self):
        backup_model = self.latest_model # Backup the global model
        results = []
        for g in self.group_list:
            num_samples = []
            tot_correct = []
            self.client_model.set_params(g.latest_model)
            for c in g.clients.values():
                ct, ns = c.test()
                tot_correct.append(ct*1.0)
                num_samples.append(ns)
            ids = [c.id for c in g.clients.values()]
            results.append((ids, g, num_samples, tot_correct))
        self.client_model.set_params(backup_model) # Recovery the model
        return results

    def group_train_error_and_loss(self):
        backup_model = self.latest_model # Backup the global model
        results = []
        for g in self.group_list:
            num_samples = []
            tot_correct = []
            losses = []
            self.client_model.set_params(g.latest_model)
            for c in g.clients.values():
                ct, cl, ns = c.train_error_and_loss() 
                tot_correct.append(ct*1.0)
                num_samples.append(ns)
                losses.append(cl*1.0)
            ids = [c.id for c in g.clients.values()]
            results.append((ids, g, num_samples, tot_correct, losses))
        self.client_model.set_params(backup_model) # Recovery the model
        return results

    def train(self):
        print('Training with {} workers ---'.format(self.clients_per_round))
        for i in range(self.num_rounds):

            # Random select clients
            indices, selected_clients = self.select_clients(i, num_clients=self.clients_per_round)  # uniform sampling
            np.random.seed(i)  # make sure that the stragglers are the same for FedProx and FedAvg
            active_clients = np.random.choice(selected_clients, round(self.clients_per_round * (1 - self.drop_percent)), replace=False)
            
            # Clear all group, the group attr of client is retrained
            for g in self.group_list: g.clear_clients()
             # Client cold start
            for c in selected_clients:
                if c.is_cold():
                    self.client_cold_start(c)

            # Reshcedule selected clients to groups
            self.reschedule_groups(selected_clients, evenly=False)
            
            # TODO: DEBUG
            for g in self.group_list:
                print("Group", g.get_group_id(), "clients", g.get_client_ids())

            # Freeze all groups before training
            for g in self.group_list:
                g.freeze() #TODO: Cannot choose from an empty sequence
            
            print("The groups difference are:", self.measure_group_diffs())

            if i % self.eval_every == 0:
                """
                stats = self.test() # have set the latest model for all clients
                # Test on training data, it's redundancy
                stats_train = self.train_error_and_loss()
                """
                group_stats = self.group_test()
                group_stats_train = self.group_train_error_and_loss()
                for stats, stats_train in zip(group_stats, group_stats_train):
                    tqdm.write('Group {}'.format(stats[1].id))
                    tqdm.write('At round {} accuracy: {}'.format(i, np.sum(stats[3])*1.0/np.sum(stats[2])))  # testing accuracy
                    tqdm.write('At round {} training accuracy: {}'.format(i, np.sum(stats_train[3])*1.0/np.sum(stats_train[2])))
                    tqdm.write('At round {} training loss: {}'.format(i, np.dot(stats_train[4], stats_train[2])*1.0/np.sum(stats_train[2])))

            # Broadcast the global model to clients(groups)
            # self.client_model.set_params(self.latest_model)
            
            # Train each group sequentially
            for g in self.group_list:
                # Backup the origin model
                print("Begin group {:2d} training".format(g.get_group_id()))
                # Each group train group_epochs round
                for _ in range(g.group_epochs):
                    # Update the optimizer, the vstar is latest_model of this group
                    #self.inner_opt.set_params(g.latest_model, self.client_model)
                    # Set the global the group model
                    self.client_model.set_params(g.latest_model)
                    # Begin group training
                    csolns = g.train()
                # After end of the training of client, update the diff list of client
                for client, weights in csolns.items():
                    diff_list = []
                    for g in self.group_list:
                        diff_g = self.measure_difference(weights, g.latest_model)
                        diff_list.append((g, diff_g))
                    client.update_difference(diff_list)
                        
                # Recovery the client model before next group training
                #self.client_model.set_params(self.latest_model)

            # Aggregate groups model and update the global (latest) model 
            #self.latest_model = self.aggregate_groups(self.group_list)
            # Aggregate groups model and update the global (latest) model 
            self.aggregate_groups(self.group_list)


    """
    def aggregate_groups(self, groups):
        # Aggregate the groups model
        gsolns = []
        for g in groups:
            gsolns.append((sum(g.num_samples), g.latest_model)) # (n_k, soln)

        self.latest_model = self.aggregate(gsolns)
        return self.aggregate(gsolns)
    """
    def aggregate_groups(self, groups):
        gsolns = [(sum(g.num_samples), g.latest_model) for g in groups]
        group_num = len(gsolns)
        # Aggregate the models of each group separately
        for idx, g in enumerate(groups):
            base = [0]*len(gsolns[idx][1])
            weights = [0.2]*group_num
            weights[idx] = group_num
            total_weights = sum(weights)
            for j, (_, gsoln) in enumerate(gsolns):
                for k, v in enumerate(gsoln):
                    base[k] += weights[j]*v.astype(np.float64)
            averaged_soln = [v / total_weights for v in base]
            g.latest_model = averaged_soln
        

    def reschedule_groups(self, selected_clients, evenly=True):
        selected_clients = selected_clients.tolist() # convert numpy array to list
        if evenly:
            """ Strategy #1: Calculate the number of clients in each group (evenly) """
            selected_clients_num = len(selected_clients)
            group_num = len(self.group_list)
            per_group_num = np.array([selected_clients_num // group_num] * group_num)
            remain = selected_clients_num - sum(per_group_num)
            random_groups = random.sample(range(group_num), remain)
            per_group_num[random_groups] += 1 # plus the remain

            for g, max in zip(self.group_list, per_group_num):
                g.max_clients = max

            """ Allocate clients to make the client num of each group evenly """
            for c in selected_clients:
                if not c.is_cold():
                    first_rank_group = c.group
                    if not first_rank_group.is_full():
                        first_rank_group.add_client(c)
                    else:
                        # The first rank group is full, choose next group
                        diff_list = c.difference
                        # Sort diff_list
                        diff_list = sorted(diff_list, key=lambda tup: tup[1])
                        for (group, diff) in diff_list:
                            if not group.is_full():
                                group.add_client(c)
                                break   
        else:
            """ Strategy #2: Allocate clients to meet the minimum client requirements """
            for g in self.group_list: g.min_clients = self.group_min_clients
            # First ensure that each group has at least self.min_clients clients.
            diff_list, assigned_clients = [], []
            for c in selected_clients:
                diff_list += [(c, g, diff) for g, diff in c.difference]
            diff_list = sorted(diff_list, key=lambda tup: tup[2])
            for c, g, diff in diff_list:
                if len(g.client_ids) < g.min_clients and c not in assigned_clients:
                    g.add_client(c)
                    assigned_clients.append(c)
                               
            # Then add the remaining clients to their first rank group
            for c in selected_clients:
                if c not in assigned_clients:
                    first_rank_group = c.group
                    if c.id not in first_rank_group.client_ids:
                        first_rank_group.add_client(c)
        return
