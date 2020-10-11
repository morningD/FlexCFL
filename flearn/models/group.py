import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from collections import OrderedDict
from flearn.models.client import Client
from flearn.optimizer.pgd import PerturbedGradientDescent
from flearn.utils.tf_utils import process_grad, process_sparse_grad
import random

class Group(object):
    def __init__(self, gid, model=None):
        self.client_model = model
        self.latest_model = model.get_params()
        self.latest_update = model.get_params()
        self.id = gid # integer
        self.clients = OrderedDict() # {cid: Client()}
        self.client_ids = [] # list of client ids
        self.group_epochs = 1
        self.num_epochs = 20
        self.model_len = 0
        self.num_samples = []
        self.max_clients = 1e4 # Init to a large number
        self.min_clients = 0

        #debug
        self.grads = None
        self.batch_size = 10
    
    """ Add a client to this group"""
    def add_client(self, c: Client):
        if c.id not in self.clients.keys():
            if not self.is_full():
                self.clients[c.id] = c
                self.client_ids.append(c.id)
                c.set_group(self)
            else:
                print("Warning: Group {:2d} is full.".format(self.id))
        else:
            print("Warning: Client {} alreay in {:2d} group.".format(c.id, self.id))

    def add_clients(self, c_list):
        for c in c_list:
            self.add_client(c)

    def clear_clients(self):
        # Note: The group id attr of clients is reatained
        # clear the OrderDict and ids of clients
        self.clients.clear()
        self.client_ids.clear()
        self.num_samples.clear()
        # reset the max_clients
        self.max_clients = 1e4
        self.min_clients = 0


    def get_client_ids(self):
        return self.client_ids

    def get_group_id(self):
        return self.id

    """ Set the prime client. You should freeze this group before train """
    def freeze(self):
        self.model_len = process_grad(self.latest_model).size # For MNIST, should be 784*10+10
        self.num_samples = [c.num_samples for c in self.clients.values()]
        if len(self.client_ids) < self.min_clients:
            print("Warning: This group does not meet the minimum client requirements.")

    def is_empty(self):
        return bool(not self.client_ids)

    def is_full(self):
        return bool(len(self.client_ids) >= self.max_clients)

    """ Aggregate the models of this group """
    def aggregate(self, wsolns):
        total_weight = 0.0
        base = [0]*len(wsolns[0][1]) # wsolns ->(n_k, soln), (bytes_w, comp, bytes_r)
        for (w, soln) in wsolns:  # w is the number of local samples
            total_weight += w
            for i, v in enumerate(soln): # for each w_i in w
                base[i] += w*v.astype(np.float64)

        averaged_soln = [v / total_weight for v in base]

        return averaged_soln

    def train(self):

        """ Pre test the diff of gradient """
        """
        local_grads = []
        global_grads = np.zeros(self.model_len)
        for i in range(self.group_epochs):
            for c in self.clients.values():
                num, client_grad = c.get_grads(self.model_len)
                local_grads.append(client_grad)
                global_grads = np.add(global_grads, client_grad * num)
        global_grads = global_grads * 1.0 / np.sum(np.asarray(self.num_samples))
        difference = 0
        for idx in range(len(self.clients)):
            difference += np.sum(np.square(global_grads - local_grads[idx]))
        difference = difference * 1.0 / len(self.clients)
        print('gradient difference: {}'.format(difference))
        """

        """ Training """
        # Backup the training model first, however we just make it logically correct,
        # Actually ,the trainig procedure didn't refresh the training model
        start_model = self.client_model.get_params()

        csolns = [] # buffer for receiving client solutions
        cupdates_dict = {} # dict buffer for send back clients' updates to server
        for c in self.clients.values():
            # communicate the latest group model
            c.set_params(self.latest_model)
            soln, stats = c.solve_inner(num_epochs=self.num_epochs, batch_size=self.batch_size)
            csolns.append(soln)
            cupdates_dict[c] = [w1-w0 for w0, w1 in zip(self.latest_update, soln[1])] # {Client:updates}
        new_model = self.aggregate(csolns)
        self.latest_update = [w1-w0 for w0, w1 in zip(self.latest_model, new_model)]
        self.latest_model = new_model

        self.client_model.set_params(start_model) # Recovery the training model

        return cupdates_dict # return {Client:updates} to server


