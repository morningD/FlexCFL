import numpy as np
import random
import time
import tensorflow as tf
from termcolor import colored

from utils.trainer_utils import process_grad
#from flearn.model.mlp import construct_model
from flearn.trainer.groupbase import GroupBase
from collections import Counter

"""
FeSEM: "Multi-Center Federated Learning"
"""
class FeSEM(GroupBase):
    def __init__(self, train_config):
        super(FeSEM, self).__init__(train_config)
        self.group_cold_start()
        # Make sure the group aggregation is disabled
        self.group_agg_lr = 0.0
        # FeSEM uses simply average aggregation strategy
        for g in self.groups: g.aggregation_strategy = 'avg'

    # Random initialize group models as centers
    def group_cold_start(self):
        # Backup the original model params
        backup_params = self.server.get_params()
        # Reinitialize num_group clients models as centers models
        for idx, g in enumerate(self.groups):
            # Change the seed of tensorflow
            new_seed = (idx + self.seed) * 888
            # Reinitialize params of model
            tf.random.set_seed(new_seed)
            new_model = self.model_loader(self.trainer_type, self.client_config['learning_rate'])
            new_params = new_model.get_weights()
            del new_model
            g.latest_params = new_params
            
        # Restore the seed of tensorflow
        tf.random.set_seed(self.seed)
        # Restore the parameter of model
        self.server.set_params(backup_params)

    """ Minimize the L2 distance
    """
    def schedule_clients(self, round, clients, groups):
        schedule_results = None
        if self.dynamic == True:
            # 1, Redo cold start distribution shift clients
            shift_count, migration_count = 0, 0
            warm_clients = [wc for wc in self.clients if wc.has_uplink() == True]
            for client in warm_clients:
                count = client.check_distribution_shift()
                if count is not None and client.distribution_shift == True:
                    shift_count += 1
                    prev_g = client.uplink[0]
                    prev_g.delete_downlink(client)
                    client.clear_uplink()
                    self.clients_cold_start([client], groups)
                    new_g = client.uplink[0]
                    client.train_label_count = count
                    client.distribution_shift = False
                    if prev_g != new_g:
                        migration_count += 1
                        print(colored(f'Client {client.id} migrate from Group {prev_g.id} \
                            to Group {new_g.id}', 'yellow', attrs=['reverse']))
            schedule_results = {'shift': shift_count, 'migration': migration_count}

        self.clients_cold_start(clients, groups)

        return schedule_results

    def clients_cold_start(self, clients, groups):
        def _calculate_l2_distance(m1, m2):
                v1, v2 = process_grad(m1), process_grad(m2)
                l2d = np.sum((v1-v2)**2)**0.5
                return l2d
        assign_results = []
        for client in clients:
            diffs = [_calculate_l2_distance(client.local_soln, g.latest_params) for g in groups]
            assigned = self.groups[np.argmin(diffs)]
            # Delete the original downlink of group if exist
            if client.has_uplink():
                client.uplink[0].delete_downlink(client)
            client.set_uplink([assigned])
            # Add the new downlink
            assigned.add_downlink([client])  
            assign_results.append(assigned)
        return assign_results      

