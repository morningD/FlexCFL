import numpy as np
import random
import time
import tensorflow as tf

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
        def _calculate_l2_distance(m1, m2):
                v1, v2 = process_grad(m1), process_grad(m2)
                l2d = np.sum((v1-v2)**2)**0.5
                return l2d
        for client in clients:
            diffs = [_calculate_l2_distance(client.latest_params, g.latest_params) for g in groups]
            assigned = self.groups[np.argmin(diffs)]
            # Delete the original downlink of group if exist
            if client.has_uplink():
                client.uplink[0].delete_downlink(client)
            client.set_uplink([assigned])
            # Add the new downlink
            assigned.add_downlink([client])

        return 

