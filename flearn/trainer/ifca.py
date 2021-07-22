import numpy as np
import random
import time
import tensorflow as tf

from utils.trainer_utils import process_grad
#from flearn.model.mlp import construct_model
from flearn.trainer.groupbase import GroupBase
from collections import Counter

"""
IFCA: "An Efficient Framework for Clustered Federated Learning"
"""
class IFCA(GroupBase):
    def __init__(self, train_config):
        super(IFCA, self).__init__(train_config)
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

    """ Minimize the Group Loss
    """
    def schedule_clients(self, round, clients, groups):        
        def _calculate_loss_distance(group, clients):
            group_params = group.latest_params
            loss_dist = []
            # Backup the current parameters
            backup_params = self.server.get_params()
            self.server.set_params(group_params)
            for c in clients:
                X, y_true = c.train_data['x'], c.train_data['y']
                loss, acc = self.server.model.evaluate(X, y_true, verbose=0)
                loss_dist.append(loss)
            # Restore the model params
            self.server.set_params(backup_params)
            return loss_dist

        diffs = []
        for g in groups:
            diffs.append(_calculate_loss_distance(g, clients))
        diffs = np.vstack(diffs)
        assigned = np.argmin(diffs, axis=0)
        for idx, c in zip(assigned, clients):
            assigned_group = groups[idx]
            # Delete the original downlink of group if exist
            if c.has_uplink():
                c.uplink[0].delete_downlink(c)
            c.set_uplink([assigned_group])
            # Add the new downlink
            assigned_group.add_downlink([c])

        return 

