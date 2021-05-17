import numpy as np

'''
    Define the config of trainer,
    The type of trainer contain: fedavg, fedgroup, splitfed, splitfg
'''
class TrainConfig(object):
    def __init__(self, dataset, model, trainer):
        self.trainer_type = trainer
        
        self.trainer_config = {
            # This is common config of trainer
            'dataset': dataset,
            'model': model,
            'seed': 2077,
            'num_rounds': 200,
            'clients_per_round': 20,
            'eval_every': 1,
            'eval_locally': True
        }

        self.client_config = {
            # This is common config of client
            'local_epochs': 20,
            # However, we compile lr to model
            'learning_rate': 0.003,
            'batch_size': 10
        }

        if trainer == 'fedgroup':
            self.trainer_config.update({
                'num_group': 3,
                'group_agg_lr': 0.01,
                'eval_global_model': True
            })

            self.group_config = {
                'max_clients': 999,
                'allow_empty': True
            }
        
        if trainer == 'splitfed':
            #TODO:
            pass
        if trainer == 'splitfg':
            #TODO:
            pass

        