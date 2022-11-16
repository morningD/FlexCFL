import numpy as np

'''
    Define the config of trainer,
    The type of trainer contain: fedavg, fedgroup, splitfed*, splitfg*
'''
class TrainConfig(object):
    def __init__(self, dataset, model, trainer):
        self.trainer_type = trainer
        self.results_path = f'results/{dataset}/'
        
        self.trainer_config = {
            # This is common config of trainer
            'dataset': dataset,
            'model': model,
            'seed': 2077,
            'num_rounds': 300,
            'clients_per_round': 20,
            'eval_every': 1,
            'eval_locally': False,
            'dynamic': False, # whether migrate clients, no meaning if runing FedAvg
            'swap_p': 0, # Randomly swap two warm clients with probability
            'shift_type': None # {all, part, increment}
        }

        self.client_config = {
            # This is common config of client
            'local_epochs': 10,
            # However, we compile lr to model
            'learning_rate': 0.003,
            'batch_size': 10,
            # The dynamic reassign clients strategy of FedGroup
            # temperature = None means disable this function
            'temperature': None
        }

        if trainer in ['fedgroup', 'fesem', 'ifca']:
            if trainer == 'fedgroup':
                self.trainer_config.update({
                    'num_group': 3,
                    'group_agg_lr': 0.0,
                    'eval_global_model': True,
                    'pretrain_scale': 20,
                    'measure': 'EDC', # {EDC, MADC}
                    'RAC': False,
                    'RCC': False,
                    'dynamic': True,
                    'temp_metrics': 'l2', # {l2, consine}
                    'temp_func': 'step', # {step, linear, lied, eied} lied-> linear increase&exponential decrease
                    'temp_agg': False,
                    'recluster_epoch': None # [50, 100, 150]
                })
                
            if trainer in ['fesem',  'ifca']:
                self.trainer_config.update({
                    'num_group': 3,
                    # The iter-group aggregation is disabled
                    'group_agg_lr': 0.0,
                    'eval_global_model': True
                })

            self.group_config = {
                # Whether the models of all clients in the group are consistent，
                # which will greatly affect the test results.
                'consensus': False,
                'max_clients': 999,
                'allow_empty': True
            }
        
        if self.trainer_config['dataset'] == 'femnist':
            self.client_config.update({'learning_rate': 0.003})
            self.trainer_config.update({'num_group': 5})

        if self.trainer_config['dataset'].startswith('mnist'):
            self.client_config.update({'learning_rate': 0.03})
            self.trainer_config.update({'num_group': 3})

        if self.trainer_config['dataset'] == 'sent140':
            self.client_config.update({'learning_rate': 0.3})
            self.trainer_config.update({'num_group': 5})
            self.trainer_config.update({'num_rounds': 800})

        if self.trainer_config['dataset'].startswith('synthetic'):
            self.client_config.update({'learning_rate': 0.01})
            self.trainer_config.update({'num_group': 5})

        if self.trainer_config['dataset'] == 'fmnist':
            self.client_config.update({'learning_rate': 0.005})
            self.trainer_config.update({'num_group': 5})
        
        if trainer == 'splitfed':
            #TODO: plan for split learning
            pass
        if trainer == 'splitfg':
            #TODO:
            pass

def process_grad(grads):
    '''
    Args:
        grads: grad 
    Return:
        a flattened grad in numpy (1-D array)
    '''

    client_grads = grads[0] # shape = (784, 10)
    for i in range(1, len(grads)):
        client_grads = np.append(client_grads, grads[i]) # output a flattened array
        # (784, 10) append (10,)

    return client_grads

def calculate_cosine_dissimilarity(w1, w2):
    flat_w1, flat_w2 = process_grad(w1), process_grad(w2)
    cosine = np.dot(flat_w1, flat_w2) / (np.linalg.norm(flat_w1) * np.linalg.norm(flat_w2))
    dissimilarity = (1.0 - cosine) / 2.0 # scale to [0, 1] then flip
    return dissimilarity