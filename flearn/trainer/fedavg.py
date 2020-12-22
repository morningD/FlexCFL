import numpy as np
import importlib
from SplitGP.utils.read_data import read_federated_data
from SplitGP.utils.trainer_utils import TrainConfig
#from flearn.model.mlp import construct_model
from flearn.server import Server
from flearn.client import Client

class FedAvg(object):
    def __init__(self, train_config):
        # transfer trainer config to self
        for key, val in train_config.trainer_config.items(): 
            setattr(self, key, val)

        # Get the config of client
        self.client_config = train_config.client_config

        # Construct the actors
        self.clients = None
        self.construct_actors(self.dataset)

        # Set the model loader according to the dataset and model name
        model_path = 'flearn.model.%s.%s' % (self.dataset, self.model)
        self.model_loader = importlib.import_module(model_path).construct_model

    def construct_actors(self, dataset):
        # 1, Read dataset and construct client model
        clients, train_data, test_data = read_federated_data(dataset)
        client_model = self.model_loader('fedavg', self.client_config['learning_rate'])
        # 2, Construct server
        self.server = Server(model=client_model)
        # 3, Construct clients and set their uplink
        self.clients = [Client(id, train_data[id], test_data[id], uplinke=[self.server],
                        model=client_model) for id in clients]
        # 4, Set the downlink of server
        self.server.delete_downlink(self.clients)

    def train(self):
        for round in range(self.num_rounds):
            selected_clients = self.select_clients(round)
            self.server.train(selected_clients)
    
    def select_clients(self, round, num_clients=20):
        '''selects num_clients clients weighted by number of samples from possible_clients
        
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))
        
        Return:
            list of selected clients objects
        '''

        num_clients = min(num_clients, len(self.clients))
        np.random.seed(round)  # make sure for each comparison, we are selecting the same clients each round
        indices = np.random.choice(range(len(self.clients)), num_clients, replace=False)
        return self.clients[indices]
        
