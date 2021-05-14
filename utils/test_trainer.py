from utils.trainer_utils import TrainConfig
from flearn.trainer.fedavg import FedAvg

def main():
    config = TrainConfig('mnist', 'mlp', 'fedgroup')
    config.client_config['learning_rate'] = 0.01
    trainer = FedAvg(config)
    trainer.train()
    #trainer.train_locally()

main()