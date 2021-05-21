from utils.trainer_utils import TrainConfig
from flearn.trainer.fedavg import FedAvg
from flearn.trainer.fedgroup import FedGroup

def main():
    config = TrainConfig('mnist', 'mclr', 'fedgroup')
    config.client_config['learning_rate'] = 0.01
    trainer = FedGroup(config)
    trainer.train()
    #trainer.train_locally()

main()