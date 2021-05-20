
from utils.trainer_utils import TrainConfig
from flearn.trainer.fedavg import FedAvg
from flearn.trainer.fedgroup import FedGroup

def main():
    config = TrainConfig('femnist', 'mlp', 'fedavg')
    trainer = FedAvg(config)
    trainer.train()
    #trainer.train_locally()

main()