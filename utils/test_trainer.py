from utils.trainer_utils import TrainConfig
from flearn.trainer.fedavg import FedAvg
from flearn.trainer.fesem import FeSEM
from flearn.trainer.fedgroup import FedGroup
from flearn.trainer.ifca import IFCA

def main():
    config = TrainConfig('femnist', 'mlp', 'fedgroup')
    config.trainer_config['dynamic'] = True
    config.trainer_config['swap_p'] = 0.05
    trainer = FedGroup(config)
    trainer.train()

main()