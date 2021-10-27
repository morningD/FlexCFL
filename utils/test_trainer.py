from utils.trainer_utils import TrainConfig
from flearn.trainer.fedavg import FedAvg
from flearn.trainer.fesem import FeSEM
from flearn.trainer.fedgroup import FedGroup
from flearn.trainer.ifca import IFCA

def main():
    config = TrainConfig('sent140', 'gru', 'fedavg')
    config.trainer_config['dynamic'] = False
    config.trainer_config['swap_p'] = 0.0
    trainer = FedAvg(config)
    #trainer.train_locally()
    trainer.train()

main()