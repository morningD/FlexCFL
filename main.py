import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from utils.trainer_utils import TrainConfig
from flearn.trainer.fedavg import FedAvg
from flearn.trainer.fedgroup import FedGroup

def main():
    config = TrainConfig('femnist', 'mlp', 'fedavg')
    config.trainer_config['dynamic'] = False
    trainer = FedAvg(config)
    trainer.train()
    #trainer.train_locally()

def main_flexcfl():
    config = TrainConfig('femnist', 'mlp', 'fedgroup')
    config.trainer_config['dynamic'] = True
    config.trainer_config['shift_type'] = "all"
    config.trainer_config['swap_p'] = 0.05
    trainer = FedGroup(config)
    trainer.train()

main_flexcfl()