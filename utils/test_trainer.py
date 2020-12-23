from utils.trainer_utils import TrainConfig
from flearn.trainer.fedavg import FedAvg

def main():
    config = TrainConfig('mnist', 'cnn', 'fedavg')
    trainer = FedAvg(config)
    trainer.train()

main()