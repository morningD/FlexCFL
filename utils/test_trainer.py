from utils.trainer_utils import TrainConfig
from flearn.trainer.fedavg import FedAvg

print(__name__)

def main():
    config = TrainConfig('mnist', 'mlp', 'fedavg')
    trainer = FedAvg(config)
    trainer.train()

main()