# FlexCFL

The source code of the Arxiv preprint article (FlexCFL):

[Flexible Clustered Federated Learning for Client-Level Data Distribution Shift](https://arxiv.org/abs/2108.09749)
> This is the extended journal version of our previous FedGroup conference paper.

# Overview
🎉FlexCFL is a wholly new reconstruction of our previous CFL framework [FedGroup](https://github.com/morningD/GrouProx).

There are many exciting improvements of FlexCFL:
- TF2.0 support. (FedGroup uses tensorflow.compat.v1 API)
- Run faster and reading friendly. (Previous FedGroup is based on FedProx)
- Easy to get started with only a few lines of configuration.
- The output is saved in `excel` format.

New functions of FlexCFL:
- Simulation of client-level distribution shift.
- Client migration strategy.
- Evaluation for auxiliary server model (global average model).
- Temperature aggregation (experimental).

Some technical fixes of FlexCFL:
- The aggregation strategy of IFCA and FeSEM change to simply averaging according to the original description.
- The maximum accuracy does not include the 'partial accuracy' (In the early training period, not all clients participate in the test)
- Cold start client gradually.

FlexCFL can simulate following (Clustered) Federated Learning frameworks:
- FedAvg & FedSGD -> [Communication-Efficient Learning of Deep Networks from Decentralized Data](http://proceedings.mlr.press/v54/mcmahan17a.html)
- FedGrop & FedGroup-RAC & FedGroup-RCC -> [FedGroup: Efficient Clustered Federated Learning via Decomposed Data-Driven Measure](https://arxiv.org/abs/2010.06870)
- IFCA -> [An Efficient Framework for Clustered Federated Learning](https://proceedings.neurips.cc/paper/2020/hash/e32cc80bf07915058ce90722ee17bb71-Abstract.html)
- FeSEM -> [Multi-center federated learning](https://arxiv.org/abs/2005.01026)
- FlexCFL & FlexCFL with group aggregation -> [Flexible Clustered Federated Learning for Client-Level Data Distribution Shift](https://arxiv.org/abs/2108.09749)

# Requirement
Python packages:
- Tensorflow (>2.0)
- Jupyter Notebook
- scikit-learn
- matplotlib
- tqdm
 
 You need to download the dataset (e.g. FEMNIST, MNIST, FashionMNIST, Synthetic) and specify a GPU id follows the guidelines of [FedProx](https://github.com/litian96/FedProx) & [Ditto](https://github.com/litian96/ditto). 

### 📌 Please download `mnist, nist, sent140, synthetic` from the `FedProx` repository and rename nist to fmnist, download `femnist` from the `Ditto` repository. The nist in FedProx is 10-class, but the femnist in Ditto is 62-class. We use the 10-class version in this project.

The directory structure of the datasets should look like this:

```
FlexCFL-->data-->mnist-->data-->train--> ***train.json
                |              |->test--> ***test.json
                |
                |->femnist-->data-->train--> ***train.json
                |                  |->test--> ***test.json
                |
                |->fmnist-->data-->...
                |
                |->synthetic_1_1-->data-->...
                |
                ...
```
# Quick Start

Just run `test.ipynb`.
The `task_list` shows examples of several configurations.
The default configurations are defined in `FlexCFL/utils/trainer_utils.py` as `TrainConfig`.

![img](https://i.imgur.com/PIJDLJD.jpg)

You can modify the configurations by directly modifying the `config` of trainer.
The commonly used hyperparameters of FlexCFL are:
```
# The dataset name, data file should be stored in floder FLexCFL/data/
trainer_config['dataset'] = 'femnist'

# The model name, model definition file should be saved in floder FLexCFL/flearn/model/
trainer_config['model'] = 'mlp'

# Total communication round
trainer_config['num_rounds'] = 300

# Evalution interval round
trainer_config['eval_every'] = 1

# Number of group
trainer_config['num_group'] = 5

# Evalute the global average model
trainer_config['eval_global_model'] = True

# Inter-group aggregation rate
trainer_config['group_agg_lr'] = 0.1

# Pretraining scale for group cold start of FlexCFL
trainer_config['pretrain_scale'] = 20

# Client data distribution shift config
trainer_config['shift_type'] = 'all'
trainer_config['swap_p'] = 0.05

# Client migration strategy
trainer_config['dynamic'] = True

# The local epoch, mini-batch size, learning rate for local SGD
client_config['local_epochs'] = 10
client_config['batch_size'] = 10
client_config['learning_rate'] = 0.003

```

You can also run FlexCFL with `python main.py`. Please modify `config` according to your needs.

# Experimental Results
All evaluation results will save in the `FlexCFL-->results-->...` directory as `excel` format files.

![img](https://i.imgur.com/87NfC3j.jpg)

# Reference
Please cite the paper of `FlexCFL` if the code helped your research 😊

- [Flexible Clustered Federated Learning for Client-Level Data Distribution Shift](https://arxiv.org/abs/2108.09749)

BibTeX
```
@article{duan2022flexible,
  title={Flexible Clustered Federated Learning for Client-Level Data Distribution Shift},
  author={Duan, Moming and Liu, Duo and Ji, Xinyuan and Wu, Yu and Liang, Liang and Chen, Xianzhang and Tan, Yujuan and Ren, Ao},
  journal={IEEE Transactions on Parallel \& Distributed Systems},
  volume={33},
  number={11},
  pages={2661--2674},
  year={2022},
  publisher={IEEE Computer Society}
}
```

