# FedGroup

The source code of the Arxiv preprint article:

[FedGroup: Ternary Cosine Similarity-based Clustered Federated Learning Framework toward High Accuracy in Heterogeneous Data](https://arxiv.org/abs/2010.06870)

# Requirement
Python packages:
- Tensorflow (>2.0)
- Jupyter Notebook
- scikit-learn
- matplotlib
- tqdm
 
 You need to download the dataset (e.g. FEMNIST, MNIST, Sent140, Synthetic) and specify a GPU id follow the guidelines of [FedProx](https://github.com/litian96/FedProx).

The directory structure of the datasets should look like this:

```
GrouProx-->data-->mnist-->data-->train--> ***train.json
               |              |->test--> ***test.json
               |
               |->nist-->data-->train--> ***train.json
               |                     |-> ***test.json
               |
               |->sent140--> ...
               |
               ...
```
# Quick Start
Just run `GrouProx_notebook.ipynb`.

You can modify the configurations by directly modifying the code of `GrouProx_notebook.ipynb`.
The common hyperparameters of FedGroup is:
```
# Name of dataset, should be list in GrouProx-->data-->...
params['dataset'] = 'sent140' 

# Name of model, should be list in GrouProx-->flearn-->models-->params['dataset']-->...
params['model'] = 'stacked_lstm'

# Name of optimizer, should be one of ['fedavg', 'fedprox', 'grouprox']
params['optimizer'] = 'grouprox'

# The dropout rate as demonstrated in the FedProx paper
params['drop_percent'] = 0

# Total communication rounds
params['num_rounds'] = 200

# Local epoch E, same as FedProx
params['num_epochs'] = 20

# Local mini-batch size, same as FedProx
params['batch_size'] = 10

# Evaluate the group model every $params['eval_every'] rounds
params['eval_every'] = 1

# The number of clients K selected per round
params['clients_per_round'] = 20

# Random seed
params['seed'] = 233

# Inter-group learning rate
params['agg_lr'] = 0.01

# Number of 'Groups'
params['num_group'] = 5

# Some specific hyperparameters of FedGroup
if params['optimizer']  == 'grouprox':
  # Whether to use Proximal method, True for FedGrouProx
  params['proximal'] = False

  # Radomly Assign Clients and Random Cluster Centers strategy, please see the paper for details 
  params['RAC'] = False
  params['RCC'] = False

  # The Group may be empty if True
  params['allow_empty'] = True

```
# Experimental Results
All evaluation results will save in the `GrouProx-->results-->...` directory as `csv` format files.

# Reference
Please cite the preprint version of `FedGroup` if the code helped your research 😊

- [FedGroup: Ternary Cosine Similarity-based Clustered Federated Learning Framework toward High Accuracy in Heterogeneous Data](https://arxiv.org/abs/2010.06870)

```
@article{duan2020fedgroup,
  title={FedGroup: Ternary Cosine Similarity-based Clustered Federated Learning Framework toward High Accuracy in Heterogeneity Data},
  author={Duan, Moming and Liu, Duo and Ji, Xinyuan and Liu, Renping and Liang, Liang and Chen, Xianzhang and Tan, Yujuan},
  journal={arXiv preprint arXiv:2010.06870},
  year={2020}
}
```

