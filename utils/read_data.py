import json
from pathlib import Path
import tensorflow as tf
import numpy as np

def read_fedprox_json(train_data_dir, test_data_dir):
    '''parses data in given train and test data directories

    assumes:
    - the data in the input directories are .json files with 
        keys 'users' and 'user_data'
    - the set of train set users is the same as the set of test set users
    
    Return:
        clients: list of client ids
        train_data: dictionary of train (numpy) data
        test_data: dictionary of test (numpy) data
    '''
    clients = []
    train_npdata = {}
    test_npdata = {}

    train_files = Path.iterdir(train_data_dir)
    train_files = [f for f in train_files if f.suffix == '.json']
    for file_path in train_files:
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        train_npdata.update(cdata['user_data'])

    test_files = Path.iterdir(test_data_dir)
    test_files = [f for f in test_files if f.suffix == '.json']
    for file_path in test_files:
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        test_npdata.update(cdata['user_data'])

    clients = list(sorted(train_npdata.keys()))

    '''
    # Convert numpy train\test data to tf.Dataset format
    train_tfdata, test_tfdata = {}, {}
    for c in clients:
        print("handling", c)
        train_tfdata[c] = tf.data.Dataset.from_tensor_slices((train_npdata[c]['x'], train_npdata[c]['y']))
        test_tfdata[c] = tf.data.Dataset.from_tensor_slices((test_npdata[c]['x'], test_npdata[c]['y']))
    '''
    return clients, train_npdata, test_npdata

def read_mnist(train_data_dir, test_data_dir):
    return read_fedprox_json(train_data_dir, test_data_dir)

def read_femnist(train_data_dir, test_data_dir):
    return read_fedprox_json(train_data_dir, test_data_dir)

def read_fmnist(train_data_dir, test_data_dir):
    return read_fedprox_json(train_data_dir, test_data_dir)

def read_federated_data(dsname):
    clients = []
    train_data = {}
    test_data = {}
    train_size, test_size = 0, 0
    wspath = Path(__file__).parent.parent.absolute() # The working path of SplitGP
    # The training data directory
    train_data_dir = Path.joinpath(wspath, 'data', dsname, 'data', 'train').absolute()
    # The testing data directory
    test_data_dir = Path.joinpath(wspath, 'data', dsname, 'data', 'test').absolute()

    if dsname.startswith('mnist'):
        clients, train_data, test_data = read_mnist(train_data_dir, test_data_dir)
    if dsname == 'femnist':
        clients, train_data, test_data = read_femnist(train_data_dir, test_data_dir)
    if dsname == 'fmnist':
        clients, train_data, test_data = read_fmnist(train_data_dir, test_data_dir)
    
    # Convert list to numpy array
    for c in train_data.keys():
        train_data[c]['x'] = np.array(train_data[c]['x'], dtype=np.float32) # shape=(num_samples, num_features)
        train_data[c]['y'] = np.array(train_data[c]['y'], dtype=np.uint8) # shape=(num_samples, )
        train_size += train_data[c]['y'].shape[0]
    for c in test_data.keys():
        test_data[c]['x'] = np.array(test_data[c]['x'], dtype=np.float32)
        test_data[c]['y'] = np.array(test_data[c]['y'], dtype=np.uint8)
        test_size += test_data[c]['y'].shape[0]
        
    # Print the size of this dataset and client count
    print(f'The dataset size: {train_size + test_size}, train size: {train_size}, test size: {test_size}.')
    print(f'The train client count: {len(train_data)}. The test client count: {len(test_data)}.')
    return clients, train_data, test_data