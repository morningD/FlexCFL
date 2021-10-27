import json
from pathlib import Path
import tensorflow as tf
import numpy as np
import re

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

def text2embs(dataset_list, emb_file, max_words=20):

    with open(emb_file, 'r') as inf:
        embs = json.load(inf)
    id2word = embs['vocab']
    word2id = {v: k for k,v in enumerate(id2word)}
    word_emb = np.array(embs['emba'])

    def _line_to_embs(line, w2d, d2e, max_words):
        word_list = re.findall(r"[\w']+|[.,!?;]", line)
        pad = int(max_words - len(word_list))
        pad_index = len(w2d)
        if pad <= 0:
            # Clip to max length
            word_list = word_list[:max_words]
        embs = []
        for word in word_list:
            if word in w2d:
                embs.append(d2e[w2d[word]])
            else:
                embs.append(d2e[pad_index])
        if pad > 0:
            # Add padding to the front of emb
            embs = [d2e[pad_index]]*pad + embs
        return embs

    new_dataset_list = []
    for dataset in dataset_list:
        for c, data in dataset.items():
            embs_list, labels_list = [], []
            for post, label in zip(data['x'], data['y']):
                embs = _line_to_embs(post[4], word2id, word_emb, max_words)
                embs_list.append(embs)
                labels_list += [1 if label=='4' else 0]
            dataset[c]['x'] = embs_list
            dataset[c]['y'] = labels_list
        new_dataset_list.append(dataset)
    return new_dataset_list


def read_mnist(train_data_dir, test_data_dir):
    return read_fedprox_json(train_data_dir, test_data_dir)

def read_femnist(train_data_dir, test_data_dir):
    return read_fedprox_json(train_data_dir, test_data_dir)

def read_fmnist(train_data_dir, test_data_dir):
    return read_fedprox_json(train_data_dir, test_data_dir)

def read_synthetic(train_data_dir, test_data_dir):
    return read_fedprox_json(train_data_dir, test_data_dir)

def read_sent140(train_data_dir, test_data_dir):
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
    if dsname.startswith('synthetic'):
        clients, train_data, test_data = read_synthetic(train_data_dir, test_data_dir)
    if dsname == 'sent140':
        max_words = 25
        emb_file = Path.joinpath(wspath, 'data', dsname, 'embs.json').absolute()
        clients, train_data, test_data = read_sent140(train_data_dir, test_data_dir)
        embs = text2embs([train_data, test_data], emb_file, max_words)
        train_data, test_data = embs[0], embs[1]

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