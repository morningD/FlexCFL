import numpy as np

from math import ceil

'''
Define the base actor of federated learning framework
like Server, Group, Client.
'''

class Actor(object):
    def __init__(self, id, actor_type, train_data={'x':[],'y':[]}, test_data={'x':[],'y':[]}, model=None):
        self.id = id
        self.train_data = train_data
        self.test_data = test_data
        self.model = model # callable tf.keras.model
        self.actor_type = actor_type
        self.name = 'NULL'
        # The latest model parameter and update of this actor, which will be modified by train, aggregate, refresh, init
        # The latest params and updates are set by fresh_latest_params_updates(), apply_update(), train()
        # The global training model will be set to latest params before training.
        self.latest_params, self.latest_updates = None, None
        # The latest local training solution and traning gradient of this actor
        # This local variables will be <automatically> set by all <local> training functions <forward propagation>,
        # which like solve inner and solve iter of actor, train and pretrain of client
        self.local_soln, self.local_gradient = None, None
        # init train and test size to zero, it will depend on the actor type
        self.train_size, self.test_size = 0, 0 
        self.uplink, self.downlink = [], [] # init to empty, depend on the actor type
        # Is this actor can train or test,
        # Note: This variable have differenct meaning according to differnent type of actor
        self.trainable, self.testable = False, False 

        self.preprocess()

    def preprocess(self):
        # Give the name of actor, for example, 'client01', 'group01'
        self.name = str(self.actor_type) + str(self.id)
        # Initialize the latest model weights and updates
        self.latest_params, self.local_soln = self.get_params(), self.get_params()
        self.latest_updates = [np.zeros_like(ws) for ws in self.latest_params]
        self.local_gradient = [np.zeros_like(ws) for ws in self.latest_params]

    '''Return the parameters of global model instance
    '''
    def get_params(self):
        if self.model:
            return self.model.get_weights()
    
    def set_params(self, weights):
        # Set the params of model,
        # But the latest_params and latest_updates will not be refreshed
        if self.model:
            self.model.set_weights(weights)

    """     
    def solve_gradients(self, num_epoch=1, batch_size=10):
        '''
        Solve the local optimization base on local training data, 
        the gradient is NOT applied to model
        
        Return: num_samples, Gradients
        '''
        if self.train_data['y'] > 0:
            X, y_true = self.train_data['x'], self.train_data['y']
            num_samples = y_true.shape[0]
            with tf.GradientTape() as tape:
                y_pred = self.model(X)
                loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            
            return num_samples, gradients
        else:
            # Return 0 and all zero gradients [0, 0, ...],
            # if this actor has not training set
            return 0, [np.zeros_like(ws) for ws in self.latest_updates]
    """

    def solve_inner(self, num_epoch=1, batch_size=10, pretrain=False):
        '''
        Solve the local optimization base on local training data,
        This Function will not change the params of model,
        Call apply_update() to change model
        
        Return: num_samples, train_acc, train_loss, update
        '''
        if self.train_data['y'].shape[0] > 0:
            X, y_true = self.train_data['x'], self.train_data['y']
            num_samples = y_true.shape[0]
            # Backup the current model params
            backup_params = self.get_params()
            # Confirm model params is euqal to latest params
            t0_weights = self.latest_params
            self.set_params(t0_weights)
            # Use model.fit() to train model
            history = self.model.fit(X, y_true, batch_size, num_epoch, verbose=0)
            t1_weights = self.get_params()
            gradient = [(w1-w0) for w0, w1 in zip(t0_weights, t1_weights)]
            
            # Roll-back the weights of current model
            self.set_params(backup_params)
            if pretrain == False:
                # Store the latest local solution params
                self.local_soln = t1_weights
                # Calculate the gradient
                self.local_gradient = gradient
            # Get the train accuracy and train loss
            #print(history.history) # Debug
            train_acc = history.history['accuracy']
            train_loss = history.history['loss']
            #print('actor.py:104', train_acc) # DEBUG
            return num_samples, train_acc, train_loss, t1_weights, gradient
        else:
            # Return 0,0,0 and all zero updates [0, 0, ...],
            # if this actor has not training set
            return 0, [0], [0], self.latest_params, [np.zeros_like(ws) for ws in self.latest_params]

    def solve_iters(self, num_iters=1, batch_size=10, pretrain=False):

        def batch_data_multiple_iters(data, batch_size, num_iters):
            data_x = data['x']
            data_y = data['y']
            data_size = data_y.shape[0]

            random_idx = np.arange(data_size)
            np.random.shuffle(random_idx)
            # Shuffle the features and labels
            data_x, data_y = data_x[random_idx], data_y[random_idx]
            max_iter = ceil(data_size / batch_size)

            for iter in range(num_iters):
                round_step = (iter+1) % max_iter # round_step: 1, 2, ..., max_iter-1, 0
                if round_step == 0:
                    # Exceed 1 epoch
                    x_part1, y_part1 = data_x[(max_iter-1)*batch_size: data_size], \
                        data_y[(max_iter-1)*batch_size: data_size]
                    # Shuffle dataset before we get the next part
                    np.random.shuffle(random_idx)
                    data_x, data_y = data_x[random_idx], data_y[random_idx]
                    x_part2, y_part2 = data_x[0: max_iter*batch_size%data_size], \
                        data_y[0: max_iter*batch_size%data_size]

                    batched_x = np.vstack([x_part1, x_part2])
                    batched_y = np.hstack([y_part1, y_part2])  
                else:
                    batched_x = data_x[(round_step-1)*batch_size: round_step*batch_size]
                    batched_y = data_y[(round_step-1)*batch_size: round_step*batch_size]

                yield (batched_x, batched_y)

        num_samples = self.train_data['y'].shape[0]
        if num_samples == 0:
            return 0, [0], [0], self.latest_params, [np.zeros_like(ws) for ws in self.latest_params]

        backup_params = self.get_params()
        t0_weights = self.latest_params
        self.set_params(t0_weights)
        train_results = []
        for X, y in batch_data_multiple_iters(self.train_data, batch_size, num_iters):
            train_results.append(self.model.train_on_batch(X, y))
        t1_weights = self.get_params()
        gradient = [(w1-w0) for w0, w1 in zip(t0_weights, t1_weights)]
        # Roll-back the weights of model
        self.set_params(backup_params)
        if pretrain == False:
            # Store the latest local solution
            self.local_soln = t1_weights
            # Calculate the updates
            self.local_gradient = gradient
        train_acc = [rest[1] for rest in train_results]
        train_loss = [rest[0] for rest in train_results]
        
        return num_samples, train_acc, train_loss, t1_weights, gradient

    def apply_update(self, update):
        '''
        Apply update to model and Refresh the latest_params and latest_updates
        Return:
            1, Latest model params
        '''
        t0_weights = self.get_params()
        t1_weights = [(w0+up) for up, w0 in zip(update, t0_weights)]
        self.set_params(t1_weights) # The group training model is set to new weights.
        # Refresh the latest_params and latest_updates attrs
        self.latest_updates = update
        self.latest_params = t1_weights
        return self.latest_params
    
    def fresh_latest_params_updates(self, update):
        '''
        Call this function to fresh the latest_params and latst_updates
        The update will not apply to self.model, compare to apply_update()
        '''
        prev_params = self.latest_params
        latest_params = [(w0+up) for up, w0 in zip(update, prev_params)]
        self.latest_updates = update
        self.latest_params = latest_params
        return self.latest_params, self.latest_updates
    
    def test_locally(self):
        '''
        Test the model (self.latest_params) on local test dataset
        Return: Number of test samples, test accuracy, test loss
        '''
        if self.test_data['y'].shape[0] > 0:
            # Backup the current model params
            backup_params = self.get_params()
            # Set the current model to actor's params
            self.set_params(self.latest_params)
            X, y_true = self.test_data['x'], self.test_data['y']
            loss, acc = self.model.evaluate(X, y_true, verbose=0)
            # Recover the model
            self.set_params(backup_params)
            return self.test_data['y'].shape[0], acc, loss
        else:
            return 0, 0, 0

    def has_uplink(self):
        if len(self.uplink) > 0:
            return True
        return False

    def has_downlink(self):
        if len(self.downlink) > 0:
            return True
        return False

    def add_downlink(self, nodes):
        if isinstance(nodes, list):
            # Note: The repetitive node is not allow
            self.downlink = list(set(self.downlink + nodes))
        if isinstance(nodes, Actor):
            self.downlink = list(set(self.downlink + [nodes]))
        return

    def add_uplink(self, nodes):
        if isinstance(nodes, list):
            self.uplink = list(set(self.uplink + nodes))
        if isinstance(nodes, Actor):
            self.uplink = list(set(self.uplink + [nodes]))
        return
    
    def delete_downlink(self, nodes):
        if isinstance(nodes, list):
            self.downlink = [c for c in self.downlink if c not in nodes]
        if isinstance(nodes, Actor):
            self.downlink.remove(nodes)
        return

    def delete_uplink(self, nodes):
        if isinstance(nodes, list):
            self.uplink = [c for c in self.uplink - nodes if c not in nodes]
        if isinstance(nodes, Actor):
            self.uplink.remove(nodes)
        return

    def clear_uplink(self):
        self.uplink.clear()
        return

    def clear_downlink(self):
        self.downlink.clear()
        return

    def set_uplink(self, nodes):
        self.clear_uplink()
        self.add_uplink(nodes)
        return

    def check_selected_trainable(self, selected_nodes):
        ''' 
        Check The selected nodes whether can be trained, and return valid trainable nodes
        '''
        nodes_trainable = False
        valid_nodes = []
        for node in selected_nodes:
            if node in self.downlink:
                if node.check_trainable() == True:
                    nodes_trainable = True
                    valid_nodes.append(node)
        return nodes_trainable, valid_nodes

    def check_selected_testable(self, selected_nodes):
        ''' 
        Check The selected nodes whether can be tested 
        '''
        nodes_testable = False
        valid_nodes = []
        for node in selected_nodes:
            if node in self.downlink:
                if node.check_testable() == True:
                    nodes_testable = True
                    valid_nodes.append(node)
        return nodes_testable, valid_nodes

    # Train() and Test() depend on actor type
    def test(self):
        return

    def train(self):
        return

    # trainable and testable depend on actor type
    def check_trainable():
        return
    def check_testable():
        return
