import numpy as np
import tensorflow as tf

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
        self.latest_params, self.latest_updates = None, None
        # init train and test size to zero, it will depend on the actor type
        self.train_size, self.test_size = 0, 0 
        self.uplink, self.downlink = [], [] # init to empty, depend on the actor type

        self.preprocess()

    def preprocess(self):
        self.name = str(self.actor_type) + str(self.id)
        self.latest_params = self.get_params()
        self.latest_updates = [np.zeros_like(ws) for ws in self.latest_params]

    def get_params(self):
        if self.model:
            return self.model.get_weights()
    
    def set_params(self, weights):
        if self.model:
            self.model.set_weights(weights)
            
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

    def solve_inner(self, num_epoch=1, batch_size=10):
        '''
        Solve the local optimization base on local training data
        
        Return: num_samples, train_acc, train_loss, Updates
        '''
        if self.train_data['y'].shape[0] > 0:
            X, y_true = self.train_data['x'], self.train_data['y']
            num_samples = y_true.shape[0]
            t0_weights = self.get_params()
            history = self.model.fit(X, y_true, batch_size, num_epoch, verbose=0)
            t1_weights = self.get_params()
            
            # Roll-back the weights
            self.set_params(t0_weights)
            # Calculate the updates
            updates = [(w1-w0) for w0, w1 in zip(t0_weights, t1_weights)]
            # Get the train accuracy and train loss
            print(history.history)
            train_acc = history.history['accuracy']
            train_loss = history.history['loss']
            print(train_acc) # Debug
            
            return num_samples, train_acc, train_loss, updates
        else:
            # Return 0,0,0 and all zero updates [0, 0, ...],
            # if this actor has not training set
            return 0, [0], [0], [np.zeros_like(ws) for ws in self.latest_params]
    
    def test_locally(self):
        '''
        Test the model on local test dataset
        '''
        if self.test_data['y'].shape[0] > 0:
            X, y_true = self.test_data['x'], self.test_data['y']
            acc, loss = self.model.evaluate(X, y_true, verbose=0)
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
        # Note: The repetitive node is not allow
        self.downlink = list(set(self.downlink + nodes))
        return

    def add_uplink(self, nodes):
        self.uplink = list(set(self.uplink + nodes))
        return
    
    def delete_downlink(self, nodes):
        self.downlink = [c for c in self.downlink if c not in nodes]
        return

    def delete_uplink(self, nodes):
        self.uplink = [c for c in self.uplink - nodes if c not in nodes]
        return

    # Train() and Test() depend on actor type
    def test(self):
        return

    def train(self):
        return

