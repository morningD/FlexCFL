import numpy as np
from tqdm import trange, tqdm
import tensorflow as tf


from .fedbase import BaseFedarated
from flearn.utils.tf_utils import process_grad
from utils.export_csv import CSVWriter

class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using Federated avg to Train')
        self.inner_opt = tf.compat.v1.train.GradientDescentOptimizer(params['learning_rate'])
        super(Server, self).__init__(params, learner, dataset)
        self.writer = CSVWriter(params['export_filename'], 'results/'+params['dataset'])

    def train(self):
        '''Train using Federated Proximal'''
        print('Training with {} workers ---'.format(self.clients_per_round))

        csolns = []  # buffer for receiving client solutions

        # Evalute model before training
        for i in range(self.num_rounds):

            diffs = [0] # Record the client diff

            # test model
            if i % self.eval_every == 0:
                stats = self.test()  # have set the latest model for all clients
                stats_train = self.train_error_and_loss()
                
                test_acc = np.sum(stats[3]) * 1.0 / np.sum(stats[2])
                tqdm.write('At round {} accuracy: {}'.format(i, test_acc))  # testing accuracy
                train_acc = np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2])
                tqdm.write('At round {} training accuracy: {}'.format(i, train_acc))
                train_loss = np.dot(stats_train[4], stats_train[2]) * 1.0 / np.sum(stats_train[2])
                tqdm.write('At round {} training loss: {}'.format(i, train_loss))

                # Write results to a csv file
                self.writer.write_stats(i, 0, test_acc, train_acc, train_loss, self.clients_per_round)

                # Calculate the client diff and writh it to csv file
                if csolns:
                    flat_cmodels = [process_grad(soln[1]) for soln in csolns]
                    flat_global_model = process_grad(self.latest_model)
                    diffs[0] = np.sum([np.sum((flat_model-flat_global_model)**2)**0.5 for flat_model in flat_cmodels])
                self.writer.write_diffs(diffs)
                tqdm.write('At round {} Discrepancy: {}'.format(i, diffs[0]))

            indices, selected_clients = self.select_clients(i, num_clients=self.clients_per_round)  # uniform sampling
            np.random.seed(i)
            active_clients = np.random.choice(selected_clients, round(self.clients_per_round * (1-self.drop_percent)), replace=False)

            csolns = [] # Reset the client solutions buffer
            for idx, c in enumerate(active_clients.tolist()):  # simply drop the slow devices
                # communicate the latest model
                c.set_params(self.latest_model)

                # solve minimization locally
                soln, stats = c.solve_inner(num_epochs=self.num_epochs, batch_size=self.batch_size)

                # gather solutions from client
                csolns.append(soln)

                # track communication cost
                self.metrics.update(rnd=i, cid=c.id, stats=stats)

            # update models
            self.latest_model = self.aggregate(csolns)
        self.writer.close()

        # final test model
        stats = self.test()
        stats_train = self.train_error_and_loss()
        self.metrics.accuracies.append(stats)
        self.metrics.train_accuracies.append(stats_train)
        tqdm.write('At round {} accuracy: {}'.format(self.num_rounds, np.sum(stats[3]) * 1.0 / np.sum(stats[2])))
        tqdm.write('At round {} training accuracy: {}'.format(self.num_rounds, np.sum(stats_train[3]) * 1.0 / np.sum(stats_train[2])))
