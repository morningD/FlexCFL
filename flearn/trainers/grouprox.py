import numpy as np
from tqdm import trange, tqdm
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from .fedbase import BaseFedarated
from flearn.optimizer.pgd import PerturbedGradientDescent
from flearn.utils.tf_utils import process_grad, process_sparse_grad
from flearn.models.client import Client
from flearn.utils.model_utils import Metrics
from flearn.models.group import Group
import random
from utils.export_csv import CSVWriter
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from collections import Counter
import time

""" This Server class is customized for Group Prox """
class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using Group prox to Train')
        self.group_list = [] # list of Group() instance
        self.group_ids = [] # list of group id
        self.num_group = params['num_group']
        self.prox = params['proximal']
        self.group_min_clients = params['min_clients']
        self.allow_empty = params['allow_empty']
        self.evenly = params['evenly']
        self.sklearn_seed = params['seed']
        self.agg_lr = params['agg_lr']
        self.RAC = params['RAC'] # Randomly Assign Clients
        self.RCC = params['RCC'] # Random Cluster Center
        if self.prox == True:
            self.inner_opt = PerturbedGradientDescent(params['learning_rate'], params['mu'])
        else:
            self.inner_opt = tf.train.GradientDescentOptimizer(params['learning_rate'])
        super(Server, self).__init__(params, learner, dataset)
        self.latest_model = self.client_model.get_params() # The global AVG model
        self.latest_update = self.client_model.get_params()

        self.create_groups()

        self.writer = CSVWriter(params['export_filename'], 'results/'+params['dataset'], self.group_ids)

    """
    initialize the Group() instants
    """
    def create_groups(self):
        self.group_list = [Group(gid, self.client_model) for gid in range(self.num_group)] # 0,1,...,num_group
        self.group_ids = [g.get_group_id() for g in self.group_list]
        self.group_cold_start(self.RCC) # init the lastest_model of all groups

    def _get_cosine_similarity(self, m1, m2):
        flat_m1 = process_grad(m1)
        flat_m2 = process_grad(m2)
        cosine = np.dot(flat_m1, flat_m2) / (
            np.sqrt(np.sum(flat_m1**2)) * np.sqrt(np.sum(flat_m2**2)))
        return cosine

    """ measure the difference between client_model and group_model """
    def measure_difference(self, client_model, group_model):
        # Strategy #1: angles (cosine) between two vectors
        diff = self._get_cosine_similarity(client_model, group_model)
        diff = 1.0 - ((diff + 1.0) / 2.0) # scale to [0, 1] then flip
        # Strategy #2: Euclidean distance between two vectors
        # diff = np.sum((client_model - group_model)**2)
        return diff

    def get_ternary_cosine_similarity_matrix(self, w, V):
        #print(type(w), type(V))
        print('Delta w shape:', w.shape, 'Matrix V shape:', V.shape)
        w, V = w.astype(np.float32), V.astype(np.float32)
        left = np.matmul(w, V) # delta_w (dot) V
        scale = np.reciprocal(np.linalg.norm(w, axis=1, keepdims=True) * np.linalg.norm(V, axis=0, keepdims=True))
        diffs = left * scale # element-wise product
        diffs = (-diffs+1.)/2. # Normalize to [0,1]
        return diffs

    def client_cold_start(self, client):
        if client.group is not None:
            print("Warning: Client already has a group: {:2d}.".format(client.group))
        
        # Training is base on the global avg model
        start_model = self.client_model.get_params() # Backup the model first
        self.client_model.set_params(self.latest_model) # Set the training model to global avg model
        
        client_model, client_update = self.pre_train_client(client)
        diff_list = [] # tuple of (group, diff)
        for g in self.group_list:
            diff_g = self.measure_difference(client_update, g.latest_update)
            diff_list.append((g, diff_g)) # w/o sort
        
        # update(Init) the diff list of client
        client.update_difference(diff_list)

        #print("client:", client.id, "diff_list:", diff_list)
        assign_group = self.group_list[np.argmin([tup[1] for tup in diff_list])]
        # Only set the group attr of client, do not actually add clients to the group
        client.set_group(assign_group)
        
        # Recovery the training model
        self.client_model.set_params(start_model)

        return
        
    """ Deal with the group cold start problem """
    def group_cold_start(self, random_centers=False):
        
        if random_centers == True:
            # Strategy #1: random pre-train num_group clients as cluster centers
            selected_clients = random.sample(self.clients, k=self.num_group)
            for c, g in zip(selected_clients, self.group_list):
                g.latest_model, g.latest_update = self.pre_train_client(c)
                c.set_group(g)
        
        if random_centers == False:
            # Strategy #2: Pre-train, then clustering the directions of clients' weights
            alpha = 20
            selected_clients = random.sample(self.clients, k=min(self.num_group*alpha, len(self.clients)))

            for c in selected_clients: c.clustering = True # Mark these clients as clustering client

            cluster = self.clustering_clients(selected_clients) # {Cluster ID: (cm, [c1, c2, ...])}
            # Init groups accroding to the clustering results
            for g, id in zip(self.group_list, cluster.keys()):
                # Init the group latest update
                new_model = cluster[id][0]
                g.latest_update = [w1-w0 for w0, w1 in zip(g.latest_model, new_model)]
                g.latest_model = new_model
                # These clients do not need to be cold-started
                # Set the "group" attr of client only, didn't add the client to group
                for c in cluster[id][1]: c.set_group(g)
        return

    """ Clustering clients by K Means"""
    def clustering_clients(self, clients, n_clusters=None, max_iter=20):
        if n_clusters is None: n_clusters = self.num_group
        # Pre-train these clients first
        csolns, cupdates = {}, {}

        # Record the execution time
        start_time = time.time()
        for c in clients:
            csolns[c], cupdates[c] = self.pre_train_client(c)
        print("Pre-training takes {}s seconds".format(time.time()-start_time))

        update_array = [process_grad(update) for update in cupdates.values()]
        update_array = np.vstack(update_array).T # shape=(n_params, n_client)
        
        # Record the execution time
        start_time = time.time()
        svd = TruncatedSVD(n_components=3, random_state=self.sklearn_seed)
        decomp_updates = svd.fit_transform(update_array) # shape=(n_params, 3)
        print("SVD takes {}s seconds".format(time.time()-start_time))
        n_components = decomp_updates.shape[-1]

        # Record the execution time
        start_time = time.time()
        diffs = []
        delta_w = update_array.T # shape=(n_client, n_params)
        diffs = self.get_ternary_cosine_similarity_matrix(delta_w, decomp_updates)
        '''
        for dir in decomp_updates.T:
            dir_diff = [self.measure_difference(cupdates[c], dir) for c in clients]
            diffs.append(dir_diff)
        diffs = np.vstack(diffs).T # shape=(n_client, 3)
        '''
        print("Ternary Cossim Matrix calculation takes {}s seconds".format(time.time()-start_time))
        
        # Record the execution time
        start_time = time.time()
        kmeans = KMeans(n_clusters, random_state=self.sklearn_seed, max_iter=max_iter).fit(diffs)
        print("Clustering takes {}s seconds".format(time.time()-start_time))
        print('Clustering Results:', Counter(kmeans.labels_))
        print('Clustering Inertia:', kmeans.inertia_)

        cluster = {} # {Cluster ID: (cm, [c1, c2, ...])}
        cluster2clients = [[] for _ in range(n_clusters)] # [[c1, c2,...], [c3, c4,...], ...]
        for idx, cluster_id in enumerate(kmeans.labels_):
            #print(idx, cluster_id, len(cluster2clients), n_clusters) # debug
            cluster2clients[cluster_id].append(clients[idx])
        for cluster_id, client_list in enumerate(cluster2clients):
            # calculate the means of cluster
            # All client have equal weight
            weighted_csolns = [(1, csolns[c]) for c in client_list]
            if weighted_csolns:
                # Update the cluster means
                cluster[cluster_id] = (self.aggregate(weighted_csolns), client_list)
            else:
                print("Error, cluster is empty")

        return cluster

    def measure_group_diffs(self):
        diffs = np.empty(len(self.group_list))
        for idx, g in enumerate(self.group_list):
            # direction
            #diff = self.measure_difference(self.group_list[0].latest_model, g.latest_model)
            # square root
            model_a = process_grad(self.latest_model)
            model_b = process_grad(g.latest_model)
            diff = np.sum((model_a-model_b)**2)**0.5
            diffs[idx] = diff
        diffs = diffs + [np.sum(diffs)] # Append the sum(discrepancies) to the end
        return diffs

    """ Pre-train the client 1 epoch and return weights """
    def pre_train_client(self, client):
        start_model = client.get_params() # Backup the start model
        if self.prox == True:
            # Set the value of vstart to be the same as the client model to remove the proximal term
            self.inner_opt.set_params(self.client_model.get_params(), self.client_model)
        soln, stat = client.solve_inner() # Pre-train the client only one epoch
        ws = soln[1] # weights of model
        updates = [w1-w0 for w0, w1 in zip(start_model, ws)]

        client.set_params(start_model) # Recovery the model
        return ws, updates

    def get_not_empty_groups(self):
        not_empty_groups = [g for g in self.group_list if not g.is_empty()]
        return not_empty_groups

    def group_test(self):
        backup_model = self.latest_model # Backup the global model
        results = []
        for g in self.group_list:
            c_list = []
            for c in self.clients:
                if c.group == g:
                    c_list.append(c)
            num_samples = []
            tot_correct = []
            self.client_model.set_params(g.latest_model)
            for c in c_list:
                ct, ns = c.test()
                tot_correct.append(ct*1.0)
                num_samples.append(ns)
            ids = [c.id for c in c_list]
            results.append((ids, g, num_samples, tot_correct))
        self.client_model.set_params(backup_model) # Recovery the model
        return results

    def group_train_error_and_loss(self):
        backup_model = self.latest_model # Backup the global model
        results = []
        for g in self.group_list:
            c_list = []
            for c in self.clients:
                if c.group == g:
                    c_list.append(c)
            num_samples = []
            tot_correct = []
            losses = []
            self.client_model.set_params(g.latest_model)
            for c in c_list:
                ct, cl, ns = c.train_error_and_loss() 
                tot_correct.append(ct*1.0)
                num_samples.append(ns)
                losses.append(cl*1.0)
            ids = [c.id for c in c_list]
            results.append((ids, g, num_samples, tot_correct, losses))
        self.client_model.set_params(backup_model) # Recovery the model
        return results

    def train(self):
        print('Training with {} workers ---'.format(self.clients_per_round))
        # Clients cold start, pre-train all clients
        for c in self.clients:
                if c.is_cold() == True:
                    self.client_cold_start(c)

        for i in range(self.num_rounds):

            # Random select clients
            indices, selected_clients = self.select_clients(i, num_clients=self.clients_per_round)  # uniform sampling
            np.random.seed(i)  # make sure that the stragglers are the same for FedProx and FedAvg
            active_clients = np.random.choice(selected_clients, round(self.clients_per_round * (1 - self.drop_percent)), replace=False)
            
            # Clear all group, the group attr of client is retrained
            for g in self.group_list: g.clear_clients()
            
            # Client cold start
            # Reshcedule selected clients to groups
            self.reschedule_groups(selected_clients, self.allow_empty, self.evenly, self.RAC)

            # Get not empty groups
            handling_groups = self.get_not_empty_groups()

            for g in self.group_list:
                if g in handling_groups:
                    print("Group {}, clients {}".format(g.get_group_id(), g.get_client_ids()))
                else:
                    print("Group {} is empty.".format(g.get_group_id()))

            # Freeze these groups before training
            for g in handling_groups:
                g.freeze()

            if i % self.eval_every == 0:
                """
                stats = self.test() # have set the latest model for all clients
                # Test on training data, it's redundancy
                stats_train = self.train_error_and_loss()
                """
                group_stats = self.group_test()
                group_stats_train = self.group_train_error_and_loss()
                test_tp, test_tot = 0, 0
                train_tp, train_tot = 0, 0
                for stats, stats_train in zip(group_stats, group_stats_train):
                    tqdm.write('Group {}'.format(stats[1].id))
                    test_tp += np.sum(stats[3])
                    test_tot += np.sum(stats[2])
                    test_acc = np.sum(stats[3])*1.0/np.sum(stats[2])
                    tqdm.write('At round {} accuracy: {}'.format(i, test_acc))  # testing accuracy
                    train_tp += np.sum(stats_train[3])
                    train_tot += np.sum(stats_train[2])
                    train_acc = np.sum(stats_train[3])*1.0/np.sum(stats_train[2])
                    tqdm.write('At round {} training accuracy: {}'.format(i, train_acc)) # train accuracy
                    train_loss = np.dot(stats_train[4], stats_train[2])*1.0/np.sum(stats_train[2])
                    tqdm.write('At round {} training loss: {}'.format(i, train_loss))
                    
                    mean_test_acc = test_tp*1.0 / test_tot
                    mean_train_acc = train_tp*1.0 / train_tot
                    
                    # Write results to csv file
                    self.writer.write_stats(i, stats[1].id, test_acc, 
                        train_acc, train_loss, len(stats[1].get_client_ids()))
                
                self.writer.write_means(mean_test_acc, mean_train_acc)
                print('At round {} mean test accuracy: {} mean train accuracy: {}'.format(
                    i, mean_test_acc, mean_train_acc))
                diffs = self.measure_group_diffs()
                print("The groups difference are:", diffs)
                self.writer.write_diffs(diffs)

            # Broadcast the global model to clients(groups)
            # self.client_model.set_params(self.latest_model)
            
            # Train each group sequentially
            for g in handling_groups:
                # Backup the origin model
                print("Begin group {:2d} training".format(g.get_group_id()))
                # Each group train group_epochs round
                for _ in range(g.group_epochs):
                    if self.prox == True:
                        # Update the optimizer, the vstar is latest_model of this group
                        self.inner_opt.set_params(g.latest_model, self.client_model)
                    # Set the global the group model
                    self.client_model.set_params(g.latest_model)
                    # Begin group training
                    cupdates = g.train()
                # After end of the training of client, update the diff list of client
                for client, update in cupdates.items():
                    diff_list = []
                    for g in self.group_list:
                        diff_g = self.measure_difference(update, g.latest_update)
                        diff_list.append((g, diff_g))
                    client.update_difference(diff_list)
                        
                # Recovery the client model before next group training
                #self.client_model.set_params(self.latest_model)

            # Aggregate groups model and update the global (latest) model 
            self.aggregate_groups(self.group_list, agg_lr=self.agg_lr)
            
            # Refresh the global model and global delta weights (latest_update)
            self.refresh_global_model(self.group_list)

        # Close the writer and end the training
        self.writer.close()

    
    # Use for matain the global AVG model and global latest update
    def refresh_global_model(self, groups):
        start_model = self.latest_model 
        # Aggregate the groups model
        gsolns = []
        for g in groups:
            gsolns.append((1.0, g.latest_model)) # (n_k, soln)
        new_model = self.aggregate(gsolns)
        self.latest_update = [w1-w0 for w0, w1 in zip(start_model, new_model)]
        self.latest_model = new_model

        return

    def aggregate_groups(self, groups, agg_lr):
        gsolns = [(sum(g.num_samples), g.latest_model) for g in groups]
        group_num = len(gsolns)
        # Calculate the scale of group models
        gscale = [0]*group_num
        for i, (_, gsoln) in enumerate(gsolns):
            for v in gsoln:
                gscale[i] += np.sum(v.astype(np.float64)**2)
            gscale[i] = gscale[i]**0.5
        # Aggregate the models of each group separately
        for idx, g in enumerate(groups):
            base = [0]*len(gsolns[idx][1])
            weights = [agg_lr*(1.0/scale) for scale in gscale]
            weights[idx] = 1 # The weight of the main group is 1
            total_weights = sum(weights)
            for j, (_, gsoln) in enumerate(gsolns):
                for k, v in enumerate(gsoln):
                    base[k] += weights[j]*v.astype(np.float64)
            averaged_soln = [v / total_weights for v in base]
            g.latest_update = [w1-w0 for w0, w1 in zip(g.latest_model, averaged_soln)]
            g.latest_model = averaged_soln

        return
  

    def reschedule_groups(self, selected_clients, allow_empty=False, evenly=False, randomly=False):
        
        def _get_even_per_group_num(selected_clients_num, group_num):
            per_group_num = np.array([selected_clients_num // group_num] * group_num)
            remain = selected_clients_num - sum(per_group_num)
            random_groups = random.sample(range(group_num), remain)
            per_group_num[random_groups] += 1 # plus the remain
            return per_group_num

        selected_clients = selected_clients.tolist() # convert numpy array to list

        if randomly==True and evenly==False:
            for c in selected_clients:
                if c.is_cold() == False:
                    if c.clustering == False:
                        # Randomly assgin client
                        random.choice(self.group_list).add_client(c)
                    else:
                        # This client is clustering client.
                        c.group.add_client(c)
                else:
                    print('Warnning: A newcomer is no pre-trained.')
            return
            
        if randomly==True and evenly==True:
            """
            # Randomly assgin client, but each group is even
            per_group_num = _get_even_per_group_num(len(selected_clients), len(self.group_list))
            for g, max in zip(self.group_list, per_group_num): g.max_clients = max
            head_idx, tail_idx = 0, 0
            for group_num, g in zip(per_group_num, self.group_list):
                tail_idx += group_num
                g.add_clients(selected_clients[head_idx, tail_idx])
                head_idx = tail_idx
            """
            print("Experimental setting is invalid.")
            return 

        if randomly==False and allow_empty==True:
            # Allocate clients to their first rank groups, some groups may be empty
            for c in selected_clients:
                if c.is_cold() != True:
                    first_rank_group = c.group
                    first_rank_group.add_client(c)
            return
        
        if randomly==False and evenly==True:
            """ Strategy #1: Calculate the number of clients in each group (evenly) """
            selected_clients_num = len(selected_clients)
            group_num = len(self.group_list)
            per_group_num = np.array([selected_clients_num // group_num] * group_num)
            remain = selected_clients_num - sum(per_group_num)
            random_groups = random.sample(range(group_num), remain)
            per_group_num[random_groups] += 1 # plus the remain

            for g, max in zip(self.group_list, per_group_num):
                g.max_clients = max

            """ Allocate clients to make the client num of each group evenly """
            for c in selected_clients:
                if c.is_cold() != True:
                    first_rank_group = c.group
                    if not first_rank_group.is_full():
                        first_rank_group.add_client(c)
                    else:
                        # The first rank group is full, choose next group
                        diff_list = c.difference
                        # Sort diff_list
                        diff_list = sorted(diff_list, key=lambda tup: tup[1])
                        for (group, diff) in diff_list:
                            if not group.is_full():
                                group.add_client(c)
                                break
            return

        if randomly==False and evenly==False:
            """ Strategy #2: Allocate clients to meet the minimum client requirements """
            for g in self.group_list: g.min_clients = self.group_min_clients
            # First ensure that each group has at least self.min_clients clients.
            diff_list, assigned_clients = [], []
            for c in selected_clients:
                diff_list += [(c, g, diff) for g, diff in c.difference]
            diff_list = sorted(diff_list, key=lambda tup: tup[2])
            for c, g, diff in diff_list:
                if len(g.client_ids) < g.min_clients and c not in assigned_clients:
                    g.add_client(c)
                    assigned_clients.append(c)
                               
            # Then add the remaining clients to their first rank group
            for c in selected_clients:
                if c not in assigned_clients:
                    first_rank_group = c.group
                    if c.id not in first_rank_group.client_ids:
                        first_rank_group.add_client(c)
            return

        return

    def test_ternary_cosine_similariy(self, alpha=20):
        ''' compare the ternary similarity and cosine similarity '''
        def calculate_cosine_distance(v1, v2):
            cosine = np.dot(v1, v2) / (np.sqrt(np.sum(v1**2)) * np.sqrt(np.sum(v2**2)))
            return cosine

        # Pre-train all clients
        csolns, cupdates = {}, {}
        for c in self.clients:
            csolns[c], cupdates[c] = self.pre_train_client(c)

        # random selecte alpha * m clients to calculate the direction matrix V
        n_clients = len(self.clients)
        clustering_clients = random.sample(self.clients, k=min(self.num_group*alpha, n_clients))
        clustering_update_array = [process_grad(cupdates[c]) for c in clustering_clients]
        clustering_update_array = np.vstack(clustering_update_array).T # shape=(n_params, n_clients)
        
        svd = TruncatedSVD(n_components=3, random_state=self.sklearn_seed)
        decomp_updates = svd.fit_transform(clustering_update_array) # shape=(n_params, 3)
        n_components = decomp_updates.shape[-1]

        # calculate the ternary similarity matrix for all clients
        ternary_cossim = []
        update_array = [process_grad(cupdates[c]) for c in self.clients]
        delta_w = np.vstack(update_array) # shape=(n_clients, n_params)
        ternary_cossim = self.get_ternary_cosine_similarity_matrix(delta_w, decomp_updates)

        # calculate the tranditional similarity matrix for all clients
        #old_cossim = np.zeros(shape=(n_clients, n_clients), dtype=np.float32)

        old_cossim = cosine_similarity(delta_w)
        old_cossim = (1.0 - old_cossim) / 2.0 # Normalize

        # Calculate the euclidean distance between every two similaries
        distance_ternary = euclidean_distances(ternary_cossim)
        distance_cossim = euclidean_distances(old_cossim) # shape=(n_clients, n_clients)
        print(distance_ternary.shape, distance_cossim.shape) # shape=(n_clients, n_clients)

        iu = np.triu_indices(n_clients)
        x, y = distance_ternary[iu], distance_cossim[iu]
        mesh_points = np.vstack((x,y)).T

        print(x.shape, y.shape)
        np.savetxt("cossim.csv", mesh_points, delimiter="\t")
        return x, y