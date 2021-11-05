import numpy as np
import random
import time
from termcolor import colored
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering

from utils.trainer_utils import process_grad, calculate_cosine_dissimilarity
#from flearn.model.mlp import construct_model
from flearn.trainer.groupbase import GroupBase
from collections import Counter

class FedGroup(GroupBase):
    def __init__(self, train_config):
        super(FedGroup, self).__init__(train_config)
        self.group_cold_start(random_centers=self.RCC)
        if self.temp_agg == True: 
            for g in self.groups: g.aggregation_strategy = 'temp'
        else:
            for g in self.groups: g.aggregation_strategy = 'fedavg'

    """ Cold strat all groups when create the trainer
    """
    def group_cold_start(self, alpha=20, clients=None, random_centers=False):

        # Clustering with all clients by default
        if clients is None: clients = self.clients

        # Strategy #1 (RCC): Randomly pre-train num_group clients as cluster centers
        # It is an optional strategy of FedGroup, named FedGroup-RCC
        if random_centers == True:
            print('Random Cluster Centers.')
            selected_clients = random.sample(clients, k=self.num_group)
            for c, g in zip(selected_clients, self.groups):
                _, _, _, g.latest_params, g.opt_updates = c.pretrain(self.init_params, iterations=50)
                g.latest_updates = g.opt_updates
                c.set_uplink([g])
                g.add_downlink([c])

        # Strategy #2: Pre-train, then clustering the directions of clients' weights
        # <FedGroup> and <FedGrouProx> use this strategy
        if random_centers == False:
            selected_clients = random.sample(clients, k=min(self.num_group*alpha, len(clients)))

            for c in selected_clients: c.clustering = True # Mark these clients as clustering client

            cluster = self.clustering_clients(selected_clients) # {Cluster ID: (cm, [c1, c2, ...])}
            # Init groups accroding to the clustering results
            for g, id in zip(self.groups, cluster.keys()):
                # Init the group latest update
                g.latest_params = cluster[id][0]
                g.opt_updates = cluster[id][1]
                g.latest_updates = g.opt_updates
                # These clients do not need to be cold-started
                # Set the "group" attr of client only, didn't add the client to group
                g.add_downlink(cluster[id][2])
                for c in cluster[id][2]:
                    c.set_uplink([g])

            # We aggregate these clustering results and get the new auxiliary global model
            self.update_auxiliary_global_model(self.groups)
            # Update the discrepancy of clustering client
            '''self.refresh_discrepancy_and_dissmilarity(selected_clients)'''
        return

    """ Clustering clients 
        Return: {Cluster ID: (parameter mean, update mean, client_list ->[c1, c2, ...])}
    """
    def clustering_clients(self, clients, n_clusters=None, max_iter=20):
        if n_clusters is None: n_clusters = self.num_group
        if len(clients) < n_clusters: 
            print("ERROR: Not enough clients for clustering!!")
            return

        # Pre-train these clients first
        csolns, cupdates = {}, {}

        # Record the execution time
        start_time = time.time()
        for c in clients:
            _, _, _, csolns[c], cupdates[c] = c.pretrain(self.init_params, iterations=50)
        print("Pre-training takes {}s seconds".format(time.time()-start_time))

        update_array = [process_grad(update) for update in cupdates.values()]
        delta_w = np.vstack(update_array) # shape=(n_clients, n_params)
        
        # Record the execution time
        start_time = time.time()
        # Decomposed the directions of updates to num_group of directional vectors
        svd = TruncatedSVD(n_components=self.num_group, random_state=self.seed)
        decomp_updates = svd.fit_transform(delta_w.T) # shape=(n_params, n_groups)
        print("SVD takes {}s seconds".format(time.time()-start_time))
        n_components = decomp_updates.shape[-1]

        # Record the execution time of EDC calculation
        start_time = time.time()
        decomposed_cossim_matrix = cosine_similarity(delta_w, decomp_updates.T) # shape=(n_clients, n_clients)

        ''' There is no need to normalize the data-driven measure because it is a dissimilarity measure
        # Normialize it to dissimilarity [0,1]
        decomposed_dissim_matrix = (1.0 - decomposed_cossim_matrix) / 2.0
        EDC = decomposed_dissim_matrix
        '''
        #EDC = self._calculate_data_driven_measure(decomposed_cossim_matrix, correction=False)
        print("EDC Matrix calculation takes {}s seconds".format(time.time()-start_time))
        
        # Test the excution time of full cosine dissimilarity
        start_time = time.time()
        full_cossim_matrix = cosine_similarity(delta_w) # shape=(n_clients, n_clients)
        '''
        # Normialize cossim to [0,1]
        full_dissim_matrix = (1.0 - full_cossim_matrix) / 2.0
        '''
        MADC = self._calculate_data_driven_measure(full_cossim_matrix, correction=True) # shape=(n_clients, n_clients)
        print("MADC Matrix calculation takes {}s seconds".format(time.time()-start_time))

        '''Apply RBF kernel to EDC or MADC
        gamma=0.2
        if self.MADC == True:
            affinity_matrix = np.exp(- MADC ** 2 / (2. * gamma ** 2))
        else: # Use EDC as default
            affinity_matrix = np.exp(- EDC ** 2 / (2. * gamma ** 2))
        '''
        # Record the execution time
        start_time = time.time()
        if self.measure == 'MADC':
            affinity_matrix = MADC
            #affinity_matrix = (1.0 - full_cossim_matrix) / 2.0
            #result = AgglomerativeClustering(n_clusters, affinity='euclidean', linkage='ward').fit(full_cossim_matrix)
            result = AgglomerativeClustering(n_clusters, affinity='precomputed', linkage='complete').fit(affinity_matrix)
        if self.measure == 'EDC': # Use EDC as default
            affinity_matrix = decomposed_cossim_matrix
            #result = AgglomerativeClustering(n_clusters, affinity='euclidean', linkage='ward').fit(decomposed_cossim_matrix)
            #result = AgglomerativeClustering(n_clusters, affinity='precomputed', linkage='average').fit(EDC)
            result = KMeans(n_clusters, random_state=self.seed, max_iter=max_iter).fit(affinity_matrix)
        #print('EDC', EDC[0][:10], '\nMADC', MADC[0][:10], '\naffinity', affinity_matrix[0][:10])
        #result = SpectralClustering(n_clusters, random_state=self.seed, n_init=max_iter, affinity='precomputed').fit(affinity_matrix)

        print("Clustering takes {}s seconds".format(time.time()-start_time))
        print('Clustering Results:', Counter(result.labels_))
        #print('Clustering Inertia:', result.inertia_)

        cluster = {} # {Cluster ID: (cm, [c1, c2, ...])}
        cluster2clients = [[] for _ in range(n_clusters)] # [[c1, c2,...], [c3, c4,...], ...]
        for idx, cluster_id in enumerate(result.labels_):
            #print(idx, cluster_id, len(cluster2clients), n_clusters) # debug
            cluster2clients[cluster_id].append(clients[idx])
        for cluster_id, client_list in enumerate(cluster2clients):
            # calculate the means of cluster
            params_list = [csolns[c] for c in client_list]
            updates_list = [cupdates[c] for c in client_list]
            if params_list:
                 # All client have equal weight
                cluster[cluster_id] = (self.simply_averaging_aggregate(params_list),\
                    self.simply_averaging_aggregate(updates_list), client_list)
            else:
                print("Error, cluster is empty")

        return cluster

    def _calculate_data_driven_measure(self, pm, correction=False):
            ''' calculate the data-driven measure such as MADD'''
            # Input: pm-> proximity matrix; Output: dm-> data-driven distance matrix
            # pm.shape=(n_clients, n_dims), dm.shape=(n_clients, n_clients)
            n_clients, n_dims = pm.shape[0], pm.shape[1]
            dm = np.zeros(shape=(n_clients, n_clients))
            
            """ Too Slow, and misunderstanding MADD. Deprecated
            for i in range(n_clients):
                for j in range(i+1, n_clients):
                    for k in range(n_clients):
                        if k !=i and k != j:
                            dm[i,j] = dm[j,i] = abs(np.sum((pm[i]-pm[k])**2)**0.5 - \
                                np.sum((pm[j]-pm[k])**2)**0.5)
            """
            # Fast version
            '''1, Get the repeated proximity matrix.
                We write Row1 = d11, d12, d13, ... ; and Row2 = d21, d22, d23, ...
                [   Row1    ]   [   Row2    ]       [   Rown    ]
                |   Row1    |   |   Row2    |       |   Rown    |
                |   ...     |   |   ...     |       |   ...     |
                [   Row1    ],  [   Row2    ], ..., [   Rown    ]
            '''
            row_pm_matrix = np.repeat(pm[:,np.newaxis,:], n_clients, axis=1)
            #print('row_pm', row_pm_matrix[0][0][:5], row_pm_matrix[0][1][:5])

            # Get the repeated colum proximity matrix
            '''
                [   Row1    ]   [   Row1    ]       [   Row1    ]
                |   Row2    |   |   Row2    |       |   Row2    |
                |   ...     |   |   ...     |       |   ...     |
                [   Rown    ],  [   Rown    ], ..., [   Rown    ]
            '''
            col_pm_matrix = np.tile(pm, (n_clients, 1, 1))
            #print('col_pm', col_pm_matrix[0][0][:5], col_pm_matrix[0][1][:5])
            
            # Calculate the absolute difference of two disstance matrix, It is 'abs(||u-z|| - ||v-z||)' in MADD.
            # d(1,2) = ||w1-z|| - ||w2-z||, shape=(n_clients,); d(x,x) always equal 0
            '''
                [   d(1,1)  ]   [   d(1,2)  ]       [   d(1,n)  ]
                |   d(2,1)  |   |   d(2,2)  |       |   d(2,n)  |
                |   ...     |   |   ...     |       |   ...     |
                [   d(n,1)  ],  [   d(n,2)  ], ..., [   d(n,n)  ]
            '''
            absdiff_pm_matrix = np.abs(col_pm_matrix - row_pm_matrix) # shape=(n_clients, n_clients, n_clients)
            # Calculate the sum of absolute differences
            if correction == True:
                # We should mask these values like sim(1,2), sim(2,1) in d(1,2)
                mask = np.zeros(shape=(n_clients, n_clients))
                np.fill_diagonal(mask, 1) # Mask all diag
                mask = np.repeat(mask[np.newaxis,:,:], n_clients, axis=0)
                for idx in range(mask.shape[-1]):
                    #mask[idx,idx,:] = 1 # Mask all row d(1,1), d(2,2)...; Actually d(1,1)=d(2,2)=0
                    mask[idx,:,idx] = 1 # Mask all 0->n colum for 0->n diff matrix,
                dm = np.sum(np.ma.array(absdiff_pm_matrix, mask=mask), axis=-1) / (n_dims-2.0)
            else:
                dm = np.sum(absdiff_pm_matrix, axis=-1) / (n_dims)
            #print('absdiff_pm_matrix', absdiff_pm_matrix[0][0][:5])

            return dm # shape=(n_clients, n_clients)

    '''Rewrite the schedule client function of GroupBase,
        This function will be call before traning.
    '''
    def schedule_clients(self, round, selected_clients, groups):
        schedule_results = None
        if self.dynamic == True:
            # 1, Redo cold start distribution shift clients
            warm_clients = [wc for wc in self.clients if wc.has_uplink() == True]
            shift_count, migration_count = 0, 0
            for client in warm_clients:
                count = client.check_distribution_shift()
                if count is not None and client.distribution_shift == True:
                    shift_count += 1
                    prev_g = client.uplink[0]
                    prev_g.delete_downlink(client)
                    client.clear_uplink()
                    self.client_cold_start(client)
                    new_g = client.uplink[0]
                    client.train_label_count = count
                    client.distribution_shift = False
                    if prev_g != new_g:
                        migration_count += 1
                        print(colored(f'Client {client.id} migrate from Group {prev_g.id} \
                            to Group {new_g.id}', 'yellow', attrs=['reverse']))
            schedule_results = {'shift': shift_count, 'migration': migration_count}

        # 2, Cold start newcomer: pretrain and assign a group
        for client in selected_clients:
        #for client in self.clients:
            if client.has_uplink() == False:
                self.client_cold_start(client, self.RAC)

        return schedule_results

    ''' Rewrite the schedule group function of GroupBase '''
    def schedule_groups(self, round, clients, groups):
        if self.dynamic == True and self.recluster_epoch is not None:
            # Reculster warm client
            if round in self.recluster_epoch:
                warm_clients = [c for c in clients if c.has_uplink() == True]
                self.recluster(warm_clients, groups)
        return

    """ Reculster () clients and cold start (reassign group) the remain clients
    """
    def recluster(self, clients, groups, alpha=20):
        if len(groups) != len(self.groups):
            print("Warning: Group Number is change!")
            # TODO: dynamic group num
            return 

        print('Reclustering...')
        # Clear the clustering mark
        for c in clients: c.clustering = False

        # Select the clients for clustering first
        selected_clients = random.sample(clients, k=min(len(groups)*alpha, len(clients)))
        remain_clients = [c for c in clients if c not in selected_clients]
        self.group_cold_start(clients=selected_clients)
        for c in remain_clients:
            # Reassign (cold start) the remain clients
            old_group = c.uplink[0]
            old_group.delete_downlink(c)
            c.clear_uplink()
            self.client_cold_start(c, self.RAC, redo=False)

        # Refresh the discrepancy of all clients (clustering clients and reassign clients)
        '''self.refresh_discrepancy_and_dissmilarity(clients)'''
        return 

    def client_cold_start(self, client, random_assign=False, redo=False):
        if client.has_uplink() == True:
            print("Warning: Client already has a group: {:2d}.".format(client.uplink[0].id))
            return

        else:
            _, _, _, csoln, cupdate = client.pretrain(self.init_params, iterations=50)

            # Calculate the cosine dissimilarity between client's update and group's update
            diff_list = []
            for g in self.groups:
                if redo == False:
                    opt_updates = g.opt_updates
                else:
                    opt_updates = g.latest_updates
                diff = calculate_cosine_dissimilarity(cupdate, opt_updates)
                diff_list.append((g, diff))
            if random_assign == True:
                # RAC: Randomly assign client
                assign_group = random.choice(self.groups)
            else:
                # Minimize the differenct
                assign_group = self.groups[np.argmin([tup[1] for tup in diff_list])]
            
            # Set the uplink of client, add the downlink of group
            client.set_uplink([assign_group])
            assign_group.add_downlink([client])

            # Reset the temperature
            client.temperature = client.max_temp
            #print(f'Assign client {client.id} to Group {assign_group.id}!')
        return assign_group

    def reassign_clients_by_temperature(self, clients, metrics, func):

        def _step_temperature(client_bias, group_bias, temp, max_temp):
            if temp <=0: return temp
            if client_bias > group_bias:
                return temp-1
            else:
                return min(temp+1, max_temp)
                #return min(temp+0, max_temp)

        def _linear_temperature(client_bias, group_bias, temp, max_temp):
            if temp <=0: return temp
            if group_bias == 0: return temp
            scale = client_bias / (group_bias + 1e-5)
            new_temp = temp + (1 - scale) * max_temp
            return min(new_temp, max_temp)

        def _lied_temperature(client_bias, group_bias, temp, max_temp):
            if temp <=0: return temp
            if client_bias > group_bias: # Temperature exponential decrease
                rate = 2 * max_temp / (max_temp - 1)
                scale = min(client_bias / (group_bias + 1e-5), 10) # Prevent Overflow
                new_temp = temp - (max_temp ** (scale - 1) - 1) * rate
                return new_temp
            else: # Temperature linear increase
                return _linear_temperature(client_bias, group_bias, temp, max_temp)

        def _eied_temperature(client_bias, group_bias, temp, max_temp):
            if temp <=0: return temp
            sign = 1 if client_bias <= group_bias else -1
            abs_bias = min(abs(client_bias - group_bias), 10) # Prevent Overflow
            rate = 2 * max_temp / (max_temp - 1)
            new_temp = temp + sign * (max_temp ** (abs_bias - 1) - 1) * rate
            return min(new_temp, max_temp)

        # Reassgin selected wram clients when their temperature reduced to zero
        for wc in [c for c in clients if c.has_uplink() == True and c.temperature is not None]:
            
            # L2 distance or Cosine Dissimilarity
            if metrics == 'l2': 
                client_bias, group_bias = wc.discrepancy, wc.uplink[0].discrepancy
            if metrics == 'cosine': 
                client_bias, group_bias = wc.cosine_dissimilarity, wc.uplink[0].cosine_dissimilarity
            #print('debug: fedgroup.py:366', client_bias, group_bias)
            # The discrepancy of this client large than the mean discrepancy of group
            if func == 'step':
                wc.temperature = _step_temperature(client_bias, group_bias, wc.temperature, wc.max_temp)
            if func == 'linear':
                wc.temperature = _linear_temperature(client_bias, group_bias, wc.temperature, wc.max_temp)
            if func == 'lied':
                wc.temperature = _lied_temperature(client_bias, group_bias, wc.temperature, wc.max_temp)
            if func == 'eied':
                wc.temperature = _eied_temperature(client_bias, group_bias, wc.temperature, wc.max_temp)

            if wc.temperature < 0: wc.temperature = 0
            ''' Redo cold start if temperature less than 0
            if wc.temperature <= 0:
                # Clear the link between client and group if its temperature below 0
                old_group = wc.uplink[0]
                old_group.delete_downlink(wc)
                wc.clear_uplink()
                # Cold start this client, the temperature will be reset
                new_group = self.client_cold_start(wc, self.RAC, redo=False)
                if old_group != new_group:
                    print(colored(f'Client {wc.id} migrate from Group {old_group.id} to Group {new_group.id}', 'yellow', attrs=['reverse']))
            '''
        return

    def schedule_clients_after_training(self, comm_round, clients, groups):
        # Refresh selected clients' temperature
        self.reassign_clients_by_temperature(clients, self.temp_metrics, self.temp_func)
        return