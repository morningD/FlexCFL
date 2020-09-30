import os
import numpy as np

class CSVWriter(object):
    def __init__(self, csv_filename, csv_dir, gids=None):
        csv_path = os.path.join(csv_dir, csv_filename)
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
        self.csv_f = open(csv_path, 'w', buffering=1)
        self.current_round = -1
        self.group_delimiter = '[*]\t'

        header = self.make_header(gids)
        self.csv_f.write(header)
    
    def make_header(self, gids):
        is_group = True
        if gids == None:
            # fedavg or fedprox does not have group
            # To simplify, we assume they have ONE group and gid is 0
            gids = [0]
            is_group = False
        header = '\t'
        for gid in gids:
            for _ in range(4):
                header += 'GroupID{:1d}\t'.format(gid)
            header += self.group_delimiter
        header += '\n'
        header += 'Round\t'
        for _ in gids:
            header += 'TestAcc\tTrainAcc\tTrainLoss\tNumClient\t'
            header += self.group_delimiter
        if is_group:
            header += 'MeanTestAcc\tMeanTrainAcc\t'
            header += 'GroupDiff\t'
        else:
            header += 'ClientDiff\t'
        return header

    def write_stats(self, round, gid, test_acc, train_acc, train_loss, num_client):
        if round != self.current_round:
            # New line
            self.csv_f.write('\n')
            self.csv_f.write('{:3d}\t'.format(round))
            self.current_round = round
        
        info = '{:.3f}\t{:.3f}\t{:.4f}\t{:2d}\t'.format(
            test_acc, train_acc, train_loss, num_client)
        info += self.group_delimiter
        self.csv_f.write(info)

    def write_means(self, mean_test, mean_train):
        info = '{:.3f}\t{:.3f}\t'.format(mean_test, mean_train)
        self.csv_f.write(info)

    def write_diffs(self, diffs):
        info = ''
        for diff in diffs:
            info += '{:.4f}\t'.format(diff)
        self.csv_f.write(info)

    def close(self):
        self.csv_f.close()

    def __del__(self):
        self.close()
