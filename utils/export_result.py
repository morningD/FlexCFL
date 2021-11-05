import os
import numpy as np
import pandas as pd

class ResultWriter(object):
    def __init__(self, train_config):
        filename = self.make_filename(train_config)
        dir = train_config.results_path
        self.filepath = os.path.join(dir, filename)
        if not os.path.exists(dir):
            os.makedirs(dir)

         # Write the file every 10 entry
        self.write_every = 10
        self.count = 0
        self.num_rounds = train_config.trainer_config['num_rounds']
        self.eval_every = train_config.trainer_config['eval_every']
        self.trainer_type = train_config.trainer_type
        self.migration = train_config.trainer_config['dynamic']
        if 'num_group' in train_config.trainer_config:
            self.num_group = train_config.trainer_config['num_group']

        header = self.make_header()
        index = self.make_index()
        idx = pd.Index(index, name='Round')
        self.df = pd.DataFrame(index=idx, columns=header)
    
    def make_header(self):
        if self.trainer_type == 'fedavg':
            header = ['TestAcc', 'TrainAcc', 'TrainLoss', 'NumClient', 'Discrepancy']
        if self.trainer_type in ['fedgroup', 'ifca', 'fesem']:
            header = ['WeightedTestAcc', 'WeightedTrainAcc', 'WeightedTrainLoss', 'NumGroup', 'Discrepancy']
            if self.migration == True: header += ['Shift', 'Migration']
            for gid in range(self.num_group):
                header += [f'G{gid}TestAcc', f'G{gid}TrainAcc', f'G{gid}TrainLoss', f'G{gid}Diff', f'G{gid}NumClinet']
        return header

    def make_index(self):
        eval_rounds = list(range(0, self.num_rounds, self.eval_every))
        # The last round must be included
        if (self.num_rounds-1) not in eval_rounds:
            eval_rounds.append(self.num_rounds-1)

        # We use (num_rounds+1) row to store the max test accuracy
        eval_rounds.append(self.num_rounds+1)
        return eval_rounds


    def make_filename(self, config):
        filename = ''
        trainer_type = config.trainer_type
        dataset = config.trainer_config['dataset']
        model = config.trainer_config['model']
        filename = filename + f'{trainer_type}-{dataset}-{model}'
        shift_type = config.trainer_config['shift_type']
        if shift_type in ['all', 'part']:
            swap = config.trainer_config['swap_p']
            if swap > 0 and swap < 1: filename += f'-{shift_type}_swap{swap}'
        if shift_type  == 'increment':
            filename += f'-incr'
        if trainer_type == 'fedgroup':
            measure = config.trainer_config['measure']
            num_group = config.trainer_config['num_group']
            RAC = config.trainer_config['RAC']
            RCC = config.trainer_config['RCC']
            temp = config.client_config['temperature']
            temp_metrics = config.trainer_config['temp_metrics']
            temp_func = config.trainer_config['temp_func']
            dynamic = config.trainer_config['dynamic']
            agglr = config.trainer_config['group_agg_lr']
            temp_agg = config.trainer_config['temp_agg']
            filename = filename + f'-FG{num_group}-{measure}-agglr{agglr}-tempagg{temp_agg}'
            if dynamic == True: 
               if temp is not None: filename += f'-TEMP{temp}-{temp_metrics}-{temp_func}'
            else:
                filename += '-static'
            if RAC == True: filename += '-RAC'
            if RCC == True: filename += '-RCC'
        filename += '.xlsx'
        return filename

    # result should like ['TestAcc', 'TrainAcc', 'TrainLoss', 'NumClient', 'Discrepancy']
    def write_row(self, round, result):
        
        if self.trainer_type == 'fedavg':
            test_acc = 'TestAcc'
        if self.trainer_type in ['fedgroup', 'ifca', 'fesem']:
            test_acc = 'WeightedTestAcc'

        self.df.loc[round] = result
        if self.count % self.write_every == 0:
            self.df.to_excel(self.filepath)
        self.count += 1
        # Summary the result and write
        if round == (self.num_rounds-1):
            max_test_acc = np.max(self.df[test_acc])
            print(f"The Max Test Accuracy is {max_test_acc}!")
            # Save the max accuracy to num_rounds+1 row
            self.df.loc[self.num_rounds+1][test_acc] = max_test_acc
            self.df.to_excel(self.filepath)
        return

    def write_summary(self, round, train_summary, test_summary, diffs, schedule_results=None):
        num_sublink, train_acc, train_loss = train_summary['Total']
        _, test_acc, _ = test_summary['Total']
        discrepancy = diffs['Total']
        row = [test_acc, train_acc, train_loss, num_sublink, discrepancy]
        if self.migration == True and schedule_results is not None:
            shift, migration = schedule_results['shift'], schedule_results['migration']
            row += [shift, migration]
        for gid in range(self.num_group):
            if f'G{gid}' in diffs: # The group has sublink clients
                train_acc, train_loss, _ = train_summary[f'G{gid}']
                _, test_acc, _ = test_summary[f'G{gid}']
                num_clients, discrepancy = diffs[f'G{gid}']
            else: # The group has not client this round
                num_clients = 0
                train_acc, train_loss, test_acc, discrepancy = np.nan, np.nan, np.nan, np.nan
                
            row += [test_acc, train_acc, train_loss, discrepancy, num_clients]
        self.write_row(round, row)
        return

    def __del__(self):
        self.df.to_excel(self.filepath)
        return
