import pdb
import pickle
import sys
# update your projecty root path before running
import traceback
from collections import defaultdict
import os
from io import BytesIO

from openpyxl import load_workbook
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

dir_path = os.path.dirname(os.path.realpath(__file__))

sys.path.insert(0, f'{dir_path}/..')
sys.path.insert(0, dir_path)
from models.macro_decoder import ResidualNode
from search.micro_encoding import make_micro_creator
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import time
import logging
import argparse
from misc import utils

import numpy as np
from search import train_search
from search import micro_encoding
from search import macro_encoding
from search import nsganet as engine
from sacred.observers import MongoObserver
from sacred import Experiment
from pymop.problem import Problem
from pymoo.optimize import minimize
from config import config_dict, set_config
import pandas as pd
parser = argparse.ArgumentParser("Multi-objetive Genetic Algorithm for NAS")
parser.add_argument('--save', type=str, default='GA-BiObj', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--search_space', type=str, default='micro', help='macro or micro search space')
# arguments for micro search space
parser.add_argument('--n_blocks', type=int, default=5, help='number of blocks in a cell')
parser.add_argument('--n_ops', type=int, default=9, help='number of operations considered')
parser.add_argument('--n_cells', type=int, default=2, help='number of cells to search')
# arguments for macro search space
parser.add_argument('--n_nodes', type=int, default=4, help='number of nodes per phases')
# hyper-parameters for algorithm
parser.add_argument('--pop_size', type=int, default=40, help='population size of networks')
parser.add_argument('--n_gens', type=int, default=50, help='population size')
parser.add_argument('--n_offspring', type=int, default=40, help='number of offspring created per generation')
# arguments for back-propagation training during search
parser.add_argument('--init_channels', type=int, default=24, help='# of filters for first cell')
parser.add_argument('--layers', type=int, default=11, help='equivalent with N = 3')
parser.add_argument('--epochs', type=int, default=25, help='# of epochs to train during architecture search')
parser.add_argument('--datasets', type=str, default='Cricket', help='datasets to run on')
parser.add_argument('--iterations', type=int, default=3, help='times to run each experiment')
parser.add_argument('--batch_size', type=int, default=128, help='batch size for tested networks')
parser.add_argument("-dm", "--debug_mode", action='store_true', help="debug mode, don't save results to disk")

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')

SERVER_IP = '132.72.80.67'

pop_hist = []  # keep track of every evaluated architecture
ex = Experiment()
ex.add_config(config_dict())


# dataset = ArticularyWordRecognition,AtrialFibrillation,BasicMotions,CharacterTrajectories,Cricket,DuckDuckGeese,EigenWorms,Epilepsy,ERing,EthanolConcentration,FaceDetection,FingerMovements,HandMovementDirection,Handwriting,Heartbeat,InsectWingbeat,JapaneseVowels,Libras,LSST,MotorImagery,NATOPS,PEMS-SF,PenDigits,PhonemeSpectra,RacketSports,SelfRegulationSCP1,SelfRegulationSCP2,SpokenArabicDigits,StandWalkJump,UWaveGestureLibrary
# EEG_dataset_1 = BCI_IV_2a,BCI_IV_2b,HG
# EEG dataset_2 = NER15,Opportunity,MentalImageryLongWords

def strfdelta(tdelta, fmt):
    d = {"days": tdelta.days}
    d["hours"], rem = divmod(tdelta.seconds, 3600)
    d["minutes"], d["seconds"] = divmod(rem, 60)
    return fmt.format(**d)


def connect_to_gdrive():
    gauth = GoogleAuth()
    # Try to load saved client credentials
    gauth.LoadCredentialsFile(f"{os.path.dirname(os.path.abspath(__file__))}/../mycreds.txt")
    if gauth.credentials is None:
        # Authenticate if they're not there
        gauth.CommandLineAuth()
    elif gauth.access_token_expired:
        # Refresh them if expired
        gauth.Refresh()
    else:
        # Initialize the saved creds
        gauth.Authorize()
    # Save the current credentials to a file
    gauth.SaveCredentialsFile(f"{os.path.dirname(os.path.abspath(__file__))}/../mycreds.txt")
    drive = GoogleDrive(gauth)
    return drive


def get_file_from_path(path):
    path_parts = path.split('/')
    drive = connect_to_gdrive()
    folder_id = 'root'
    for folder_name in path_parts[:-1]:
        folder_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
        folder = folder_list[[f['title'] for f in folder_list].index(folder_name)]
        folder_id = folder['id']
    file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
    file = file_list[[f['title'] for f in file_list].index(path_parts[-1])]
    file.GetContentFile(file['title'])
    return file.content


def save_file_to_path(path):
    path_parts = path.split('/')
    filename = path_parts[-1]
    drive = connect_to_gdrive()
    folder_id = 'root'
    for folder_name in path_parts[:-1]:
        folder_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
        folder = folder_list[[f['title'] for f in folder_list].index(folder_name)]
        folder_id = folder['id']
    file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
    file = file_list[[f['title'] for f in file_list].index(path_parts[-1])]
    file.SetContentFile(filename)
    file.Upload()


def upload_exp_results_to_gdrive(results_line, path):
    file = get_file_from_path(path)
    wb = load_workbook(filename=BytesIO(file.read()))
    wb.active = 0
    results = wb.active
    results.append(results_line)
    wb.save(filename=path.split('/')[-1])
    save_file_to_path(path)
    os.remove(path.split('/')[-1])


# ---------------------------------------------------------------------------------------------------------
# Define your NAS Problem
# ---------------------------------------------------------------------------------------------------------
class NAS(Problem):
    # first define the NAS problem (inherit from pymop)
    def __init__(self, search_space='micro', n_var=20, n_obj=1, n_constr=0, lb=None, ub=None,
                 init_channels=24, layers=8, epochs=25, save_dir=None, batch_size=128):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, type_var=np.int)
        self.xl = lb
        self.xu = ub
        self._search_space = search_space
        self._init_channels = init_channels
        self._layers = layers
        self._epochs = epochs
        self._save_dir = save_dir
        self._n_evaluated = 0  # keep track of how many architectures are sampled
        self.batch_size = batch_size

    def _evaluate(self, x, out, *args, **kwargs):

        objs = np.full((x.shape[0], self.n_obj), np.nan)

        for i in range(x.shape[0]):
            arch_id = self._n_evaluated + 1
            print('\n')
            logging.info('Network id = {}'.format(arch_id))

            # call back-propagation training
            if self._search_space == 'micro':
                micro_genome = micro_encoding.convert(x[i, :])
                macro_genome = None

            elif self._search_space == 'macro':
                macro_genome = macro_encoding.convert(x[i, :])
                micro_genome = None

            elif self._search_space == 'micromacro':
                macro_genome = macro_encoding.convert(x[i, :21])
                micro_genome = micro_encoding.convert(x[i, 21:])

            performance = train_search.main(macro_genome=macro_genome,
                                            micro_genome=micro_genome,
                                            search_space=self._search_space,
                                            init_channels=self._init_channels,
                                            layers=self._layers, cutout=False,
                                            epochs=self._epochs,
                                            save='arch_{}'.format(arch_id),
                                            expr_root=self._save_dir,
                                            batch_size=self.batch_size)

            # all objectives assume to be MINIMIZED !!!!!
            objs[i, 0] = 100 - performance['valid_acc']
            print(f'valid acc - {performance["valid_acc"]}')
            objs[i, 1] = performance['flops']
            ex.log_scalar(f"arch_valid_{config_dict()['performance_measure']}", performance['valid_acc'], arch_id)
            ex.log_scalar("arch_flops", performance['flops'], arch_id)
            self._n_evaluated += 1

        out["F"] = objs
        # if your NAS problem has constraints, use the following line to set constraints
        # out["G"] = np.column_stack([g1, g2, g3, g4, g5, g6]) in case 6 constraints


# ---------------------------------------------------------------------------------------------------------
# Define what statistics to print or save for each generation
# ---------------------------------------------------------------------------------------------------------
def do_every_generations(algorithm):
    # this function will be call every generation
    # it has access to the whole algorithm class
    gen = algorithm.n_gen
    pop_var = algorithm.pop.get("X")
    pop_obj = algorithm.pop.get("F")

    # report generation info to files
    logging.info("generation = {}".format(gen))
    logging.info("population error: best = {}, mean = {}, "
                 "median = {}, worst = {}".format(np.min(pop_obj[:, 0]), np.mean(pop_obj[:, 0]),
                                                  np.median(pop_obj[:, 0]), np.max(pop_obj[:, 0])))
    ex.log_scalar("best_error", np.min(pop_obj[:, 0]), gen)
    logging.info("population complexity: best = {}, mean = {}, "
                 "median = {}, worst = {}".format(np.min(pop_obj[:, 1]), np.mean(pop_obj[:, 1]),
                                                  np.median(pop_obj[:, 1]), np.max(pop_obj[:, 1])))
    ex.log_scalar("best_complexity", np.min(pop_obj[:, 1]), gen)


def set_micro_exp(args):
    n_var = int(4 * args.n_blocks * 2)
    lb = np.zeros(n_var)
    ub = np.ones(n_var)
    h = 1
    for b in range(0, n_var // 2, 4):
        ub[b] = args.n_ops - 1
        ub[b + 1] = h
        ub[b + 2] = args.n_ops - 1
        ub[b + 3] = h
        h += 1
    ub[n_var // 2:] = ub[:n_var // 2]
    return n_var, lb, ub


def set_macro_exp(args):
    n_var = int(((args.n_nodes - 1) * args.n_nodes / 2 + 1) * 3)
    lb = np.zeros(n_var)
    ub = np.ones(n_var)
    return n_var, lb, ub


@ex.main
def main():
    for exp_type in config_dict()['exp_order']:
        save_dir = f'{os.path.dirname(os.path.abspath(__file__))}/search-{args.save}-{exp_type}-{time.strftime("%Y%m%d-%H%M%S")}'
        utils.create_exp_dir(save_dir)
        fh = logging.FileHandler(os.path.join(save_dir, 'log.txt'))
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)

        np.random.seed(args.seed)
        logging.info("args = %s", args)

        # setup NAS search problem
        if exp_type == 'micro':  # NASNet search space
            n_var, lb, ub = set_micro_exp(args)
        elif exp_type == 'macro':  # modified GeneticCNN search space
            n_var, lb, ub = set_macro_exp(args)
        elif exp_type == 'micromacro':  # modified GeneticCNN search space
            n_var_mac, lb_mac, ub_mac = set_macro_exp(args)
            n_var_mic, lb_mic, ub_mic = set_micro_exp(args)
            n_var = n_var_mic + n_var_mac
            lb = np.array([*lb_mac, *lb_mic])
            ub = np.array([*ub_mac, *ub_mic])
        else:
            raise NameError('Unknown search space type')

        problem = NAS(n_var=n_var, search_space=exp_type,
                      n_obj=2, n_constr=0, lb=lb, ub=ub,
                      init_channels=args.init_channels, layers=args.layers,
                      epochs=args.epochs, save_dir=save_dir, batch_size=args.batch_size)

        # configure the nsga-net method
        method = engine.nsganet(pop_size=args.pop_size,
                                n_offsprings=args.n_offspring,
                                eliminate_duplicates=True)

        res = minimize(problem,
                       method,
                       callback=do_every_generations,
                       termination=('n_gen', args.n_gens))

        val_accs = res.pop.get('F')[:, 0]
        # if exp_type == 'micro':
        #     best_idx = np.where(val_accs == np.min(val_accs))[0][0]
        #     best_genome = res.pop[best_idx].X
        #     with open(f'{save_dir}/best_genome.pkl', 'wb') as pkl_file:
        #         pickle.dump(best_genome, pkl_file)
        #     set_config('micro_creator', make_micro_creator(best_genome))

    return (100 - np.min(val_accs)) / 100


def add_exp(all_exps, run, dataset, iteration, search_space):
    all_exps['algorithm'].append(f'NSGA_{search_space}')
    all_exps['architecture'].append('best')
    all_exps['measure'].append('accuracy')
    all_exps['dataset'].append(dataset)
    all_exps['iteration'].append(iteration)
    all_exps['result'].append(run.result)
    all_exps['runtime'].append(strfdelta(run.stop_time - run.start_time, '{hours}:{minutes}:{seconds}'))
    all_exps['omniboard_id'].append(run._id)
    return [lst[-1] for lst in all_exps.values()]


if __name__ == '__main__':
    first = True
    all_exps = defaultdict(list)
    args = parser.parse_args()
    if not args.debug_mode:
        ex.observers.append(MongoObserver.create(url=f'mongodb://{SERVER_IP}/EEGNAS', db_name='EEGNAS'))
    for iteration in range(1, args.iterations+1):
        for dataset in args.datasets.split(','):
            try:
                x_train = np.load(f'{os.path.dirname(os.path.abspath(__file__))}/../data/{dataset}/X_train.npy')
                x_test = np.load(f'{os.path.dirname(os.path.abspath(__file__))}/../data/{dataset}/X_test.npy')
                y_train = np.load(f'{os.path.dirname(os.path.abspath(__file__))}/../data/{dataset}/y_train.npy')
                y_test = np.load(f'{os.path.dirname(os.path.abspath(__file__))}/../data/{dataset}/y_test.npy')
                if 'netflow' in dataset:
                    set_config('problem', 'regression')
                    set_config('performance_measure', 'minus_mse')
                else:
                    set_config('problem', 'classification')
                    set_config('performance_measure', 'acc')
                set_config('dataset', dataset)
                set_config('x_train', x_train)
                set_config('x_test', x_test)
                set_config('y_train', y_train)
                set_config('y_test', y_test)
                set_config('INPUT_HEIGHT', x_train.shape[2])
                set_config('n_channels', x_train.shape[1])
                if 'exp_order' not in config_dict():
                    set_config('exp_order', [args.search_space])
                if y_train.ndim > 1:
                    set_config('n_classes', y_train.shape[1])
                else:
                    set_config('n_classes', len(np.unique(y_train)))
                set_config('micro_creator', ResidualNode)
                ex.add_config({'DEFAULT':{'dataset': dataset}})
                run = ex.run(options={'--name': f'NSGA_{dataset}_{args.search_space}'})
                exp_line = add_exp(all_exps, run, dataset, iteration, args.search_space)
                if not args.debug_mode:
                    upload_exp_results_to_gdrive(exp_line, 'University/Masters/Experiment Results/EEGNAS_results.xlsx')
                if first:
                    first_run_id = run._id
                    first = False
                pd.DataFrame(all_exps).to_csv(f'reports/{first_run_id}.csv', index=False)
            except Exception as e:
                print(f'failed dataset {dataset} iteration {iteration}')
                traceback.print_exc()