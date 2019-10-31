import pdb
import sys
# update your projecty root path before running
import traceback
from collections import defaultdict

sys.path.insert(0, '/home/eladr/nsga-net_remote')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
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

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')

ITERATIONS = 1
SERVER_IP = '132.72.80.67'

pop_hist = []  # keep track of every evaluated architecture
ex = Experiment(f'NSGA-net_{config_dict()["nsga_strategy"]}_{config_dict()["dataset"]}_{config_dict()["data_type"]}')
ex.add_config(config_dict())


# dataset = ArticularyWordRecognition,AtrialFibrillation,BasicMotions,CharacterTrajectories,Cricket,DuckDuckGeese,EigenWorms,Epilepsy,ERing,EthanolConcentration,FaceDetection,FingerMovements,HandMovementDirection,Handwriting,Heartbeat,InsectWingbeat,JapaneseVowels,Libras,LSST,MotorImagery,NATOPS,PEMS-SF,PenDigits,PhonemeSpectra,RacketSports,SelfRegulationSCP1,SelfRegulationSCP2,SpokenArabicDigits,StandWalkJump,UWaveGestureLibrary
# EEG_dataset_1 = BCI_IV_2a,BCI_IV_2b,HG
# EEG dataset_2 = NER15,Opportunity,MentalImageryLongWords

def strfdelta(tdelta, fmt):
    d = {"days": tdelta.days}
    d["hours"], rem = divmod(tdelta.seconds, 3600)
    d["minutes"], d["seconds"] = divmod(rem, 60)
    return fmt.format(**d)

# ---------------------------------------------------------------------------------------------------------
# Define your NAS Problem
# ---------------------------------------------------------------------------------------------------------
class NAS(Problem):
    # first define the NAS problem (inherit from pymop)
    def __init__(self, search_space='micro', n_var=20, n_obj=1, n_constr=0, lb=None, ub=None,
                 init_channels=24, layers=8, epochs=25, save_dir=None):
        super().__init__(n_var=n_var, n_obj=n_obj, n_constr=n_constr, type_var=np.int)
        self.xl = lb
        self.xu = ub
        self._search_space = search_space
        self._init_channels = init_channels
        self._layers = layers
        self._epochs = epochs
        self._save_dir = save_dir
        self._n_evaluated = 0  # keep track of how many architectures are sampled

    def _evaluate(self, x, out, *args, **kwargs):

        objs = np.full((x.shape[0], self.n_obj), np.nan)

        for i in range(x.shape[0]):
            arch_id = self._n_evaluated + 1
            print('\n')
            logging.info('Network id = {}'.format(arch_id))

            # call back-propagation training
            if self._search_space == 'micro':
                genome = micro_encoding.convert(x[i, :])
            elif self._search_space == 'macro':
                genome = macro_encoding.convert(x[i, :])
            performance = train_search.main(genome=genome,
                                            search_space=self._search_space,
                                            init_channels=self._init_channels,
                                            layers=self._layers, cutout=False,
                                            epochs=self._epochs,
                                            save='arch_{}'.format(arch_id),
                                            expr_root=self._save_dir)

            # all objectives assume to be MINIMIZED !!!!!
            objs[i, 0] = 100 - performance['valid_acc']
            print(f'valid acc - {performance["valid_acc"]}')
            objs[i, 1] = performance['flops']
            ex.log_scalar("arch_valid_acc", performance['valid_acc'], arch_id)
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


@ex.main
def main():
    args = parser.parse_args(config_dict()['arg_string'].split())
    args.search_space = sys.argv[1]
    args.save = 'search-{}-{}-{}'.format(args.save, args.search_space, time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(args.save)
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    np.random.seed(args.seed)
    logging.info("args = %s", args)

    # setup NAS search problem
    if args.search_space == 'micro':  # NASNet search space
        n_var = int(4 * args.n_blocks * 2)
        lb = np.zeros(n_var)
        ub = np.ones(n_var)
        h = 1
        for b in range(0, n_var//2, 4):
            ub[b] = args.n_ops - 1
            ub[b + 1] = h
            ub[b + 2] = args.n_ops - 1
            ub[b + 3] = h
            h += 1
        ub[n_var//2:] = ub[:n_var//2]
    elif args.search_space == 'macro':  # modified GeneticCNN search space
        n_var = int(((args.n_nodes-1)*args.n_nodes/2 + 1)*3)
        lb = np.zeros(n_var)
        ub = np.ones(n_var)
    else:
        raise NameError('Unknown search space type')

    problem = NAS(n_var=n_var, search_space=args.search_space,
                  n_obj=2, n_constr=0, lb=lb, ub=ub,
                  init_channels=args.init_channels, layers=args.layers,
                  epochs=args.epochs, save_dir=args.save)

    # configure the nsga-net method
    method = engine.nsganet(pop_size=args.pop_size,
                            n_offsprings=args.n_offspring,
                            eliminate_duplicates=True)

    res = minimize(problem,
                   method,
                   callback=do_every_generations,
                   termination=('n_gen', args.n_gens))

    return (100 - np.min(res.pop.get('F')[:, 0])) / 100


def add_exp(all_exps, run, dataset, iteration):
    all_exps['algorithm'].append(f'NSGA_{sys.argv[1]}')
    all_exps['architecture'].append('best')
    all_exps['measure'].append('accuracy')
    all_exps['dataset'].append(dataset)
    all_exps['iteration'].append(iteration)
    all_exps['result'].append(run.result)
    all_exps['runtime'].append(strfdelta(run.stop_time - run.start_time, '{hours}:{minutes}:{seconds}'))
    all_exps['omniboard_id'].append(run._id)


if __name__ == '__main__':
    first = True
    all_exps = defaultdict(list)
    if 'debug' not in sys.argv:
        ex.observers.append(MongoObserver.create(url=f'mongodb://{SERVER_IP}/EEGNAS', db_name='EEGNAS'))
    for iteration in range(1, ITERATIONS+2):
        for dataset in sys.argv[2].split(','):
            try:
                x_train = np.load(f'{os.path.dirname(os.path.abspath(__file__))}/../data/{dataset}/X_train.npy')
                x_test = np.load(f'{os.path.dirname(os.path.abspath(__file__))}/../data/{dataset}/X_test.npy')
                y_train = np.load(f'{os.path.dirname(os.path.abspath(__file__))}/../data/{dataset}/y_train.npy')
                y_test = np.load(f'{os.path.dirname(os.path.abspath(__file__))}/../data/{dataset}/y_test.npy')
                set_config('dataset', dataset)
                set_config('x_train', x_train)
                set_config('x_test', x_test)
                set_config('y_train', y_train)
                set_config('y_test', y_test)
                set_config('INPUT_HEIGHT', x_train.shape[2])
                set_config('n_channels', x_train.shape[1])
                set_config('n_classes', len(np.unique(y_train)))
                ex.add_config({'DEFAULT':{'dataset': dataset}})
                run = ex.run(options={'--name': f'NSGA_{dataset}_{sys.argv[1]}'})
                add_exp(all_exps, run, dataset, iteration)
                if first:
                    first_run_id = run._id
                    first = False
                pd.DataFrame(all_exps).to_csv(f'reports/{first_run_id}.csv', index=False)
            except Exception as e:
                print(f'failed dataset {dataset} iteration {iteration}')
                traceback.print_exc()