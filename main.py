from os import name
from numpy.random import seed
import yaml
import numpy as np
from utils import *
from configmanager import ConfigManager 
from pricing import PersonGenerator
import argparse
import logging

logging.getLogger(__name__)

SEED = 1
np.random.seed(seed=SEED)
cm = ConfigManager()
#------------------------------
# ARGUMENTS PARSER
#------------------------------
# to be used wen running the file main.py

# --experiment or -e used to choose the experiment
# e.g. python main.py -e 4 (will run and plot exp 4)

# --logfile or -lf if specified put the log in a file
# e.g. python main.py -e 4 -lf

# --log or -l used to choose the logging level
# levels used are 
# DEBUG   = 10
# INFO    = 20
# WARNING = 30
# ERROR   = 40
# e.g. python main.py -e 4 -l 10 (to activate DEBUG)

parser = argparse.ArgumentParser(description='Expriments launcher')
parser.add_argument('--experiment', '-e',  type=str,  default=3, help='experiment number')
parser.add_argument('--logfile', '-lf',  type=str, help='wheter to output in a logfile or not')
parser.add_argument('--log', '-l',  type=int,  default=40, help='wheter to output in a logfile or not')

parser.add_argument('--days', '-d',  type=int,  default=365, help='number of days for the experiment')
parser.add_argument('--n_exp', '-ne',  type=int,  default=10, help='number of experiments to perform')
parser.add_argument('--delay', '-del',  type=int,  default=30, help='delay to apply on the experiment, just for 5,6,7')
args = parser.parse_args()

#------------------------------
# LOGGER
#------------------------------
filename = None
if args.logfile is not None:
    filename = args.logfile
    logging.basicConfig(level=args.log, filename=filename)
else:
    logging.basicConfig(level=args.log)
logging.debug('Started')


#------------------------------
# EXPERIMENTS
#------------------------------
from experiments.experiment_3 import Experiment3
from experiments.experiment_4 import Experiment4
from experiments.experiment_5 import Experiment5
from experiments.experiment_6 import Experiment6
from experiments.experiment_7 import Experiment7

exp = []
if args.experiment == '3':
    exp.append(Experiment3(days=args.days, n_exp=args.n_exp))
elif args.experiment == '4':
    exp.append(Experiment4(days=args.days, n_exp=args.n_exp))
elif args.experiment == '5':
    exp.append(Experiment5(days=args.days, n_exp=args.n_exp, delay=args.delay))
elif args.experiment == '6':
    exp.append(Experiment6(days=args.days, n_exp=args.n_exp, delay=args.delay))
elif args.experiment == '7':
    exp.append(Experiment7(days=args.days, n_exp=args.n_exp, delay=args.delay))

elif args.experiment == 'all':
    exp.append(Experiment3(cm.exp_values['exp3']['days'], cm.exp_values['exp3']['n_exp']))
    exp.append(Experiment4(cm.exp_values['exp4']['days'], cm.exp_values['exp4']['n_exp']))
    exp.append(Experiment5(cm.exp_values['exp5']['days'], cm.exp_values['exp5']['n_exp'], cm.exp_values['exp5']['delay']))
    exp.append(Experiment6(cm.exp_values['exp6']['days'], cm.exp_values['exp6']['n_exp'], cm.exp_values['exp6']['delay']))
    exp.append(Experiment7(cm.exp_values['exp7']['days'], cm.exp_values['exp7']['n_exp'], cm.exp_values['exp7']['delay']))
else:
    print(f'Error Experiment {args.experiment} does not exist')

import traceback
if len(exp) > 0:
    for e in exp:
        try:
            print(f'[Running {e.NAME} | num_exp: {e.n_experiments} | days: {e.T}]')
            e.run()   
            e.plot()
        except:
            print('Error running Experiment {e.NAME}')
            traceback.print_exc()
else:
    logging.error("Experiment selected doesn't exists")
