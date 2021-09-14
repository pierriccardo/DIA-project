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
from experiments.experiment_6_return_time import Experiment6_b
from experiments.experiment_7_Davide import Experiment7_b
from experiments.experiment_7_return_time import Experiment7_c
from experiments.experiment_5_new import Experiment5new
from experiments.experiment_6_new import Experiment6new

exp = None
if args.experiment == '3':
    exp = Experiment3()
elif args.experiment == '4':
    exp = Experiment4()
elif args.experiment == '5':
    exp = Experiment5()
elif args.experiment == '6':
    exp = Experiment6()
elif args.experiment == '7':
    exp = Experiment7()
elif args.experiment == '6b':
    exp = Experiment6_b()
elif args.experiment == '7b':
    exp = Experiment7_b()
elif args.experiment == '7c':
    exp = Experiment7_c()
elif args.experiment == '5new':
    exp = Experiment5new()
elif args.experiment == '6new':
    exp = Experiment6new()
else:
    print(f'Error Experiment {args.experiment} does not exist')

if exp is not None:
    exp.run()   
    exp.plot()
else:
    logging.error("Experiment selected doesn't exists")
