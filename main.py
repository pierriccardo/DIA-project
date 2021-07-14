from os import name
import yaml
import numpy as np
from utils import *
from configmanager import ConfigManager 
from pricing import PersonGenerator
from context import *
import argparse
import logging

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
# e.g. python main.py -e 4 -l 10 (to activate DEBUG level)

parser = argparse.ArgumentParser(description='Expriments launcher')
parser.add_argument('--experiment', '-e',  type=str,  default=3, help='experiment number')
#sparser.add_argument('--logfile', '-lf',  type=bool,  action=argparse.BooleanOptionalAction, default=False, help='wheter to output in a logfile or not')
parser.add_argument('--log', '-l',  type=int,  default=40, help='wheter to output in a logfile or not')
args = parser.parse_args()

#------------------------------
# LOGGER
#------------------------------
filename = None
if args.logfile:
    filename =  "./log.txt"
logging.basicConfig(level=args.log, filename=filename)
logging.debug('Started')


#------------------------------
# EXPERIMENTS
#------------------------------
from experiments.experiment_3 import Experiment3
from experiments.experiment_4 import Experiment4
from experiments.experiment_5 import Experiment5
from experiments.experiment_6 import Experiment6
from experiments.experiment_7 import Experiment7

e3 = Experiment3()
e4 = Experiment4()
e5 = Experiment5()
e6 = Experiment6()
e7 = Experiment7()

all_exp = [e3, e4, e5, e6, e7]

exps = {}
exps["3"] = e3
exps["4"] = e4
exps["5"] = e5
exps["6"] = e6
exps["7"] = e7
exps["all"] = all_exp

try:
    exp = exps[args.experiment]
    exp.run()   
    exp.plot()

except Exception:
    logging.error("Experiment selected doesn't exists")
