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
parser = argparse.ArgumentParser(description='Expriments launcher')
parser.add_argument('--experiment', '-e',  type=int,  default=3, help='experiment number')
parser.add_argument('--logfile', '-lf',  type=bool,  default=False, help='wheter to output in a logfile or not')
args = parser.parse_args()

#------------------------------
# LOGGER
#------------------------------
filename = None
if args.logfile:
    filename =  "./log.txt"
logging.basicConfig(level=logging.INFO, filename=filename)
logging.info('Started')


#------------------------------
# EXPERIMENTS
#------------------------------
#from experiments import experiment_3 as e3
from experiments import experiment_4 as e4

exp = None

if args.experiment == 4:
    exp = e4.Experiment4()

if exp is not None:
    exp.run()
    exp.plot()
else:
    logging.error("Experiment selected doesn't exists")
