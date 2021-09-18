# DIA-project

## Setup

To run the code fist you have to install all the dependencies with:

    pip install -r requirements.txt

## Run the experiments

### How to run an experiment:
To run an experiment execute the file main.py and add "-e" followed by the experiment number you want to run: 

    python main.py -e 3        //will run experiment 3 
    python main.py -e all      //will run all the experiments 

    // available experiments are 3,4,5,6,7 and 'all'

to set manually the number of experiments to perform and the number of days add the following arguments:

    python main.py -e 3 -ne 10 -d 365

    -ne 10          //number of experiments to perform
    -d              //number of days for each experiment
    -del            //delay to apply, valid only for exp 5,6,7 
    -np             //binomial n param to generate people number, for exp 3,4

To debug the experiments is possible to show the log on terminal or to put the log inside a file:

    python main.py -e 4 -l 10 -lf

    -lf log.txt     //will save the log in a file called 'log.txt'
    -l 10           //will set the log granularity

the following levels of granularity are possible (default logging levels):
* 10 shows debug logging
* 20 shows info logging
* 30 shows warning logging
* 40 shows error logging
Look logging documentation for more details 
    
# Folders:
- experiments: experiments files
- images: 
    - environment: plots of the functions used for the environment
    - experiments: plots of the results for each experiment


