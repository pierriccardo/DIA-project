# DIA-project

## Setup

To run the code fist you have to install all the dependencies with:

    pip install -r requirements.txt

## Documentation

### How to run an experiment:
To run an experiment execute the file main.py and add "-e" followed byt the experiment number you want to run: 

    python main.py -e 3 

To debug the experiments is possible to show the log on terminal or to put the log inside a file:

    python main.py -e 4 -l 10 -lf

where:

    -lf

Needs to be specified to save the log in a file 'log.txt'

    -l 10 

indicates the log granularity, the following levels of granularity are possible (default logging levels):
* 10 shows debug logging
* 20 shows info logging
* 30 shows warning logging
* 40 shows error logging
Look logging documentation for more details 
    
# Folders:
- img: images generated
- others: resources, tutorial

# git tutorial

per controllare le modifiche nei file 

    git status

per fare una commit

    # aggiungere tutte le modifiche fatte alla commit
    git add --all   

    # fare la commit, -m serve a specificare il messaggio
    git commit -m "messaggio di commit" 

    # inviare la commit
    git push

per fare un branch:

    # git branch "nome del branch"

    # git checkout "nome del branch"


