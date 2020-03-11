
# NLP Author Case

Find author based on small text sentences

## Setup

1. Install git and checkout the https://bitbucket.org/jerome_belp/nlp_author_case/src/master/
2. Install [anaconda] python version 3.6+
3. Change working directory the project root directory
4. Create the self contained conda environment. Open anaconda prompt and go to the project root directory and enter the command:

    `conda env create --file conda_env.yml`

## Using the Python Conda environment

Once the Python Conda environment has been set up, you can

* Activate the environment using the following command in a terminal window:

    * Windows: `activate my_environment`
    * Linux, OS X: `source activate my_environment`
    * The __environment is activated per terminal session__, so you must activate it every time you open terminal.

* Deactivate the environment using the following command in a terminal window:

    * Windows: `deactivate my_environment`
    * Linux, OS X: `source deactivate my_environment`
               
* Delete the environment using the command (can't be undone):

    * `conda remove --name my_environment --all`


## Guide with instructions
For more information on the project please follow the Jupyter Notebook guide in `./scripts/NLP_author_case_guide.ipynb`

## File Structure

```
├── .gitignore               <- Files that should be ignored by git. Add seperate .gitignore files in sub folders if 
│                               needed
├── conda_env.yml            <- Conda environment definition for ensuring consistent setup across environments
├── LICENSE
├── README.md                <- The top-level README for developers using this project.
├── setup.py                 <- Metadata about your project for easy distribution.
│
├── config                   <- Config files 
│
├── data
│   ├── 0_raw                <- Raw data files
│   ├── 1_interim            <- Interim files
│   ├── 2_processed          <- The final, canonical data sets for modeling
│   └── temp                 <- Temporary file
│
├── docs                     <- Documentation
│
├── extras                   <- Miscellaneous extras
│
├── models                   <- Trained models
│
├── presentations            <- Powerpoint presentations
│
├── reports                  <- Visual exports (Tableau, Qlik, PowerBI, Images etc.)
│
├── scripts                  <- All scripts
│
├── src                      <- Code for use in this project.
│    └──__init__.py      <- Python package initialisation
│
└── tests                    <- Test cases (named after module)
```

## Contacts
* Author: Jerome Belpaire
* Agilytic team: Jerome Belpaire
* Client: Agilytic

## References
* https://jerome_belp@bitbucket.org/jerome_belp/cookiecutter-agilytic.git - The master template for this project
* http://docs.python-guide.org/en/latest/writing/structure/
* https://drivendata.github.io/cookiecutter-data-science/

[//]: #
   [anaconda]: <https://www.continuum.io/downloads>
