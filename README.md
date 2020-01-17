
# NLP Author Case

Find author based on small text sentences

## Setup
Optionnal if using GIT:

1. Install git and checkout the [git code repository]

Optionnal if python project:

2. Install [anaconda] python version 3.6+
3. Change working directory the project root directory
4. Create the self contained conda environment. Open anaconda prompt and go to the project root directory and enter the command:

    `conda env create --file conda_env.yml`

5. Any python modules under src need to be available to other scripts. This can be done in a couple of ways. The recommended way
   is to copy paste the following code at the top of your script or notebook

        import os 
        import sys

        if "__file__" in globals():
            path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        else:
            path = os.path.dirname(globals()['_dh'][0])
        sys.path.insert(0, path)
    Note that if your script is in a subfolder, this code will need to be adapted


6. .. Place your own project specific setup steps here e.g. copying data files ...


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

## Initial File Structure

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

## Testing
Reproducability and the correct functioning of code are essential to avoid wasted time. If a code block is copied more 
than once then it should be placed into a common script / module under src and unit tests added. The same applies for 
any other non trivial code to ensure the correct functioning.

To run tests, install pytest using pip or conda (should have been setup already if you used the conda_env.yml file) and 
then from the repository root run
 
```
pytest
```

## Development Process
Contributions to this project are greatly appreciated and encouraged.

To contribute an update simply:

* Create a new branch / fork for your updates.
* Check that your code follows the PEP8 guidelines (line lengths up to 120 are ok) and other general conventions within this document.
* Ensure that as far as possible there are unit tests covering the functionality of any new code.
* Update the yml file for the python environment if new packages are installed
* Check that all existing unit tests still pass.
* Edit this document if needed to describe new files or other important information.
* Create a pull request.

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
