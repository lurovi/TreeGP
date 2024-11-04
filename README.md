# Tree GP

Vanilla tree-based Genetic Programming (GP) implementation in Python 3.11 with the adoption of genepro 1.3.2 (https://github.com/giorgia-nadizar/genepro).
Installation:

```bash
git clone https://github.com/lurovi/TreeGP.git
cd TreeGP
conda env create -f environment.yml
conda activate treegp_env
cd ..
git clone -b v1.3.2 https://github.com/giorgia-nadizar/genepro.git
cd genepro
pip3 install -U .
cd ..
cd TreeGP
pip3 install -U .
```

Be careful to properly set PYTHONPATH environment variable. Plus, mind that code leverages a custom environment variable called CURRENT\_CODEBASE\_FOLDER that contains the absolute path of the folder (the path ends with '/') that contains all code and data relevant for the project. You just have to set this variable properly to make it easier to access locations for code and data. Change the paths in the experiment scripts accordingly, always be careful that for paths pointing to a folder you must always put a '/' at the end.

For the datasets, the scripts for processing them can be found in the benchmark package. The experiments expect datasets in .csv format named trainSEED.csv and testSEED.csv where SEED is the integer value of the seed. Each dataset has d + 1 columns where d is the number of variables/features. The column names for the variables range from 0 to d-1. The last column is called target (the y values). Column names are available in the header of the .csv file (very first row of the file). Besides the header, each row is a dataset observation/record. The .csv separator is the comma. All train-test splits belonging to the same dataset go into the same directory called with the name of the dataset (e.g., vladislavleva14/, airfoil/).

The main scripts for experiments can be found in the exps package.
