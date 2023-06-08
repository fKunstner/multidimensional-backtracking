# Multidimensional backtracking codebase


We recommend installing the library in a fresh environment.

### Installation 

To install dependencies and the experiment library, 
clone this repository and from this folder,

```
# Install dependencies 
cd dependencies/dataset-downloader 
pip install -r requirements.txt
pip install . 

# Go back to root folder to install library
cd ../..
pip install -r requirements.txt
pip install . 
```

### Configuration
Where to store the datasets and the results of the experiments
is specified through an environment variable.

On Windows, use
```
set PRECSEARCH_WORKSPACE=C:\path\to\folder
```
On Unix, use
```
export PRECSEARCH_WORKSPACE=~/path/to/folder
```

### Running optimizers

The experiment scripts are contained in `scripts`. 
To run all experiments, run `bash run_all.sh`, 
which also calls the plotting scripts. 
The files `exp_[size]_[task].py` define the experiments to run.


Our optimizer is implemented using a scipy-style interface 
in `src/precsearch/optimizers/preconditioner_search.py`. 
The file contains the definition of solve_precsearch