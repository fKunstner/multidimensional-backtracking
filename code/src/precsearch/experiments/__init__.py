import multiprocessing
from typing import List

from tqdm import tqdm

from precsearch.experiments.datalogger import DataLogger, load_logs
from precsearch.experiments.experiment import Experiment


def runexp(exp):
    exp.run()


def run_all_experiments(experiments: List[Experiment], multiproc=True):
    if multiproc:
        pool = multiprocessing.Pool(10)
        pool.map(runexp, experiments)
    else:
        for exp in experiments:
            runexp(exp)


def run_all_new_experiments(experiments: List[Experiment], multiproc=True):
    run_all_experiments(
        [exp for exp in experiments if not exp.has_already_run()], multiproc=multiproc
    )
