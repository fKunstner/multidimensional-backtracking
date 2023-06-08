import os
import time
import warnings
from dataclasses import dataclass

import numpy as np

from precsearch import config
from precsearch.experiments import DataLogger, load_logs
from precsearch.experiments.rate_limited_logger import RateLimitedLogger
from precsearch.optimizers.optimizers import Optimizer, PrecSearch
from precsearch.optimizers.preconditioner_search import PreconditionerSet
from precsearch.problems import OptProblemAtPoint, Problem


@dataclass
class Experiment:
    prob: Problem
    opt: Optimizer

    def uname(self) -> str:
        return self.prob.uname() + "-" + self.opt.uname()

    def has_already_run(self):
        exp_filepath = config.exp_filepath(self.uname())
        return os.path.isfile(exp_filepath)

    def run(self):
        prob_at_x0 = self.prob.opt_problem()

        datalogger = DataLogger(self.uname())
        logger = RateLimitedLogger()

        t = 0
        start_time = time.time()

        logger.log(f"Starting experiment...", force=True)
        logger.log(f"Problem   = {self.prob}", force=True)
        logger.log(f"Optimizer = {self.opt}", force=True)
        logger.log(f"Will save in {config.exp_filepath(self.uname())}", force=True)

        def callback(prob_at_x: OptProblemAtPoint, state):
            nonlocal t

            t = t + 1
            datalogger.log(
                {
                    "f": prob_at_x.f,
                    "grad_norm": np.linalg.norm(prob_at_x.g),
                    "time": time.time() - start_time,
                }
            )

            if isinstance(self.opt, PrecSearch):
                if len(prob_at_x.x) < 20:
                    datalogger.log(
                        {"stepsizes": state["prec_set"].get_tentative_preconditioner()}
                    )
                datalogger.log({"logvolume": state["prec_set"].log_volume()})

            def item_is_small(v):
                if hasattr(v, "__len__") and len(v) > 100:
                    return False
                return True

            state_small_only = {k: v for k, v in state.items() if item_is_small(v)}
            datalogger.log(state_small_only)
            logger.log(
                f"Iter {t}. "
                f"Loss {prob_at_x.f}, "
                f"grad {np.linalg.norm(prob_at_x.g)}"
                f"state {state_small_only}"
            )
            datalogger.end_step()

        solve_func = self.opt.get_solve_func(self.prob)
        prob_at_end = solve_func(prob_at_x0, callback)

        datalogger.save()

        print(f"Experiment finished.")
        print(f"At start: {prob_at_x0.f, np.linalg.norm(prob_at_x0.g)}. ")
        print(f"At end: {prob_at_end.f, np.linalg.norm(prob_at_end.g)}. ")
        print(f"Last logged line:{datalogger._dicts[-1]}")

    def load(self):
        return load_logs(self.uname())
