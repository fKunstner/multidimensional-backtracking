import os
from pathlib import Path


def base_workspace():
    workspace = os.environ.get("PRECSEARCH_WORKSPACE", None)
    if workspace is None:
        raise ValueError(
            "Workspace not set. "
            "Define the PRECSEARCH_WORKSPACE env. variable to store data"
        )
    return workspace


def experiment_dir() -> Path:
    return Path(os.path.join(base_workspace(), "exps"))


def problems_dir() -> Path:
    return Path(os.path.join(base_workspace(), "problems"))


def problem_info_filepath(problem) -> Path:
    prob_folder = Path(os.path.join(problems_dir(), problem.uname()))
    prob_folder.mkdir(parents=True, exist_ok=True)
    return Path(os.path.join(prob_folder, "prob.csv"))


def exp_filepath(exp_id: str) -> Path:
    exp_folder = Path(os.path.join(experiment_dir(), exp_id))
    exp_folder.mkdir(parents=True, exist_ok=True)
    return Path(os.path.join(exp_folder, "exp.csv"))


def get_console_logging_level():
    return "INFO"
