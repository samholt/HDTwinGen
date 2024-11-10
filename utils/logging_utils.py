from torch import multiprocessing
import logging
from omegaconf import DictConfig
from enum import Enum

class Experiment(Enum):
    MAIN_TABLE = 1
    LESS_SAMPLES = 2
    OOD_INSIGHT = 3
    NSDT_ABLATION_NO_CRITIC = 4
    NSDT_ABLATION_NO_MEMORY = 5
    TRANSFORMER_HYP_OPT_PARAMS = 6


def generate_log_file_path(file, log_folder='logs', config = {}):
    import os, time, logging
    file_name = os.path.basename(os.path.realpath(file)).split('.py')[0]
    from pathlib import Path
    Path(f"./{log_folder}").mkdir(parents=True, exist_ok=True)
    path_run_name = '{}-{}'.format(file_name, time.strftime("%Y%m%d-%H%M%S"))
    experiment = Experiment[config.setup.experiment]
    return f"{log_folder}/{path_run_name}_{'-'.join(config.setup.methods_to_evaluate)}_{'-'.join(config.setup.envs_to_evaluate)}_{config.setup.seed_start}_{config.setup.seed_runs}-runs_log_{experiment.name}.txt"

def create_logger_in_process(log_file_path):
    logger = multiprocessing.get_logger()
    if not logger.hasHandlers():
        formatter = logging.Formatter("%(processName)s| %(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s")
        stream_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(log_file_path)
        stream_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
    return logger