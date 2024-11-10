import pandas as pd
import numpy as np
import torch
import random
from utils.results_utils import load_df, seed_all, generate_main_results, generate_main_results_less_samples, load_df_folder, generate_main_results_ood_table, load_df_file_evolutionary_plots, plot_evolutionary_plots, plot_intervention_COVID
from utils.logging_utils import Experiment
from time import time
import shelve
from enum import Enum
seed_all(0)

LOG_FOLDER = None
# Main Samples in paper
LOG_FOLDER = 'results/core_three_seeds/main_table/'

if LOG_FOLDER is not None:
    df = load_df_folder(LOG_FOLDER)
    if df.iloc[0]['experiment'] == Experiment.MAIN_TABLE.name or df.iloc[0]['experiment'] == Experiment.TRANSFORMER_HYP_OPT_PARAMS.name:
        _, table = generate_main_results(df)
        print('')
        print(table)
    elif df.iloc[0]['experiment'] == Experiment.LESS_SAMPLES.name:
        _, table = generate_main_results_less_samples(df)
        print('')
        print(table)
    elif df.iloc[0]['experiment'] == Experiment.OOD_INSIGHT.name:
        _, table = generate_main_results_ood_table(df)
        print('')
        print(table)
    elif df.iloc[0]['experiment'] == Experiment.NSDT_ABLATION_NO_CRITIC.name or df.iloc[0]['experiment'] == Experiment.NSDT_ABLATION_NO_MEMORY.name:
        _, table = generate_main_results(df)
        print('')
        print(table)
else:
    FILE_PATH = "results/rebuttal_AP_optimization/subsection.txt"
    df, df_pops, dfm_r = load_df_file_evolutionary_plots(FILE_PATH)
    plot_evolutionary_plots(df_pops)