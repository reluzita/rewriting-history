import os
import pandas as pd
import json
from datetime import datetime

def save_results(args, label_correction_args, results:pd.DataFrame):
    """
    Save the results and parameters of the experiment

    Parameters
    ----------
    args : argparse.Namespace
        Parameters for the experiment
    label_correction_args : dict
        Parameters for the label correction algorithm
    results : pandas.DataFrame
        Dataframe containing the performance metrics for the different combinations of training and test sets
        
    """
    results_dir = f'experiments/{args.dataset}_{args.correction_alg}_{datetime.now().strftime("%d%m%Y%H%M%S")}'
    if not os.path.exists(results_dir):
        if not os.path.exists('experiments'):
            os.mkdir('experiments')
        os.mkdir(results_dir)
    results.to_csv(f'{results_dir}/results.csv')
    json.dump({
        'Dataset': args.dataset,
        'Sensitive Attribute': args.sensitive_attr,
        'Label Correction Algorithm': label_correction_args,
        'Test Set Size': args.test_size,
        'Classifier': args.model
    }, open(f'{results_dir}/args.json', 'w'))