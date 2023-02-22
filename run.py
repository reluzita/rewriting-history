import argparse
from label_correction import get_label_correction_model, get_params
from train import fit_predict
from evaluation import evaluate
from format_data import get_data
from store_results import save_results
import random
import pandas as pd
import numpy as np
import warnings
import logging
import mlflow
from datetime import datetime

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    pd.set_option('mode.chained_assignment', None)
    warnings.filterwarnings("ignore")
    random.seed(0)
    np.random.seed(0)

    parser = argparse.ArgumentParser(description='Label correction testing.')
    parser.add_argument('dataset', type=str, help='OpenML dataset id', choices=['adult'])
    parser.add_argument('sensitive_attr', type=str, help='Sensitive attribute')
    parser.add_argument('correction_alg', type=str, help='Label noise correction algorithm', choices=['PL', 'STC', 'CC', 'HLNC'])
    parser.add_argument('--test_size', type=float, help='Test set size', required=False, default=0.2)
    parser.add_argument('--model', type=str, help='Classification algorithm to use', required=False, default='LogReg', choices=['LogReg'])
    parser.add_argument('--n_iterations', type=int, help='Number of iterations to run Cluster-based Correction', required=False, default=50)
    parser.add_argument('--n_clusters', type=int, help='Number of clusters to use in Cluster-based Correction', required=False, default=1000)

    args = parser.parse_args()

    mlflow.set_experiment(f'{args.dataset}_{args.correction_alg}')

    # get data
    X, y = get_data(args.dataset)

    # split data
    test_idx = random.sample(range(len(X)), int(len(X) * args.test_size))
    X_train = X.drop(test_idx)  
    y_train = y.drop(test_idx)
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    label_correction_params = get_params(args)
    label_correction_model = get_label_correction_model(args.correction_alg, label_correction_params)
    y_train_corrected = label_correction_model.correct(X_train, y_train)
    y_test_corrected = label_correction_model.correct(X_test, y_test)

    run_tag = f'{datetime.now().strftime("%d%m%Y_%H%M%S")}'
    for train_set in ['noisy', 'corrected']:
        for test_set in ['noisy', 'corrected']:
            with mlflow.start_run(tags={'train_set':train_set, 'test_set':test_set, 'run':run_tag}) as run:
                mlflow.log_param('dataset', args.dataset)
                mlflow.log_param('columns', list(X.columns))
                mlflow.log_param('test_size', args.test_size)
                
                label_correction_model.log_params()

                mlflow.log_param('classifier', args.model)

                if train_set == 'noisy':
                    y_pred = fit_predict(X_train, y_train, X_test, args.model)
                else:
                    y_pred = fit_predict(X_train, y_train_corrected, X_test, args.model)
                
                mlflow.log_param('senstive_attr', args.sensitive_attr)

                if test_set == 'noisy':
                    evaluate(y_test, y_pred, X_test[args.sensitive_attr])
                else:
                    evaluate(y_test_corrected, y_pred, X_test[args.sensitive_attr])

