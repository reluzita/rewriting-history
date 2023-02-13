import argparse
from label_correction import apply_label_correction, get_args
from train import fit_predict
from evaluation import evaluate
from format_data import get_data
from store_results import save_results
import random
import pandas as pd

pd.set_option('mode.chained_assignment', None)
random.seed(0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Label correction testing.')
    parser.add_argument('dataset', type=str, help='OpenML dataset id', choices=['adult'])
    parser.add_argument('sensitive_attr', type=str, help='Sensitive attribute')
    parser.add_argument('correction_alg', type=str, help='Label noise correction algorithm', choices=['PL', 'STC'])
    parser.add_argument('--test_size', type=float, help='Test set size', required=False, default=0.2)
    parser.add_argument('--model', type=str, help='Classification algorithm to use', required=False, default='LogReg', choices=['LogReg'])

    args = parser.parse_args()

    # get data
    X, y = get_data(args.dataset)

    # apply label correction
    label_correction_args = get_args(args.correction_alg)
    y_corrected = apply_label_correction(X, y, args.correction_alg, label_correction_args)

    # split data
    test_idx = random.sample(range(len(X)), int(len(X) * args.test_size))

    X_train = X.drop(test_idx)  
    y_train = y.drop(test_idx)
    y_train_corrected = y_corrected.drop(test_idx)

    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]
    y_test_corrected = y_corrected.iloc[test_idx]

    # get predictions

    y_pred_noisy = fit_predict(X_train, y_train, X_test, args.model)
    y_pred_corrected = fit_predict(X_train, y_train_corrected, X_test, args.model)

    # evaluate

    results = evaluate(y_test, y_test_corrected, y_pred_noisy, y_pred_corrected, X_test[args.sensitive_attr])
    save_results(args, label_correction_args, results)

    