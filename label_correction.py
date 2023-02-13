from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

ARGS = {
    'PL': {
        'name': 'Polishing Labels',
        'folds': 10,
        'classifier': 'LogReg'
    }
}

CLASSIFIERS = {
    'LogReg': LogisticRegression
}

def get_args(algorithm):
    """
    Get arguments for the label correction algorithm

    Parameters
    ----------
    algorithm : str
        Label correction algorithm to use

    Returns
    -------
    args : dict
        Dictionary containing the arguments for the label correction algorithm
    """
    return ARGS[algorithm]

def apply_label_correction(X, y, algorithm, args):
    """
    Apply label correction algorithm

    Parameters
    ----------
    X : pandas.DataFrame
        Dataframe containing the features
    y : pandas.Series
        Series containing the labels
    algorithm : str
        Label correction algorithm to use
    args : dict
        Dictionary containing the arguments for the label correction algorithm

    Returns
    -------
    corrected_labels : pandas.Series
        Series containing the corrected labels
    """
    if algorithm == 'PL':
        return polishing_labels(X, y, CLASSIFIERS[args['classifier']], args['folds'])

def polishing_labels(X:pd.DataFrame, y:pd.Series, classifier=LogisticRegression, n_folds=10):
    """
    Polishing labels algorithm

    Parameters
    ----------
    X : pandas.DataFrame
        Dataframe containing the features
    y : pandas.Series
        Series containing the labels
    classifier : sklearn classifier, optional, default=LogisticRegression
        Classifier to use for the k-fold cross validation
    n_folds : int, optional, default=10
        Number of folds to use for the k-fold cross validation

    Returns
    -------
    corrected_labels : pandas.Series
        Series containing the corrected labels
    """

    kf = KFold(n_splits=n_folds, random_state=42, shuffle=True)
    clf_list = []

    for train_index, _ in kf.split(X):
        X_train = X.loc[train_index]
        y_train = y.loc[train_index]

        clf_list.append(classifier(random_state=42).fit(X_train.values, y_train.values))

    corrected_labels = X.apply(
        lambda x: 0 if np.mean([clf_list[i].predict([x.values])[0] for i in range(n_folds)]) < 0.5 else 1, 
        axis=1)

    return corrected_labels