from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

ARGS = {
    'PL': {
        'name': 'Polishing Labels',
        'folds': 10,
        'classifier': 'LogReg'
    },
    'STC': {
        'name': 'Self-Training Correction',
        'folds': 10,
        'classifier': 'LogReg',
        'correction_rate': 0.8
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
    elif algorithm == 'STC':
        return self_training_correction(X, y, CLASSIFIERS[args['classifier']], args['folds'], args['correction_rate'])

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
    y_corrected : pandas.Series
        Series containing the corrected labels
    """

    kf = KFold(n_splits=n_folds, random_state=42, shuffle=True)
    clf_list = []

    for train_index, _ in kf.split(X):
        X_train = X.loc[train_index]
        y_train = y.loc[train_index]

        clf_list.append(classifier(random_state=42).fit(X_train.values, y_train.values))

    y_corrected = X.apply(
        lambda x: 0 if np.mean([clf_list[i].predict([x.values])[0] for i in range(n_folds)]) < 0.5 else 1, 
        axis=1)

    return y_corrected

def self_training_correction(X:pd.DataFrame, y:pd.Series, classifier=LogisticRegression, n_folds=10, correction_rate=0.8):
    """
    Self-training correction algorithm

    Parameters
    ----------
    X : pandas.DataFrame
        Dataframe containing the features
    y : pandas.Series
        Series containing the labels
    classifier : sklearn classifier, optional, default=LogisticRegression
        Classifier to use for the STC algorithm
    n_folds : int, optional, default=10
        Number of folds to use for the STC algorithm
    correction_rate : float, optional, default=0.8
        Correction rate to use for the STC algorithm

    Returns
    -------
    y_corrected : pandas.Series
        Series containing the corrected labels
    """
    kf = KFold(n_splits=n_folds, random_state=42, shuffle=True)
    noisy = set()

    # Split the current training data set using an n-fold cross-validation scheme
    for train_index, test_index in kf.split(X):
        X_train = X.loc[train_index]
        X_test = X.loc[test_index]
        y_train = y.loc[train_index]
        y_test = y.loc[test_index]

        # For each of these n parts, a learning algorithm is trained on the other n-1 parts, resulting in n different classifiers
        model = classifier(random_state=42).fit(X_train, y_train)

        # These n classifiers are used to tag each instance in the excluded part as either correct or mislabeled, by comparing the training label with that assigned by the classifier.
        y_pred = pd.Series(model.predict(X_test), index=test_index)
        
        # The misclassified examples from the previous step are added to the noisy data set.
        for i, value in y_pred.items():
            if value != y_test.loc[i]:
                noisy.add(i)

    noisy = list(noisy)

    X_clean = X.drop(noisy)
    y_clean = y.drop(noisy)
    X_noisy = X.loc[noisy]

    # Build a model from the clean set and uses that to calculate the confidence that each of the instances from the noisy set is mislabeled
    model = classifier(random_state=42).fit(X_clean, y_clean)
    y_pred = pd.Series(model.predict(X_noisy), index=noisy)
    y_prob = pd.DataFrame(model.predict_proba(X_noisy), index=noisy, columns=[0, 1])

    corrected = pd.DataFrame(columns=['y_pred', 'y_prob'])
    for i in list(noisy):
        if y_pred.loc[i] != y.loc[i]:
            corrected.loc[i] = [y_pred.loc[i], y_prob.loc[i, y_pred.loc[i]]]

    # The noisy instance with the highest calculated likelihood of belonging to some class that is not equal to its current class 
    # is relabeled to the class that the classifier determined is the instance’s most likely true class. 
    correction_n = int(correction_rate*len(corrected))
    y_corrected = y.copy()
    for i in corrected.sort_values('y_prob', ascending=False)[:correction_n].index:
        y_corrected.loc[i] = corrected.loc[i, 'y_pred']

    return y_corrected
