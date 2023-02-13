import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

def fpr(y_true, y_pred, sensitive_attr, group):
    """
    False positive rate for a given group

    Parameters
    ----------
    y_true : numpy.ndarray
        Array containing the true labels
    y_pred : numpy.ndarray
        Array containing the predicted labels
    sensitive_attr : numpy.ndarray
        Array containing the sensitive attribute
    group : int
        Value of the sensitive attribute to consider

    Returns
    -------
    fpr : float
        False positive rate for the given group
    """

    return np.sum((y_true == 0) & (y_pred == 1) & (sensitive_attr == group))/np.sum((y_true == 0) & (sensitive_attr == group))

def fnr(y_true, y_pred, sensitive_attr, group):
    """
    False negative rate for a given group

    Parameters
    ----------
    y_true : numpy.ndarray
        Array containing the true labels
    y_pred : numpy.ndarray
        Array containing the predicted labels
    sensitive_attr : numpy.ndarray
        Array containing the sensitive attribute
    group : int
        Value of the sensitive attribute to consider

    Returns
    -------
    fnr : float
        False negative rate for the given group
    """
    return np.sum((y_true == 1) & (y_pred == 0) & (sensitive_attr == group))/np.sum((y_true == 1) & (sensitive_attr == group))

def predictive_equality(y_test, y_pred, sensitive_attr):
    """
    Predicted equality - difference between false positive rates for protected and unprotected groups

    Parameters
    ----------
    y_test : numpy.ndarray
        Array containing the true labels
    y_pred : numpy.ndarray
        Array containing the predicted labels
    sensitive_attr : numpy.ndarray
        Array containing the sensitive attribute
    
    Returns
    -------
    predictive_equality : float
        Difference between false positive rates for protected and unprotected groups
    """

    fpr_0 = fpr(y_test, y_pred, sensitive_attr, 0)
    fpr_1 = fpr(y_test, y_pred, sensitive_attr, 1)

    return np.abs(fpr_0 - fpr_1)

def equal_opportunity(y_test, y_pred, sensitive_attr):
    """
    Equal opportunity - difference between false negative rates for protected and unprotected groups

    Parameters
    ----------
    y_test : numpy.ndarray
        Array containing the true labels
    y_pred : numpy.ndarray
        Array containing the predicted labels
    sensitive_attr : numpy.ndarray
        Array containing the sensitive attribute

    Returns
    -------
    equal_opportunity : float
        Difference between false negative rates for protected and unprotected groups
    """
    fnr_0 = fnr(y_test, y_pred, sensitive_attr, 0)
    fnr_1 = fnr(y_test, y_pred, sensitive_attr, 1)

    return np.abs(fnr_0 - fnr_1)

def evaluate(y_test, y_test_corrected, y_pred_noisy, y_pred_corrected, sensitive_attr):
    """
    Evaluate the performance of the label noise correction method

    Parameters
    ----------
    y_test : numpy.ndarray
        Array containing the true labels
    y_test_corrected : numpy.ndarray
        Array containing the true labels after correction
    y_pred_noisy : numpy.ndarray
        Array containing the predicted labels when trained with noisy labels
    y_pred_corrected : numpy.ndarray
        Array containing the predicted labels when trained with corrected labels
    sensitive_attr : numpy.ndarray
        Array containing the sensitive attribute

    Returns
    -------
    results : pandas.DataFrame
        Dataframe containing the performance metrics for the different combinations of training and test sets
    """

    results = pd.DataFrame(
        data=[
            ['noisy', 'noisy', predictive_equality(y_test, y_pred_noisy, sensitive_attr), equal_opportunity(y_test, y_pred_noisy, sensitive_attr), accuracy_score(y_test, y_pred_noisy)],
            ['noisy', 'corrected', predictive_equality(y_test_corrected, y_pred_noisy, sensitive_attr), equal_opportunity(y_test_corrected, y_pred_noisy, sensitive_attr), accuracy_score(y_test_corrected, y_pred_noisy)],
            ['corrected', 'noisy', predictive_equality(y_test, y_pred_corrected, sensitive_attr), equal_opportunity(y_test, y_pred_corrected, sensitive_attr), accuracy_score(y_test, y_pred_corrected)],
            ['corrected', 'corrected', predictive_equality(y_test_corrected, y_pred_corrected, sensitive_attr), equal_opportunity(y_test_corrected, y_pred_corrected, sensitive_attr), accuracy_score(y_test_corrected, y_pred_corrected)]],
        columns=['Train labels', 'Test labels', 'Predictive Equality', 'Equal Opportunity', 'Accuracy'])

    return results