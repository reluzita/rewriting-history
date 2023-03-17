import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import mlflow
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference, false_positive_rate, false_negative_rate

def predictive_equality_difference(y_true, y_pred, sensitive_attr):
    """
    Predictive equality difference

    Parameters
    ----------
    y_true : pd.Series
        True labels
    y_pred : pd.Series
        Predicted labels
    sensitive_attr : pd.Series
        Sensitive attribute

    Returns
    -------
    float
        Predictive equality difference
    """
    y_true_0 = y_true[sensitive_attr == 0]
    y_pred_0 = y_pred[sensitive_attr == 0]

    y_true_1 = y_true[sensitive_attr == 1]
    y_pred_1 = y_pred[sensitive_attr == 1]

    return abs(fp_rate(y_true_0, y_pred_0) - fp_rate(y_true_1, y_pred_1))

def equal_opportunity_difference(y_true, y_pred, sensitive_attr):
    """
    Equal opportunity difference

    Parameters
    ----------
    y_true : pd.Series
        True labels
    y_pred : pd.Series
        Predicted labels
    sensitive_attr : pd.Series
        Sensitive attribute

    Returns
    -------
    float
        Equal opportunity difference
    """
    y_true_0 = y_true[sensitive_attr == 0]
    y_pred_0 = y_pred[sensitive_attr == 0]

    y_true_1 = y_true[sensitive_attr == 1]
    y_pred_1 = y_pred[sensitive_attr == 1]

    return abs(fn_rate(y_true_0, y_pred_0) - fn_rate(y_true_1, y_pred_1))

def tp_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return tp / (tp + fn)

def fp_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fp / (fp + tn)

def fn_rate(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return fn / (fn + tp)

def eq_odds_difference(y_true, y_pred, sensitive_attr):
    # TPR difference

    tpr_0 = tp_rate(y_true.loc[sensitive_attr == 0], y_pred.loc[sensitive_attr == 0])
    tpr_1 = tp_rate(y_true.loc[sensitive_attr == 1], y_pred.loc[sensitive_attr == 1])
    tpr_diff = abs(tpr_0 - tpr_1)

    # FPR difference
    fpr_0 = fp_rate(y_true.loc[sensitive_attr == 0], y_pred.loc[sensitive_attr == 0])
    fpr_1 = fp_rate(y_true.loc[sensitive_attr == 1], y_pred.loc[sensitive_attr == 1])
    fpr_diff = abs(fpr_0 - fpr_1)

    return max(tpr_diff, fpr_diff)

def evaluate(y_test:pd.Series, y_pred, y_pred_proba, sensitive_attr):
    """
    Calculate and log evaluation metrics to MLflow

    Parameters
    ----------
    y_test : pd.Series
        True labels
    y_pred : pd.Series
        Predicted labels
    sensitive_attr : pd.Series
        Sensitive attribute
    """
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    if len(set(y_pred_proba)) > 1 and len(y_test.unique()) > 1:
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, y_pred_proba))
    mlflow.log_metric("demographic_parity_difference", demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_attr))
    mlflow.log_metric("equalized_odds_difference", eq_odds_difference(y_test, y_pred, sensitive_attr))
    mlflow.log_metric("predictive_equality_difference", predictive_equality_difference(y_test, y_pred, sensitive_attr))
    mlflow.log_metric("equal_opportunity_difference", equal_opportunity_difference(y_test, y_pred, sensitive_attr))

def evaluate_correction(y:pd.Series, y_train_corrected:pd.Series, y_test_corrected:pd.Series, noisy_train_labels, noisy_test_labels):
    original_labels = y.loc[noisy_train_labels + noisy_test_labels].sort_index()
    corrected_labels = pd.concat([y_train_corrected.loc[noisy_train_labels], y_test_corrected.loc[noisy_test_labels]]).sort_index()
    
    acc = accuracy_score(original_labels.values, corrected_labels.values)
    fpr = false_positive_rate(original_labels.values, corrected_labels.values)
    fnr = false_negative_rate(original_labels.values, corrected_labels.values)

    return acc, fpr, fnr