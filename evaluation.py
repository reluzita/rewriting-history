import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import mlflow
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference, false_positive_rate, false_negative_rate

def predictive_equality_difference(y_true, y_pred, sensitive_attr):
    y_true_0 = y_true[sensitive_attr == 0]
    y_pred_0 = y_pred[sensitive_attr == 0]

    y_true_1 = y_true[sensitive_attr == 1]
    y_pred_1 = y_pred[sensitive_attr == 1]

    return abs(false_positive_rate(y_true_0, y_pred_0) - false_positive_rate(y_true_1, y_pred_1))

def equal_opportunity_difference(y_true, y_pred, sensitive_attr):
    y_true_0 = y_true[sensitive_attr == 0]
    y_pred_0 = y_pred[sensitive_attr == 0]

    y_true_1 = y_true[sensitive_attr == 1]
    y_pred_1 = y_pred[sensitive_attr == 1]

    return abs(false_negative_rate(y_true_0, y_pred_0) - false_negative_rate(y_true_1, y_pred_1))

def evaluate(y_test, y_pred, sensitive_attr):
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("demographic_parity_difference", demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_attr))
    mlflow.log_metric("equalized_odds_difference", equalized_odds_difference(y_test, y_pred, sensitive_features=sensitive_attr))
    mlflow.log_metric("predictive_equality_difference", predictive_equality_difference(y_test, y_pred, sensitive_attr))
    mlflow.log_metric("equal_opportunity_difference", equal_opportunity_difference(y_test, y_pred, sensitive_attr))

