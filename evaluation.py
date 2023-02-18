import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import mlflow
from fairlearn.metrics import demographic_parity_difference, equalized_odds_difference

def evaluate(y_test, y_pred, sensitive_attr):
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("demographic_parity_difference", demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_attr))
    mlflow.log_metric("equalized_odds_difference", equalized_odds_difference(y_test, y_pred, sensitive_features=sensitive_attr))