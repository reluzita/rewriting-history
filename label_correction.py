from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
import pandas as pd
from abc import ABC, abstractmethod
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import BaggingClassifier
import random
import mlflow
import numpy as np
import math
from tqdm import tqdm

CLASSIFIERS = {
    'LogReg': LogisticRegression,
    'DT': DecisionTreeClassifier
}

class LabelCorrectionModel(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def correct(self, X:pd.DataFrame, y:pd.Series) -> pd.Series:
        """
        Corrects the labels of the given dataset

        Parameters
        ----------
        X : pd.DataFrame
            Dataset features
        y : pd.Series
            Labels to correct

        Returns
        -------
        y_corrected: pd.Series
            Corrected labels
        """
        pass

    @abstractmethod
    def log_params(self):
        pass

class SelfTrainingCorrection(LabelCorrectionModel):
    """
    Self-Training Correction algorithm

    Reference:
    Nicholson, Bryce, et al. "Label noise correction methods." 2015 IEEE International Conference on Data Science and Advanced Analytics (DSAA). IEEE, 2015.

    Attributes
    ----------
    classifier
        Type of classifier to use 
    n_folds : int
        Number of folds to use
    correction_rate : float
        Percentage of labels to correct
    """
    def __init__(self, classifier, n_folds, correction_rate):
        self.classifier = classifier
        self.n_folds = n_folds
        self.correction_rate = correction_rate

    def correct(self, X:pd.DataFrame, y:pd.Series):
        original_index = X.index

        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        kf = StratifiedKFold(n_splits=self.n_folds, random_state=42, shuffle=True)
        noisy = set()

        # Split the current training data set using an n-fold cross-validation scheme
        for train_index, test_index in kf.split(X, y):
            X_train = X.loc[train_index]
            X_test = X.loc[test_index]
            y_train = y.loc[train_index]
            y_test = y.loc[test_index]

            # For each of these n parts, a learning algorithm is trained on the other n-1 parts, resulting in n different classifiers
            model = self.classifier(random_state=42).fit(X_train, y_train)

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

        if y_clean.unique().shape[0] == 1:
            print('All selected clean labels are the same, returning original labels')
            return pd.Series(y.values, index=original_index)
        
        if len(noisy) == 0:
            print('All labels are clean, returning original labels')
            return pd.Series(y.values, index=original_index)

        # Build a model from the clean set and uses that to calculate the confidence that each of the instances from the noisy set is mislabeled
        model = self.classifier(random_state=42).fit(X_clean, y_clean)
        y_pred = pd.Series(model.predict(X_noisy), index=noisy)
        y_prob = pd.DataFrame(model.predict_proba(X_noisy), index=noisy, columns=[0, 1])

        corrected = pd.DataFrame(columns=['y_pred', 'y_prob'])
        for i in list(noisy):
            if y_pred.loc[i] != y.loc[i]:
                corrected.loc[i] = [y_pred.loc[i], y_prob.loc[i, y_pred.loc[i]]]

        # The noisy instance with the highest calculated likelihood of belonging to some class that is not equal to its current class 
        # is relabeled to the class that the classifier determined is the instance’s most likely true class. 
        correction_n = int(self.correction_rate*len(corrected))
        y_corrected = y.values
        for i in corrected.sort_values('y_prob', ascending=False)[:correction_n].index:
            y_corrected[i] = corrected.loc[i, 'y_pred']

        return pd.Series(y_corrected, index=original_index)

    def log_params(self):
        mlflow.log_param('correction_alg', 'Self-Training Correction')
        mlflow.log_param('correction_classifier', self.classifier.__name__)
        mlflow.log_param('n_folds', self.n_folds)
        mlflow.log_param('correction_rate', self.correction_rate)

class PolishingLabels(LabelCorrectionModel):
    """ 
    Polishing Labels algorithm

    Reference:
    Nicholson, Bryce, et al. "Label noise correction methods." 2015 IEEE International Conference on Data Science and Advanced Analytics (DSAA). IEEE, 2015.

    Attributes
    ----------
    classifier
        Type of classifier to use
    n_folds : int
        Number of folds to use
    """
    def __init__(self, classifier, n_folds):
        self.classifier = classifier
        self.n_folds = n_folds

    def correct(self, X:pd.DataFrame, y:pd.Series):
        original_index = X.index

        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        kf = StratifiedKFold(n_splits=self.n_folds, random_state=42, shuffle=True)
        
        models = []
        for train_index, _ in kf.split(X, y):
            X_train = X.loc[train_index]
            y_train = y.loc[train_index]

            models.append(self.classifier(random_state=42).fit(X_train.values, y_train.values))

        y_corrected = X.apply(
            lambda x: 0 if np.mean([models[i].predict([x.values])[0] for i in range(self.n_folds)]) < 0.5 else 1, 
            axis=1).to_list()
        
        return pd.Series(y_corrected, index=original_index)
    
    def log_params(self):
        mlflow.log_param('correction_alg', 'Polishing Labels')
        mlflow.log_param('correction_classifier', self.classifier.__name__)
        mlflow.log_param('n_folds', self.n_folds)

class ClusterBasedCorrection(LabelCorrectionModel):
    """
    Cluster-based correction algorithm

    Reference:
    Nicholson, Bryce, et al. "Label noise correction methods." 2015 IEEE International Conference on Data Science and Advanced Analytics (DSAA). IEEE, 2015.

    Attributes
    ----------
    n_iterations : int
        Number of iterations to run
    n_clusters : int
        Number of clusters to use
    """
    def __init__(self, n_iterations, n_clusters):
        self.n_iterations = n_iterations
        self.n_clusters = n_clusters

    def calc_weights(self, cluster_labels:pd.Series, label_dist, n_labels):
        d = [cluster_labels.value_counts().loc[l]/len(cluster_labels) if l in cluster_labels.value_counts().index else 0 for l in range(n_labels)]
        u = 1/n_labels
        multiplier = min(math.log(len(cluster_labels), 10), 2)

        return [multiplier * ((d[l] - u)/label_dist[l]) for l in range(n_labels)]

    def correct(self, X:pd.DataFrame, y:pd.Series):
        original_index = X.index

        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        n_labels = len(y.unique())
        label_totals = [y.value_counts().loc[l]/len(y) for l in range(n_labels)]
        ins_weights = np.zeros((X.shape[0], n_labels))

        if len(X) < self.n_clusters:
            print('Number of samples is less than the number of clusters, using half of the samples as the number of clusters')
            self.n_clusters = int(len(X)/2)

        for i in range(1, self.n_iterations+1):
            k = int((i/self.n_iterations) * self.n_clusters) # on the original paper, the number of clusters varies from 2 to half of the number of samples

            if k == 0:
                k = 2

            C = KMeans(n_clusters=k, random_state=42).fit(X)

            clusters = pd.Series(C.labels_, index=X.index)
            cluster_weights = {c: self.calc_weights(y.loc[clusters == c], label_totals, n_labels) for c in range(k)}
            
            for idx in X.index:
                ins_weights[idx] += cluster_weights[C.labels_[idx]]

        y_corrected = [np.argmax(ins_weights[idx]) for idx in X.index]
        return pd.Series(y_corrected, index=original_index)

    def log_params(self):
        mlflow.log_param('correction_alg', 'Cluster-Based Correction')
        mlflow.log_param('n_iterations', self.n_iterations)
        mlflow.log_param('n_clusters', self.n_clusters)

class HybridLabelNoiseCorrection(LabelCorrectionModel):
    """ 
    Hybrid Label Noise Correction algorithm

    Reference:
    Xu, Jiwei, Yun Yang, and Po Yang. "Hybrid label noise correction algorithm for medical auxiliary diagnosis." 2020 IEEE 18th International Conference on Industrial Informatics (INDIN). Vol. 1. IEEE, 2020.

    Attributes
    ----------
    n_clusters : int
        Number of clusters to use in KMeans clustering
    """
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
    
    def correct(self, X:pd.DataFrame, y:pd.Series):
        original_index = X.index

        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        data = X.copy()
        data['y'] = y

        if len(data) < self.n_clusters:
            print('Number of samples is less than the number of clusters, using half of the samples as the number of clusters')
            self.n_clusters = int(len(data)/2)

        C = KMeans(n_clusters=self.n_clusters, random_state=0).fit(data)
        clusters = pd.Series(C.labels_, index=X.index)
        cluster_labels = [1 if x[-1] > 0.5 else 0 for x in C.cluster_centers_]    

        high_conf = []
        low_conf = []

        for i in data.index:
            if cluster_labels[clusters.loc[i]] == y.loc[i]:
                high_conf.append(i)
            else:
                low_conf.append(i)

        y_corrected = y.copy()

        while len(low_conf) > self.n_clusters:
            # SSK-Means
            seed_set = data.loc[high_conf]

            C = KMeans(n_clusters=self.n_clusters, random_state=0).fit(seed_set)
            cluster_labels = [1 if x[-1] > 0.5 else 0 for x in C.cluster_centers_] 
            centers = np.array([x[:-1] for x in C.cluster_centers_])

            C_ss = KMeans(n_clusters=self.n_clusters, random_state=0, init=centers, n_init=1).fit(X.loc[low_conf])
            y_pred_sskmeans = [cluster_labels[x] for x in C_ss.labels_]

            # Co-training
            # should the feature sets be randomized each iteration?
            features_dt = random.sample(list(X.columns), len(X.columns)//2)
            features_svm = [col for col in X.columns if col not in features_dt]


            if y_corrected.loc[high_conf].unique().shape[0] == 1:
                print('All high confidence labels are the same, co-training will return the same label')
                y_pred_dt = [y_corrected.loc[high_conf].unique()[0]] * len(low_conf)
                y_pred_svm = [y_corrected.loc[high_conf].unique()[0]] * len(low_conf)
            else:      
                dt = DecisionTreeClassifier(random_state=42).fit(X.loc[high_conf, features_dt], y_corrected.loc[high_conf])
                svm = SVC(random_state=42).fit(X.loc[high_conf, features_svm], y_corrected.loc[high_conf])

                y_pred_dt = dt.predict(X.loc[low_conf, features_dt])
                y_pred_svm = svm.predict(X.loc[low_conf, features_svm])

            # Correct labels if classifiers agree, else send back to low confidence data
            for i in range(len(low_conf)):
                if y_pred_dt[i] == y_pred_svm[i] and y_pred_dt[i] == y_pred_sskmeans[i]:
                    high_conf.append(low_conf[i])
                    y_corrected.loc[low_conf[i]] = y_pred_dt[i]
                    
            low_conf = [x for x in low_conf if x not in high_conf]

        return pd.Series(y_corrected.values, index=original_index)
    
    def log_params(self):
        mlflow.log_param('correction_alg', 'Hybrid Label Noise Correction')
        mlflow.log_param('n_clusters', self.n_clusters)

class OrderingBasedCorrection(LabelCorrectionModel):
    """
    Ordering-Based Correction algorithm

    Reference:
    Feng, Wei, and Samia Boukir. "Class noise removal and correction for image classification using ensemble margin." 2015 IEEE International Conference on Image Processing (ICIP). IEEE, 2015.

    Attributes
    ----------
    threshold : float
        Threshold for the margin of the ensemble classifier
    """
    def __init__(self, threshold):
        self.threshold = threshold

    def calculate_margins(self, X, y, bagging:BaggingClassifier):
        margins = pd.Series(dtype=float)
        for i in X.index:
            preds = pd.Series([dt.predict(X.loc[i].values.reshape(1, -1))[0] for dt in bagging.estimators_])
            true_y = y.loc[i]

            if true_y not in preds.unique():
                v_y = 0
            else:
                v_y = preds.value_counts().loc[true_y]
                
            v_c = len(preds) - v_y
            
            margin = (v_y - v_c) / len(preds)
            margins.loc[i] = margin

        return margins

    def correct(self, X:pd.DataFrame, y:pd.Series):
        y_corrected = y.copy()

        bagging = BaggingClassifier(n_estimators=100, random_state=42).fit(X, y)
        y_pred = pd.Series(bagging.predict(X), index=y.index)
        misclassified = [i for i in y.index if y.loc[i] != y_pred.loc[i]]

        margins = self.calculate_margins(X.loc[misclassified], y.loc[misclassified], bagging)
        margins = margins.apply(lambda x: abs(x)).sort_values(ascending=False)

        correct = margins.loc[margins > self.threshold].index
        y_corrected.loc[correct] = y_pred.loc[correct]

        return y_corrected
    
    def log_params(self):
        mlflow.log_param('correction_alg', 'Ordering-Based Correction')
        mlflow.log_param('threshold', self.threshold)


class BayesianEntropy(LabelCorrectionModel):
    """
    Framework for identifying and correcting mislabeled instances

    Reference:
    Sun, Jiang-wen, et al. "Identifying and correcting mislabeled training instances." Future generation communication and networking (FGCN 2007). Vol. 1. IEEE, 2007.

    Attributes
    ----------
    alpha : float
        Noise rate
    n_folds : int
        Number of folds to use
    """
    def __init__(self, alpha, n_folds):
        self.alpha = alpha
        self.n_folds = n_folds

    def evaluate(self, X:pd.DataFrame, y:pd.Series):
        kf = StratifiedKFold(n_splits=self.n_folds, random_state=42, shuffle=True)
        entropy = pd.Series(index=y.index, dtype=float)
        y_pred = pd.Series(index=y.index, dtype=int)

        for train_index, test_index in kf.split(X, y):
            X_train = X.loc[train_index]
            X_test = X.loc[test_index]
            y_train = y.loc[train_index]

            gnb = GaussianNB().fit(X_train, y_train)
            
            y_pred.loc[test_index] = gnb.predict(X_test)

            probs = gnb.predict_proba(X_test)
            entropy.loc[test_index] = [- sum(x * math.log(x, 2) if x != 0 else 0 for x in probs[i]) for i in range(len(probs))]

        return entropy, y_pred
    
    def correct(self, X: pd.DataFrame, y: pd.Series):
        original_index = X.index

        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        entropy, y_pred = self.evaluate(X, y)
        threshold = entropy.sort_values(ascending=True, ignore_index=True)[int(self.alpha*len(entropy))]

        y_corrected = y.copy()

        while True:
            stop = False

            for i in y_corrected.index:
                if entropy.loc[i] < threshold and y_corrected.loc[i] != y_pred.loc[i]:
                    y_corrected.loc[i] = y_pred.loc[i]
                    stop = True

            if stop:
                return pd.Series(y_corrected.values, index=original_index)

            entropy, y_pred = self.evaluate(X, y_corrected)

        
    
    def log_params(self):
        mlflow.log_param('correction_alg', 'BE')
        mlflow.log_param('n_folds', self.n_folds)
        mlflow.log_param('alpha', self.alpha)


def get_label_correction_model(args) -> LabelCorrectionModel:
    """
    Initialize the label correction model

    Parameters
    ----------
    args : argparse.Namespace
        Command line arguments containing the correction algorithm and its parameters

    Returns
    -------
    model: LabelCorrectionModel
        Label correction model
    """
    if args.correction_alg == 'PL':
        return PolishingLabels(CLASSIFIERS[args.base_classifier], args.n_folds)
    elif args.correction_alg == 'STC':
        return SelfTrainingCorrection(CLASSIFIERS[args.base_classifier], args.n_folds, args.correction_rate)
    elif args.correction_alg == 'CC':
        return ClusterBasedCorrection(args.n_iterations, args.n_clusters)
    elif args.correction_alg == 'HLNC':
        return HybridLabelNoiseCorrection(args.n_clusters)
    elif args.correction_alg == 'OBNC':
        return OrderingBasedCorrection(args.threshold)
    elif args.correction_alg == 'BE':
        return BayesianEntropy(args.alpha, args.n_folds)