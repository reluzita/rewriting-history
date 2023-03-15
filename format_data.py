import openml
import os
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
import random
import numpy as np

def format_adult():
    data, _, _, _ = openml.datasets.get_dataset(43898).get_data(dataset_format="dataframe")
    data = data.dropna().reset_index(drop=True)    

    data['y'] = data['class'].apply(lambda x: 1 if x == '>50K' else 0).astype('int')
    data = data.drop('class', axis=1)
    data = pd.get_dummies(data)
    data = data.drop(['native_country_?', 'workclass_?', 'occupation_?', 'sex_Female'], axis=1)

    return data

def format_german():
    data, _, _, _ = openml.datasets.get_dataset(31).get_data(dataset_format="dataframe")

    data['sex_Male'] = data['personal_status'].apply(lambda x: 1 if (x == 'male single' or x == 'male mar/wid' or x == 'male div/sep') else 0)
    data['single'] = data['personal_status'].apply(lambda x: 1 if (x == 'female single' or x == 'male single') else 0)
    data['own_telephone'] = data['own_telephone'].apply(lambda x: 1 if x == 'yes' else 0).astype('int')
    data['foreign_worker'] = data['foreign_worker'].apply(lambda x: 1 if x == 'yes' else 0).astype('int')
    data['y'] = data['class'].apply(lambda x: 1 if x == 'good' else 0).astype('int')
    data = data.drop(['personal_status','class'], axis=1)
    data = pd.get_dummies(data)

    return data

def format_compas():
    data, _, _, _ = openml.datasets.get_dataset(45039).get_data(dataset_format="dataframe")

    data['y'] = data['twoyearrecid'].astype('int')
    data = data.drop('twoyearrecid', axis=1)

    return data

def format_ricci():
    data, _, _, _ = openml.datasets.get_dataset(42665).get_data(dataset_format="dataframe")

    data['Position_Captain'] = data['Position'].apply(lambda x: 1 if x == 'Captain' else 0).astype('int')
    data['y'] = data['Promotion'].apply(lambda x: 1 if x == 'Promotion' else 0).astype('int')
    data = data.drop(['Promotion', 'Position'], axis=1)
    data = pd.get_dummies(data)

    return data

def format_diabetes():
    data, _, _, _ = openml.datasets.get_dataset(43903).get_data(dataset_format="dataframe")

    data['y'] = data['readmit_30_days'].astype('int')
    data = data.drop('readmit_30_days', axis=1)
    for col in ['medicare', 'medicaid', 'had_emergency', 'had_inpatient_days', 'had_outpatient_days']:
        data[col] = data[col].astype('int')
    data['change'] = data['change'].apply(lambda x: 1 if x == 'Ch' else 0).astype('int')
    data['diabetesMed'] = data['diabetesMed'].apply(lambda x: 1 if x == 'Yes' else 0).astype('int')
    data = pd.get_dummies(data)
    data = data.drop(['gender_Female', 'gender_Unknown/Invalid'], axis=1)

    return data

def format_phishing():
    data, _, _, _ = openml.datasets.get_dataset(4534).get_data(dataset_format="dataframe")

    data = data.astype(int)
    for col in data.columns:
        if data[col].value_counts().shape[0] == 2:
            data[col] = data[col].apply(lambda x: 1 if x == 1 else 0)

    # for i in data.loc[data['having_IP_Address'] == 0].index:
    #     if random.random() < 0.5:
    #         data.loc[i, 'Result'] = 1 - data.loc[i, 'Result']

    data['y'] = data['Result']
    data = data.drop('Result', axis=1)

    return data

def format_titanic():
    data, _, _, _ = openml.datasets.get_dataset(40945).get_data(dataset_format="dataframe")

    data = data.drop(['cabin', 'ticket', 'boat', 'body', 'home.dest'], axis=1)
    data['sex'] = data['sex'].apply(lambda x: 1 if x == 'male' else 0).astype('int')
    data['title'] = data['name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    data = data.drop(columns='name')
    data['title'] = data['title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Countess', 'Sir', 'Jonkheer', 'Lady', 'Capt', 'Don'], 'Others')
    data['title'] = data['title'].map({'Ms': 'Miss', 'Mme': 'Mrs', 'Mlle': 'Miss'})
    data['embarked'] = data['embarked'].fillna('S')
    NaN_indexes = data['age'][data['age'].isnull()].index
    for i in NaN_indexes:
        pred_age = data['age'][((data.sibsp == data.iloc[i]["sibsp"]) & (data.parch == data.iloc[i]["parch"]) & (data.pclass == data.iloc[i]["pclass"]))].median()
        if not np.isnan(pred_age):
            data['age'].iloc[i] = pred_age
        else:
            data['age'].iloc[i] = data['age'].median()
    for i, row in data.loc[data['fare'].isna()].iterrows():
        data.loc[i, 'fare'] = data.loc[data['pclass'] == data.loc[i, 'pclass']].fare.median()
    data['y'] = data['survived'].astype('int')
    data = data.drop('survived', axis=1)

    data = pd.get_dummies(data)

    return data

def format_bank():
    data, _, _, _ = openml.datasets.get_dataset(1461).get_data(dataset_format="dataframe")

    data = data.rename(columns={
        "V1": "age","V2": "job","V3": "marital","V4": "education","V5": "default","V6": "balance","V7": "housing","V8": "loan","V9": "contact",
        "V10": "day","V11": "month","V12": "duration","V13": "campaign","V14": "pdays","V15": "previous","V16": "poutcome", "V17": "y"
    })

    data['education'] = data['education'].map({'unknown': np.nan, 'primary':1, 'secondary':2, 'tertiary':3}).astype(float)
    data = data.dropna()

    for c in ['default', 'housing', 'loan']:
        data[c] = data[c].map({'no': 0, 'yes': 1}).astype(int)

    data['month'] = data['month'].map({'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}).astype(int)
    data['y'] = data['Class'].astype(int).apply(lambda x: 1 if x == 2 else 0).astype(int)
    data = data.drop(columns=['Class'])

    data = pd.get_dummies(data)
    data = data.drop(columns=['job_unknown', 'marital_single', 'contact_unknown', 'poutcome_unknown'])

    # for i in data.loc[data['housing'] == 1].index:
    #     if random.random() < 0.5:
    #         data.loc[i, 'y'] = 1 - data.loc[i, 'y']

    return data

def get_data(dataset):
    """
    Get dataset from OpenML, format it for the experiments and save it to a csv file

    Parameters
    ----------
    dataset : str
        Name of the dataset to get

    Returns
    -------
    X : pandas.DataFrame
        Dataframe containing the features
    y : pandas.Series
        Series containing the labels
    """
    if os.path.exists(f'data/{dataset}.csv'):
        data = pd.read_csv(f'data/{dataset}.csv')
        X = data.drop('y', axis=1)
        y = data['y']

        return X, y
    
    if dataset == 'adult':
        data = format_adult() 
    elif dataset == 'german':
        data = format_german()
    elif dataset == 'compas':
        data = format_compas()
    elif dataset == 'ricci':
        data = format_ricci()
    elif dataset == 'diabetes':
        data = format_diabetes()
    elif dataset == 'phishing':
        data = format_phishing()
    elif dataset == 'titanic':
        data = format_titanic()
    elif dataset == 'bank':
        data = format_bank()

    X = data.drop('y', axis=1)
    y = data['y']

    if y.value_counts()[0]/len(y) < 0.2 or y.value_counts()[1]/len(y) < 0.2:
        print('Dataset is too unbalanced, performing random undersampling')
        X, y = RandomUnderSampler(sampling_strategy=0.5, random_state=42).fit_resample(X, y)
        data = X.copy()
        data['y'] = y

    if not os.path.exists('data'):
        os.mkdir('data')
    data.to_csv(f'data/{dataset}.csv', index=False)

    return X, y