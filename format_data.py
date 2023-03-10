import openml
import os
import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

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
        data, _, _, _ = openml.datasets.get_dataset(43898).get_data(dataset_format="dataframe")
        data = data.dropna().reset_index(drop=True)    

        #X = data.drop('class', axis=1)
        data['y'] = data['class'].apply(lambda x: 1 if x == '>50K' else 0).astype('int')
        data = data.drop('class', axis=1)
        data = pd.get_dummies(data)
        data = data.drop(['native_country_?', 'workclass_?', 'occupation_?', 'sex_Female'], axis=1)

    elif dataset == 'german':
        data, _, _, _ = openml.datasets.get_dataset(31).get_data(dataset_format="dataframe")

        data['sex_Male'] = data['personal_status'].apply(lambda x: 1 if (x == 'male single' or x == 'male mar/wid' or x == 'male div/sep') else 0)
        data['single'] = data['personal_status'].apply(lambda x: 1 if (x == 'female single' or x == 'male single') else 0)
        data['own_telephone'] = data['own_telephone'].apply(lambda x: 1 if x == 'yes' else 0).astype('int')
        data['foreign_worker'] = data['foreign_worker'].apply(lambda x: 1 if x == 'yes' else 0).astype('int')
        data['y'] = data['class'].apply(lambda x: 1 if x == 'good' else 0).astype('int')
        data = data.drop(['personal_status','class'], axis=1)
        data = pd.get_dummies(data)

    elif dataset == 'compas':
        data, _, _, _ = openml.datasets.get_dataset(45039).get_data(dataset_format="dataframe")

        data['y'] = data['twoyearrecid'].astype('int')
        data = data.drop('twoyearrecid', axis=1)

    elif dataset == 'ricci':
        data, _, _, _ = openml.datasets.get_dataset(42665).get_data(dataset_format="dataframe")

        data['Position_Captain'] = data['Position'].apply(lambda x: 1 if x == 'Captain' else 0).astype('int')
        data['y'] = data['Promotion'].apply(lambda x: 1 if x == 'Promotion' else 0).astype('int')
        data = data.drop(['Promotion', 'Position'], axis=1)
        data = pd.get_dummies(data)

    elif dataset == 'diabetes':
        data, _, _, _ = openml.datasets.get_dataset(43903).get_data(dataset_format="dataframe")

        data['y'] = data['readmit_30_days'].astype('int')
        data = data.drop('readmit_30_days', axis=1)
        for col in ['medicare', 'medicaid', 'had_emergency', 'had_inpatient_days', 'had_outpatient_days']:
            data[col] = data[col].astype('int')
        data['change'] = data['change'].apply(lambda x: 1 if x == 'Ch' else 0).astype('int')
        data['diabetesMed'] = data['diabetesMed'].apply(lambda x: 1 if x == 'Yes' else 0).astype('int')
        data = pd.get_dummies(data)
        data = data.drop(['gender_Female', 'gender_Unknown/Invalid'], axis=1)


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