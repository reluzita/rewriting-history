import openml
import os
import pandas as pd

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

        X = data[['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']]
        X['sex'] = [1 if x == 'Male' else 0 for x in data['sex']]
        y = data['class'].apply(lambda x: 1 if x == '>50K' else 0)
        
    data = X.copy()
    data['y'] = y
    if not os.path.exists('data'):
        os.mkdir('data')
    data.to_csv(f'data/{dataset}.csv', index=False)

    return X, y