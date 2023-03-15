import pandas as pd
import random

def inject_noise(y:pd.Series, group:pd.Series, noise_rate, noise_type):
    y_noisy = y.copy()

    changed_labels = []
    for i in group.loc[group == 1].index:
        if random.random() < noise_rate:
            if noise_type == 'flip':
                y_noisy.loc[i] = 1 - y.loc[i] 
            elif noise_type == 'bias': 
                y_noisy.loc[i] = 1

            changed_labels.append(i)
    
    return y_noisy, changed_labels
