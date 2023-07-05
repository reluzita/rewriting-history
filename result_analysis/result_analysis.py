import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import imageio

abbv = ['fpr', 'fnr']
names = ['false positive', 'false negative']

colors = {
    'PL':'tab:blue', 
    'STC':'tab:orange', 
    'CC':'tab:green', 
    'HLNC':'tab:red', 
    'OBNC':'tab:purple', 
    'BE':'tab:brown',
    'OBNC-remove-sensitive': 'tab:pink',
    'OBNC-optimize-demographic-parity-0': 'tab:cyan',
    'OBNC-optimize-demographic-parity-0.5': 'darkcyan',
    'OBNC-fair-0': 'tab:gray',
    'OBNC-fair-0.5': 'dimgrey'
}

alg_names = {
    'PL':'PL', 
    'STC':'STC', 
    'CC':'CC', 
    'HLNC':'HLNC', 
    'BE':'BE',
    'OBNC': 'OBNC',
    'OBNC-remove-sensitive': 'Fair-OBNC-rs',
    'OBNC-optimize-demographic-parity-0': 'Fair-OBNC-dp',
    'OBNC-optimize-demographic-parity-0.5': 'Fair-OBNC-dp (prob = 0.5)',
    'OBNC-fair-0': 'Fair-OBNC',
    'OBNC-fair-0.5': 'Fair-OBNC (prob = 0.5)'
}

noise_type_colors = {
    'random': 'tab:blue',
    'flip': 'tab:orange',
    'bias': 'tab:green',
    'balanced_bias': 'tab:red'
}

type_names = {
    'random': 'Random',
    'flip': 'Label Flipping',
    'bias': 'Positive Bias',
    'balanced_bias': 'Balanced bias'
}

pred_metrics = ['accuracy', 'roc_auc']

fair_metrics = [
        # 'auc_difference',
        'equal_opportunity_difference', 
        'predictive_equality_difference',
        'demographic_parity_difference',
        'equalized_odds_difference']

# Noise correction analysis

def show_correction_performance(noise_type, algorithms, experiments, runs, nr):
    fig = plt.figure(figsize=(12, 4))
    axs = fig.subplots(1, 2, sharey=True)

    for i in range(2):
        for alg in algorithms:
            values = []
            for noise_rate in nr:
                avg_value = []
                for exp in experiments:
                    run = runs[noise_type][f'{exp}_{alg}']
                    avg_value.append(run.loc[run['params.noise_rate'] == noise_rate][f'metrics.correction_{abbv[i]}'].mean())
                values.append(np.mean(avg_value))

            axs[i].plot(nr, values, label=alg, c=colors[alg])
        axs[i].set_xlabel('Noise rate')
        axs[i].set_title(f'% of {names[i]} labels after correction')
        axs[i].legend()

    plt.show()

def show_correction_similarity(noise_type, algorithms, experiments, runs, nr):
    fig = plt.figure(figsize=(6, 4))

    for alg in algorithms:
        values = []
        for noise_rate in nr:
            avg_value = []
            for exp in experiments:
                run = runs[noise_type][f'{exp}_{alg}']
                avg_value.append(run.loc[run['params.noise_rate'] == noise_rate][f'metrics.correction_acc'].mean())
            values.append(np.mean(avg_value))

        plt.plot(nr, values, label=alg, c=colors[alg])
    plt.xlabel('Noise rate')
    plt.ylabel('Similarity (% of correct labels)')
    plt.title('Similarity to original labels after correction')
    plt.legend()

    plt.show()

def show_correction_similarity_errors(noise_type, algorithms, experiments, runs, nr):
    fig = plt.figure(figsize=(14, 5))
    axs = fig.subplots(2, 3, sharey=True, sharex=True)


    for i in range(len(algorithms)):
        alg = algorithms[i]
        row = i // 3
        col = i % 3
        values = []
        errors = []
        for noise_rate in nr:
            avg_value = []
            for exp in experiments:
                run = runs[noise_type][f'{exp}_{alg}']
                avg_value.append(run.loc[run['params.noise_rate'] == noise_rate][f'metrics.correction_acc'].mean())
            values.append(np.mean(avg_value))
            errors.append(np.std(avg_value))

        axs[row, col].set_ylim([0, 1])
        axs[row, col].errorbar(nr, values, yerr=errors, label=alg, c=colors[alg])
        if row == 1:
            axs[row, col].set_xlabel('Noise rate')
        if col == 0:
            axs[row, col].set_ylabel('Similarity')
        axs[row, col].set_title(f'{alg}')
    
    plt.suptitle('Similarity to original labels after correction')
    plt.subplots_adjust(wspace=0.07)
    plt.show()

def compare_noise_types(metric, algorithms, noise_types, experiments, runs, nr):
    fig = plt.figure(figsize=(12, 6))
    axs = fig.subplots(2, 3, sharey=True)
    fig.tight_layout(pad=3.0)

    for i in range(6):
        alg = algorithms[i]
        row = i // 3
        col = i % 3
        for noise_type in noise_types:
            values = []
            for noise_rate in nr:
                avg_value = []
                for exp in experiments:
                    run = runs[noise_type][f'{exp}_{alg}']
                    avg_value.append(run.loc[run['params.noise_rate'] == noise_rate][f'metrics.correction_{metric}'].mean())
                values.append(np.mean(avg_value))
            
            axs[row, col].plot(nr, values, label=noise_type)
        axs[row, col].set_xlabel('Noise rate')
        axs[row, col].set_title(f'Noise correction algorithm: {alg}')

    axs[1, 2].legend()
    plt.show()

def compare_correction_similarity(noise_types, algorithms, experiments, runs, nr):
    if len(noise_types) == 3:
        fig = plt.figure(figsize=(16, 4))
        axs = fig.subplots(1, 3, sharey=True)
    else:
        
        fig = plt.figure(figsize=(12, 8))
        axs = fig.subplots(2, 2, sharey=True, sharex=True)

    for i in range(len(noise_types)):
        if len(noise_types) == 3:
            ax = axs[i]
        else:
            row = i // 2
            col = i % 2
            ax = axs[row, col]
        for alg in algorithms:
            values = []
            for noise_rate in nr:
                avg_value = []
                for exp in experiments:
                    run = runs[noise_types[i]][f'{exp}_{alg}']
                    avg_value.append(run.loc[run['params.noise_rate'] == noise_rate][f'metrics.correction_acc'].mean())
                values.append(np.mean(avg_value))
            
            ax.plot(nr, values, label=alg)
        
        if len(noise_types) == 3:
            ax.set_xlabel('Noise rate')         
            if i == 0:
                ax.set_ylabel('Similarity (% of correct labels)')
        else:
            if row == 1:
                ax.set_xlabel('Noise rate')
            if col == 0:
                ax.set_ylabel('Similarity (% of correct labels)')
        ax.set_title(f'Noise type: {noise_types[i]}')

    y_pos = 1.01 if len(noise_types) == 3 else 0.95
    plt.suptitle('Similarity to original labels after correction', fontsize=14, y=y_pos)
    ax.legend()
    plt.subplots_adjust(wspace=0.07, hspace=0.15)
    plt.show()

# Predictive performance comparison

def show_metric(exp, noise_type, test_set, metric, ax, runs, algorithms, nr):
    run = runs[noise_type][f'{exp}_{algorithms[0]}']
    run = run.loc[run['tags.test_set'] == test_set]
    if test_set == 'original':
        ax.plot(nr, [run.loc[(run['params.noise_rate'] == f'{noise_rate}') & (run['tags.train_set'] == 'original')][f'metrics.{metric}'].values[0] for noise_rate in nr], label='original', color='black', linestyle='--', linewidth=2)
    ax.plot(nr, [run.loc[(run['params.noise_rate'] == f'{noise_rate}') & (run['tags.train_set'] == 'noisy')][f'metrics.{metric}'].values[0] for noise_rate in nr], label='noisy', color='r', linestyle='--',linewidth=2)

    for alg in algorithms:
        run = runs[noise_type][f'{exp}_{alg}']
        run = run.loc[run['tags.test_set'] == test_set]
        ax.plot(nr, [run.loc[(run['params.noise_rate'] == f'{noise_rate}') & (run['tags.train_set'] == 'corrected')][f'metrics.{metric}'].values[0] for noise_rate in nr], label=alg)

    ax.set_title(f'{test_set} test set')
    ax.set_xlabel('Noise rate')
    ax.set_ylabel(f'{metric}')

# def show_all_test_sets(exp, noise_type, metric, runs, algorithms):
#     fig = plt.figure(figsize=(20, 4))
#     axs = fig.subplots(1, 3, sharey=True)
#     test_sets = ['original', 'noisy', 'corrected']

#     for i in range(3):
#         show_metric(exp, noise_type, test_sets[i], metric, axs[i], runs, algorithms)

    
#     axs[2].legend(bbox_to_anchor=(1, 1))
#     plt.suptitle(f'{exp} - {metric} comparison ({noise_type} noise)', fontsize=16, y=1.05)
#     plt.show()

def show_metric_aggregated(noise_type, test_set, metric, ax, title, algorithms, experiments, runs, nr, limit=False, thresh=0.5):
    #ax.set_ylim([0, 1])
    results = {alg: {noise_rate: [] for noise_rate in nr} for alg in algorithms}
    results['noisy'] = {noise_rate: [] for noise_rate in nr}
    if test_set == 'original':
        results['original'] = {noise_rate: [] for noise_rate in nr}

    metric_name = f'{metric}_{thresh}' if metric not in ['roc_auc', 'auc_difference'] else metric

    for exp in experiments:
        run = runs[noise_type][f'{exp}_{algorithms[0]}']
        run = run.loc[run['tags.test_set'] == test_set]
        for noise_rate in nr:
            results['noisy'][noise_rate].append(run.loc[(run['params.noise_rate'] == noise_rate) & (run['tags.train_set'] == 'noisy')][f'metrics.{metric_name}'].values[0])
            if test_set == 'original':
                results['original'][noise_rate].append(run.loc[(run['params.noise_rate'] == noise_rate) & (run['tags.train_set'] == 'original')][f'metrics.{metric_name}'].values[0])
        for alg in algorithms:
            run = runs[noise_type][f'{exp}_{alg}']
            run = run.loc[run['tags.test_set'] == test_set]
            for noise_rate in nr:
                results[alg][noise_rate].append(run.loc[(run['params.noise_rate'] == noise_rate) & (run['tags.train_set'] == 'corrected')][f'metrics.{metric_name}'].values[0])

    if test_set == 'original':       
        ax.plot(nr, [np.mean(results['original'][noise_rate]) for noise_rate in nr], label='original', color='black', linestyle='--', linewidth=2)
    ax.plot(nr, [np.mean(results['noisy'][noise_rate]) for noise_rate in nr], label='noisy', color='r', linestyle='--',linewidth=2)

    if len(algorithms) == 1:
        ax.errorbar(nr, [np.mean(results[alg][noise_rate]) for noise_rate in nr], yerr=[np.std(results[alg][noise_rate]) for noise_rate in nr], label=alg, c=colors[alg])
    else:
        for alg in algorithms:
            ax.plot(nr, [np.mean(results[alg][noise_rate]) for noise_rate in nr], label=alg, c=colors[alg])

    ax.set_title(title)
    ax.set_xlabel('Noise rate')
    if limit:
        ax.set_ylim([0, 1])
    #ax.set_ylabel(f'{metric}')

def show_all_metrics_pred(noise_type, test_set, algorithms, experiments, runs, nr, limit=False, thresh=0.5):
    fig = plt.figure(figsize=(14, 4))
    axs = fig.subplots(1, 2, sharey=True)

    for i in range(2):
        show_metric_aggregated(noise_type, test_set, pred_metrics[i], axs[i], f'{pred_metrics[i]}', algorithms, experiments, runs, nr, limit, thresh)

    
    axs[1].legend()
    plt.subplots_adjust(wspace=0.07)
    plt.show()

def show_all_metrics_fair(noise_type, test_set, algorithms, experiments, runs, nr, limit=False, thresh=0.5):
    fig = plt.figure(figsize=(23, 4))
    axs = fig.subplots(1, 4, sharey=True)
    

    for i in range(4):
        show_metric_aggregated(noise_type, test_set, fair_metrics[i], axs[i], f'{fair_metrics[i]}', algorithms, experiments, runs, nr, limit, thresh)

    
    axs[0].legend(bbox_to_anchor=(4.5, 1))
    plt.subplots_adjust(wspace=0.07)
    plt.show()

def show_corrected_test_performance(noise_type, metric, algorithms, experiments, runs, nr, thresh=0.5):
    fig = plt.figure(figsize=(12, 8))
    axs = fig.subplots(3, 2, sharey=True, sharex=True)

    metric_name = f'{metric}_{thresh}' if metric not in ['roc_auc', 'auc_difference'] else metric

    for i in range(6):
        row = i // 2
        col = i % 2
        alg = algorithms[i]

        results = {noise_rate: [] for noise_rate in nr}
        results_original = {noise_rate: [] for noise_rate in nr}

        for exp in experiments:
            run = runs[noise_type][f'{exp}_{alg}']
            run = run.loc[run['tags.test_set'] == 'corrected']
            for noise_rate in nr:
                results[noise_rate].append(run.loc[(run['params.noise_rate'] == noise_rate) & (run['tags.train_set'] == 'corrected')][f'metrics.{metric_name}'].values[0])
            
            run = runs[noise_type][f'{exp}_{alg}']
            run = run.loc[run['tags.test_set'] == 'original']
            for noise_rate in nr:
                results_original[noise_rate].append(run.loc[(run['params.noise_rate'] == noise_rate) & (run['tags.train_set'] == 'corrected')][f'metrics.{metric_name}'].values[0])
            
        axs[row, col].plot(nr, [np.mean(results_original[noise_rate]) for noise_rate in nr], label='original test set', color='black', linestyle='--', linewidth=2)
        axs[row, col].errorbar(nr, [np.mean(results[noise_rate]) for noise_rate in nr], yerr=[np.std(results[noise_rate]) for noise_rate in nr] , label=f'{alg_names[alg]} corrected test set', color=colors[alg])

        if row == 2:
            axs[row, col].set_xlabel('Noise rate')
        if col == 0:
            axs[row, col].set_ylabel(f'{metric}')

        axs[row, col].legend()
        axs[row, col].set_title(f'Train set: {alg} corrected')

    plt.subplots_adjust(wspace=0.07, hspace=0.2)
    plt.suptitle(f'{type_names[noise_type]} noise', fontsize=16, y=0.95)
    plt.show()

def show_noise_types_performance(alg, test_set, metric, ax, nt, experiments, runs, nr, thresh=0.5):
    results = {noise_type: {noise_rate: [] for noise_rate in nr} for noise_type in nt}
    if test_set == 'original':
        results['original'] = {noise_rate: [] for noise_rate in nr}

    metric_name = f'{metric}_{thresh}' if metric not in ['roc_auc', 'auc_difference'] else metric

    for exp in experiments:
        for noise_type in nt:
            run = runs[noise_type][f'{exp}_{alg}']
            run = run.loc[run['tags.test_set'] == test_set]
            for noise_rate in nr:
                results[noise_type][noise_rate].append(run.loc[(run['params.noise_rate'] == noise_rate) & (run['tags.train_set'] == 'noisy')][f'metrics.{metric_name}'].values[0])
        
        if test_set == 'original':
            for noise_rate in nr:
                results['original'][noise_rate].append(run.loc[(run['params.noise_rate'] == noise_rate) & (run['tags.train_set'] == 'original')][f'metrics.{metric_name}'].values[0])
        

    if test_set == 'original':       
        ax.plot(nr, [np.mean(results['original'][noise_rate]) for noise_rate in nr], label='original', color='black', linestyle='--', linewidth=2)
    
    for noise_type in nt:
        ax.plot(nr, [np.mean(results[noise_type][noise_rate]) for noise_rate in nr], label=noise_type, c=noise_type_colors[noise_type])

    ax.set_title(f'Correction algorithm: {alg}')
    # ax.set_title(f'{metric}')
    ax.set_xlabel('Noise rate')
    ax.set_ylabel(f'{metric}')

def show_metric_alg(test_set, metric, algorithms, noise_types, experiments, runs, nr, thresh=0.5):
    fig = plt.figure(figsize=(18, 7))
    axs = fig.subplots(2, 3, sharey=True, sharex=True)

    for i in range(6):
        show_noise_types_performance(algorithms[i], test_set, metric, axs[i // 3, i % 3], noise_types, experiments, runs, nr, thresh)

    
    axs[1, 2].legend(bbox_to_anchor=(1.4, 1.5))
    plt.subplots_adjust(wspace=0.07)
    plt.show()

def show_pred_metrics(test_set, alg, noise_types, experiments, runs, nr, thresh=0.5):
    fig = plt.figure(figsize=(12, 4))
    axs = fig.subplots(1, 2, sharey=True)

    for i in range(2):
        show_noise_types_performance(alg, test_set, pred_metrics[i], axs[i], noise_types, experiments, runs, nr, thresh)

    
    axs[1].legend(bbox_to_anchor=(1, 1))
    plt.subplots_adjust(wspace=0.07)
    plt.show()

def show_fair_metrics(test_set, alg, experiments, runs, nr, thresh=0.5):
    fig = plt.figure(figsize=(20, 4))
    axs = fig.subplots(1, 4, sharey=True)

    for i in range(4):
        show_noise_types_performance(alg, test_set, fair_metrics[i], axs[i], ['bias', 'balanced_bias'], experiments, runs, nr, thresh)

    
    axs[3].legend(bbox_to_anchor=(1, 1))
    plt.subplots_adjust(wspace=0.07)
    plt.show()

# Accuracy/fairness trade offs

def show_trade_off(noise_type, pred_metric, fair_metric, test_set, noise_rate, ax, algorithms, experiments, runs, xlimit=None, ylimit=None, thresh=0.5):
    if test_set == 'original':
        train_sets = ['original', 'noisy']
    else:
        train_sets = ['noisy']

    fair_metric_name = fair_metric if fair_metric == 'auc_difference' else f'{fair_metric}_{thresh}'
    pred_metric_name = pred_metric if pred_metric == 'roc_auc' else f'{pred_metric}_{thresh}'

    for train_set in train_sets:
        predictive_performance = []
        fairness = []
        for exp in experiments:
            run = runs[noise_type][f'{exp}_{algorithms[0]}']
            run = run.loc[(run['tags.test_set'] == test_set) & (run['params.noise_rate'] == noise_rate) & (run['tags.train_set'] == train_set)]
            predictive_performance.append(run[f'metrics.{pred_metric_name}'].values[0])
            fairness.append(run[f'metrics.{fair_metric_name}'].values[0])

        if train_set == 'original':
            c = 'black'
        else:
            c = 'r'
        
        ax.scatter(np.mean(fairness), np.mean(predictive_performance), label=train_set, color=c, marker='x', s=100)
        ax.axvline(x=np.mean(fairness), alpha=0.2, color=c, linestyle='--')
        ax.axhline(y=np.mean(predictive_performance), alpha=0.2, color=c, linestyle='--')
        

    for alg in algorithms:
        predictive_performance = []
        fairness = []
        for exp in experiments:
            run = runs['bias'][f'{exp}_{alg}']
            run = run.loc[(run['tags.test_set'] == test_set) & (run['params.noise_rate'] == noise_rate) & (run['tags.train_set'] == 'corrected')]
            predictive_performance.append(run[f'metrics.{pred_metric_name}'].values[0])
            fairness.append(run[f'metrics.{fair_metric_name}'].values[0])
        ax.scatter(np.mean(fairness), np.mean(predictive_performance), label=alg, s=40, c=colors[alg])

    if len(algorithms) == 1:
        ax.errorbar(np.mean(fairness), np.mean(predictive_performance), xerr=np.std(fairness), yerr=np.std(predictive_performance), c=colors[alg])
    #ax.set_title(f'{test_set} test set')
    ax.set_xlabel(fair_metric)
    ax.set_ylabel(pred_metric)
    if xlimit:
        ax.set_xlim(xlimit)
    if ylimit:
        ax.set_ylim(ylimit)

def show_trade_off_all_metrics(noise_type, noise_rate, test_set, algorithms, experiments, runs, xlimit=None, ylimit=None, thresh=0.5):
    fig = plt.figure(figsize=(25, 8))
    axs = fig.subplots(2, 4, sharey=True, sharex=True)

    for i in range(8):
        row = i // 4
        col = i % 4
        show_trade_off(noise_type, pred_metrics[row], fair_metrics[col], test_set, noise_rate, axs[row, col], algorithms, experiments, runs, xlimit, ylimit, thresh)

    axs[1, 3].legend()
    plt.suptitle(f'Noise rate: {noise_rate}', fontsize=16,y=0.95)
    plt.subplots_adjust(wspace=0.07)

    path = f'plots/{noise_type}_{test_set}_{thresh}'
    if len(algorithms) == 1:
        path += f'_{algorithms[0]}'
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(f'{path}/{int(noise_rate*10)}.png', bbox_inches='tight')

    plt.show()

def create_gif(noise_type, test_set, thresh, nr, alg=None):
    images = []
    path = f'plots/{noise_type}_{test_set}_{thresh}'
    if alg is not None:
        path += f'_{alg}'
    for i in range(1, len(nr)+1):
        images.append(imageio.imread(f'{path}/{i}.png'))
    imageio.mimsave(f'{path}.gif', images, format='GIF', duration=0.3)

def create_trade_off_gif(noise_type, test_set, algorithms, experiments, runs, nr, xlimit=None, ylimit=None, thresh=0.5):
    for noise_rate in nr:
        show_trade_off_all_metrics(noise_type, noise_rate, test_set, algorithms, experiments, runs, xlimit, ylimit, thresh)
        
    if len(algorithms) == 1:
        create_gif(noise_type, test_set, thresh, nr, algorithms[0])
    else:
        create_gif(noise_type, test_set, thresh, nr)


def single_trade_off_gif(noise_type, test_set, algorithms, experiments, runs, nr, xlimit=None, ylimit=None, thresh=0.5):
    for noise_rate in nr:
        show_trade_off_all_metrics(noise_type, noise_rate, test_set, algorithms, experiments, runs, xlimit, ylimit, thresh)
        
    if len(algorithms) == 1:
        create_gif(noise_type, test_set, thresh, nr, algorithms[0])
    else:
        create_gif(noise_type, test_set, nr, thresh)