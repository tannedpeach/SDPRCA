import pickle
from collections import defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

FEATURE_NAMES = ['latency','http_status', 'intensity']

DEBUG = False  # very slow


def distribution_criteria(empirical, reference, threshold):
    empirical, reference = np.array(empirical), np.array(reference)
    historical_mean, historical_std = np.mean(reference), np.std(reference)
    ref_ratio = sum(np.abs(reference - historical_mean) > 3 * historical_std) / reference.shape[0]
    emp_ratio = sum(np.abs(empirical - historical_mean) > 3 * historical_std) / empirical.shape[0]
    return (emp_ratio - ref_ratio) > threshold * ref_ratio


def fisher_criteria(empirical, reference, side='two-sided'):
    if side == 'two-sided':
        diff_mean = (np.abs(np.mean(empirical) - np.mean(reference)) ** 2)
    elif side == 'less':
        diff_mean = np.maximum(np.mean(empirical) - np.mean(reference), 0) ** 2
    elif side == 'greater':
        diff_mean = np.maximum(np.mean(reference) - np.mean(empirical), 0) ** 2
    else:
        raise RuntimeError(f'invalid side: {side}')
    variance = np.maximum(np.var(empirical) + np.var(reference), 0.1)
    return diff_mean / variance


def stderr_criteria(empirical, reference, threshold):
    empirical, reference = np.array(empirical), np.array(reference)
    historical_mean, historical_std = np.mean(reference), np.std(reference)
    historical_std = np.maximum(historical_std, historical_mean * 0.01 + 0.01)
    ref_ratio = np.mean(np.abs(reference - historical_mean)) / historical_std
    emp_ratio = np.mean(np.abs(empirical - historical_mean)) / historical_std
    return (emp_ratio - ref_ratio) > threshold * ref_ratio


def selecting_feature_main(input_file: str, output_file: str, history: str, fisher_threshold):
    input_file = Path(input_file)
    output_file = Path(output_file)
    with open(history, 'rb') as f:
        history = pickle.load(f)
    with open(str(input_file), 'rb') as f:
        df = pickle.load(f)
    # df = df[:100]
    df = pd.DataFrame(df)

    df = df.explode(['timestamp', 'latency', 'http_status', 'endtime', 's_t']).reset_index(drop=True)
    df3 = pd.DataFrame(df['s_t'].to_list(), columns=['source', 'target'])
    df = pd.concat([df, df3], axis=1, ignore_index=False, sort=False)
    df.dropna(subset=['source', 'target'])
    history.dropna(subset=['source', 'target'])
    df = df.set_index(keys=['source', 'target'], drop=True).sort_index()
    df['http_status'] = pd.to_numeric(df['http_status'])
    history['http_status'] = pd.to_numeric(history['http_status'])
    history['source'] = history['source'].astype(str)
    history['target'] = history['target'].astype(str)
    history = history.set_index(keys=['source', 'target'], drop=True).sort_index()
    indices = np.intersect1d(np.unique(df.index.values), np.unique(history.index.values))
    useful_features_dict = defaultdict(list)
    if DEBUG:
        plot_dir = output_file.parent / 'selecting_feature.debug'
        plot_dir.mkdir(exist_ok=True)
    for (source, target), feature in tqdm(product(indices, FEATURE_NAMES)):
        empirical = np.sort(df.loc[(source, target), feature].values)
        reference = np.sort(history.loc[(source, target), feature].values)
        fisher = stderr_criteria(empirical, reference, 0.5)
        if fisher:
            useful_features_dict[(source, target)].append(feature)
    with open(output_file, 'w+') as f:
        print(dict(useful_features_dict), file=f)


if __name__ == '__main__':
    input_file = r'D:/sdprca-main/data/detection/ip/admin-order_abort_1011.pkl'
    history = r'D:/sdprca-main/data/intensity/op/trace_with_intensity.pkl'
    output_file = r'D:/sdprca-main/data/detection/op/useful_feature_2'
    fisher_threshold = 1
    selecting_feature_main(input_file = input_file,output_file = output_file,history = history,fisher_threshold = fisher_threshold)



