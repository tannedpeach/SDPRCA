import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_NAMES = ['latency', 'http_status', 'intensity']

DEBUG = True

threshold = 1.0


def anomaly_detection_isolation_forest(df, result_column, cache):
    indices = np.unique(df.index.values)
    for source, target in indices:
        empirical = df.loc[(source, target), FEATURE_NAMES].values
        token = f"IF-{source}-{target}"
        if token not in cache:
            df.loc[(source, target), result_column] = 0
            continue
        model = cache[token]
        predict = model.predict(empirical)
        df.loc[(source, target), result_column] = predict
    return df


# An observation is considered as an outlier if its least squares residual exceeds three times its standard deviation (SD).
def anomaly_detection_3sigma(df, result_column, useful_feature):
    indices = np.unique(df.index.values)
    for source, target in indices:
        if (source, target) not in useful_feature:  # all features are not useful
            df.loc[(source, target), result_column] = 0
            continue
        features = useful_feature[(source, target)]
        empirical = df.loc[(source, target), features].values
        empirical = np.array(empirical).astype(int)
        mean, std = [], []
        for idx, feature in enumerate(features):
            mean.append(np.mean(empirical, axis=0)[idx])
            std.append(np.maximum(np.std(empirical, axis=0)[idx], 0.1))
        mean = np.asarray(mean)
        std = np.asarray(std)
        predict = np.zeros(empirical.shape)
        for idx, feature in enumerate(features):
            predict[:, idx] = np.abs(empirical[:, idx] - mean[idx]) > threshold * std[idx]
        predict = np.max(predict, axis=1)

        df.loc[(source, target), result_column] = predict
    return df


def invo_anomaly_detection_main(input_file, output_file, useful_feature, main_threshold):
    global threshold
    threshold = main_threshold

    with open(useful_feature, 'r') as f:
        useful_feature = eval("".join(f.readlines()))

    input_file = Path(input_file)

    with open(input_file, 'rb') as f:
        df = pickle.load(f)

    df = pd.DataFrame(df)
    df = df.explode(['timestamp', 'latency', 'http_status', 'endtime', 's_t']).reset_index(drop=True)
    df3 = pd.DataFrame(df['s_t'].to_list(), columns=['source', 'target'])
    df = pd.concat([df, df3], axis=1, ignore_index=False, sort=False)
    df = df.set_index(keys=['source', 'target'], drop=False).sort_index()
    df = anomaly_detection_3sigma(df, '3sigma-predict', useful_feature)

    df['predict'] = df['3sigma-predict']
    # print(df.head(4))
    with open(output_file, 'wb+') as f:
        pickle.dump(df, f)


if __name__ == '__main__':
    input_file = r'../data/detection/ip/admin-order_abort_1011.pkl'
    history = r'../data/intensity/op/trace_with_intensity.pkl'
    output_file = r'../data/detection/op/invo_anomaly_detection_2.pkl'
    useful_feature = r'../data/detection/op/useful_feature_2'
    main_threshold = 1

    invo_anomaly_detection_main(input_file=input_file, output_file=output_file,
                                useful_feature=useful_feature,
                                main_threshold=main_threshold)
