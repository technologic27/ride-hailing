import numpy as np
import pandas as pd
import click
import logging
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split
import json
import pickle
import os


select_features = ['trip_distance', 'pick_up_distance', 'pickup_latitude', 'pickup_longitude', 'created_weekday', 'created_hour_weekofyear',
                   'created_hour', 'pickup_cluster', 'driver_cluster', 'is_region_diff', 'driver_cluster_count', 'pickup_cluster_count']

cat_features = ['created_weekday', 'created_hour_weekofyear',
                'created_hour', 'pickup_cluster', 'driver_cluster', 'is_region_diff']

cnt_features = ['trip_distance', 'pick_up_distance',
                'pickup_latitude', 'pickup_longitude']


def create_x_y(df, feature_col):
    X = df[feature_col].values
    y = df['label'].values
    return X, y


def scaled_fit_transform(df, cnt_col):
    ss = StandardScaler()
    scaled_features = df.copy()
    features = scaled_features[cnt_col]
    features_scaled = ss.fit_transform(features.values)
    scaled_features[cnt_col] = features_scaled
    return scaled_features


def create_train_test_model(df, cnt_col, feature_col):
    scaled_features = scaled_fit_transform(df, cnt_col)
    X, y = create_x_y(scaled_features, feature_col)
    rus = RandomUnderSampler(random_state=42)
    X, y = rus.fit_sample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    return X_train, y_train, X_test, y_test


def load_rfc_estimator(params):
    n_estimators = params['n_estimators']
    max_depth = params['max_depth']
    min_samples_leaf = params['min_samples_leaf']
    max_features = params['max_features']
    estimator = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                       min_samples_leaf=min_samples_leaf, max_features=max_features, n_jobs=-1)
    return estimator


def train_model(X_train, y_train, X_test, y_test, estimator):
    model = estimator.fit(X_train, y_train)
    y_pred_train = estimator.predict(X_train)
    y_pred_test = estimator.predict(X_test)
    acc_train = metrics.accuracy_score(y_train, y_pred_train)
    acc_test = metrics.accuracy_score(y_test, y_pred_test)
    return model, acc_train, acc_test, y_pred_train, y_pred_test


def train_cv_model(X, y, estimator):
    models = []
    acc_train = []
    acc_test = []
    y_test = []
    y_pred = []

    kf = KFold(n_splits=5)
    for train_index, test_index in kf.split(X):
        X_train_cv, X_test_cv = X[train_index], X[test_index]
        y_train_cv, y_test_cv = y[train_index], y[test_index]
        estimator.fit(X_train_cv, y_train_cv)
        models.append(estimator)
        y_pred_train = estimator.predict(X_train_cv)
        y_pred_test = estimator.predict(X_test_cv)
        acc_train.append(metrics.accuracy_score(y_train_cv, y_pred_train))
        acc_test.append(metrics.accuracy_score(y_test_cv, y_pred_test))
    index = values.index(max(acc_test))
    return models[index], acc_train[index], acc_test[index], y_test[index], y_pred[index]


@click.command()
@click.argument('data_filepath', type=click.Path(exists=True))
@click.argument('parameter_filepath', type=click.Path())
@click.argument('model_output_filepath', type=click.Path())
def main(data_filepath, parameter_filepath, model_output_filepath):

    df = pd.read_csv(os.path.join(data_filepath, 'features.csv'))

    X_train, y_train, X_test, y_test = create_train_test_model(
        df, cnt_features, select_features)

    with open(os.path.join(parameter_filepath, 'model_rfc.json')) as json_file:
        params = json.load(json_file)

    estimator = load_rfc_estimator(params)

    model, acc_train, acc_test, y_pred_train, y_pred_test = train_model(
        X_train, y_train, X_test, y_test, estimator)

    rfc_results = {"acc_train": acc_train, "acc_test": acc_test,
                   "y_test": y_test.tolist(), "y_pred_test": y_pred_test.tolist(),
                   "y_train": y_train.tolist(), "y_pred_train": y_pred_train.tolist()}

    with open(os.path.join(model_output_filepath, 'rfc_results.json'), 'w') as json_file:
        json.dump(rfc_results, json_file)

    model_path = os.path.join(model_output_filepath, 'model_rfc.sav')

    pickle.dump(model, open(model_path, 'wb'))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
