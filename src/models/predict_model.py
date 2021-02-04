import pickle
import pandas as pd
import click
import logging
import os
from train_model import scaled_fit_transform
from src.features.build_features import pick_up_distance, check_regions, trip_to_cluster, trip_from_cluster


cnt_features = ['trip_distance', 'pick_up_distance',
                'pickup_latitude', 'pickup_longitude']

select_features = ['trip_distance', 'pick_up_distance', 'pickup_latitude', 'pickup_longitude', 'created_weekday', 'created_hour_weekofyear',
                   'created_hour', 'pickup_cluster', 'driver_cluster', 'is_region_diff', 'driver_cluster_count', 'pickup_cluster_count']


def temporal_features(df):
    df['created_weekday'] = df['timestamp'].dt.weekday
    df['created_hour_weekofyear'] = df['timestamp'].dt.weekofyear
    df['created_hour'] = df['timestamp'].dt.hour
    return df


def clusters(df):
    filename = 'src/features/kmeans.sav'
    kmeans = pickle.load(open(filename, 'rb'))
    df['pickup_cluster'] = kmeans.predict(
        df[['pickup_latitude', 'pickup_longitude']])
    df['driver_cluster'] = kmeans.predict(
        df[['driver_latitude', 'driver_longitude']])
    return df


def create_feature_df(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = pick_up_distance(df)
    df = temporal_features(df)
    df = clusters(df)
    df = check_regions(df)
    df = trip_to_cluster(df)
    df = trip_from_cluster(df)
    return df


def create_x(df, feature_col):
    return df[feature_col].values


@click.command()
@click.argument('data_filepath', type=click.Path(exists=True))
@click.argument('model_filepath', type=click.Path())
@click.argument('output_filepath', type=click.Path())
def main(data_filepath, model_filepath, output_filepath):
    """
    Extract, create, scale and transform features.
    Predict probability of driver completing trip.
    """
    logger = logging.getLogger(__name__)

    df = pd.read_csv(os.path.join(data_filepath, 'test_data.csv'))

    logger.info('Extract and create features, scale and transform features')
    df = create_feature_df(df)
    df = scaled_fit_transform(df, cnt_features)
    X = create_x(df, select_features)

    filename = 'model_rcf.sav'
    model_path = os.path.join(model_filepath, filename)
    model_clf = pickle.load(open(model_path, 'rb'))

    logger.info(
        'Predict driver acceptance and completed trip probability, match driver to order')
    df['proba'] = model_clf.predict_proba(X)[:, 1]
    order_driver_match = df.loc[df.groupby('order_id')['proba'].idxmax()][
        ['order_id', 'driver_id']]
    order_driver_match.to_csv(os.path.join(
        output_filepath, 'order_driver_match.csv'), index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()