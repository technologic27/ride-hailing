import pandas as pd
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import click
import logging
import os
import pickle

kmeans_cluster = 40


def compute_distances(d_lat, d_lng, p_lat, p_lng):
    """
    Calculate the distance of driver latitude and longitude to
    pickup latitude and longitude using the great circle distance (haversine)

    Parameters
    ----------
        d_lat: float
            Driver latitude.

        d_lng: float
            Driver longitude.

        p_lat: float
            Pick up latitude.

        p_lng: float
            Pick up longitude.

    Returns
    -------
        c: float
            Haversine distance in km from driver to pick up latitude
            and longitude.
    """
    rad_lat, rad_lng, rad_pt_lat, rad_pt_lng = map(
        np.radians, [d_lat, d_lng, p_lat, p_lng]
    )
    dlng = rad_lng - rad_pt_lng
    dlat = rad_lat - rad_pt_lat
    a = np.sin(dlat / 2.0) ** 2 + np.cos(rad_lat) * np.cos(rad_pt_lat) \
        * np.sin(dlng / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return c


def pick_up_distance(df):
    """
    Generated pick up distance and creates a column in dateframe to
    store pick up distance per order id and given driver id.

    Parameters
    ----------
        df: DataFrame
            A dataframe with columns:

        df.driver_latitude: float
            Latitude of driver location.

        df.driver_longitude: float
            Longitude of driver location.

        df.pickup_latitude: float
            Latitude of pick up location.

        df.pickup_longitude: float
            Longitude of pick up location.

    Returns
    -------
        df: Dataframe
            Dataframe with new `pick_up_distance` column.
    """
    df['pick_up_distance'] = compute_distances(df['driver_latitude'], df[
                                               'driver_longitude'], df['pickup_latitude'], df['pickup_longitude'])
    return df


def temporal_features(df):
    """
    Creates temporal features for order_ids on condition when
    `booking_status` is 'CREATED'.

    Parameters
    ----------
        df: DataFrame
            A dataframe with columns:

        df.booking_status: str
            Booking status for a order.

        df.timestamp: datetime64
            Timestamp

    Returns
    -------
        df: Dataframe
            Dataframe with new columns:

        df.created_weekday: int
            Weekday of order created

        df.created_hour_weekofyear: int
            Hour week of year of order created

        df.created_hour: int
            Hour of order created
    """
    df_created = df[df['booking_status'] == 'CREATED']
    df_created['created_weekday'] = df_created['timestamp'].dt.weekday
    df_created['created_hour_weekofyear'] = df_created[
        'timestamp'].dt.weekofyear
    df_created['created_hour'] = df_created['timestamp'].dt.hour
    columns = ['order_id', 'created_weekday',
               'created_hour_weekofyear', 'created_hour']
    df = df.merge(df_created[columns], on='order_id')
    return df


def clusters(df, no_clusters):
    """
    Assigns each order_id a cluster for pickup location and
    driver location respectively.

    Parameters
    ----------
        df: DataFrame
            A dataframe with columns:

        df.driver_latitude: float
            Latitude of driver location.

        df.driver_longitude: float
            Longitude of driver location.

        df.pickup_latitude: float
            Latitude of pick up location.

        df.pickup_longitude: float
            Longitude of pick up location.

        no_clusters: int
            Number of clusters to generate for assignment

    Returns
    -------
        df: Dataframe with new columns:

        df.pickup_cluster: int
            Cluster assigned to order_id pickup location

        df.driver_cluster: int
            Cluster assigned to order_id driver location
    """
    coords = np.vstack((df[['pickup_latitude', 'pickup_longitude']].values, df[
                       ['driver_latitude', 'driver_longitude']].values))
    sample_ind = np.random.permutation(len(coords))[:500000]
    kmeans = MiniBatchKMeans(
        n_clusters=no_clusters, batch_size=10000).fit(coords[sample_ind])
    filename = 'kmeans.sav'
    pickle.dump(kmeans, open(filename, 'wb'))
    df['pickup_cluster'] = kmeans.predict(
        df[['pickup_latitude', 'pickup_longitude']])
    df['driver_cluster'] = kmeans.predict(
        df[['driver_latitude', 'driver_longitude']])
    return df


def check_regions(df):
    """
    Checks each order_id, if pickup cluster is different from driver cluster. 
        Assigns Boolean 1: Cluster is different.
        Assigns Boolean 0: Cluster is not different.

    Parameters
    ----------
        df: DataFrame
            A dataframe with columns:

        df.pickup_cluster: int
            Cluster of order pickup location.

        df.driver_longitude: int
            Cluster of order driver location.

    Returns
    -------
        df: Dataframe
            Dataframe with new column:

        df.is_region_diff: bool
            Boolean assigned to each order_id if pickup and driver location is different.
    """
    df['is_region_diff'] = np.where(
        df['pickup_cluster'] != df['driver_cluster'], 1, 0)
    return df


def _compute_freq_cluster(df):
    """
    Computes pick up datetime group and assigns each order_id to the group.

    Parameters
    ----------
        df: DataFrame
            A Datraframe with columns:

        df.order_id: int
            Order_id.

        df.timestamp: datetime64
            Timestamp of each order_id

        df.pickup_cluster: int
            Pickup cluster order_id belongs to.

        df.driver_cluster: int
            Driver cluster order_id belongs to.

    Returns
    -------
        df: Dataframe
            Dataframe with new column:

        df.pickup_datetime_group:
            Pick up datetime group for each order_id

        df_cluster: Dataframe
            A subset of `df` with columns:

        df_cluster:.order_id: int
            Order_id.

        df_cluster.timestamp: datetime64
            Timestamp of each order_id

        df_cluster.pickup_cluster: int
            Pickup cluster order_id belongs to.

        df_cluster.driver_cluster: int
            Driver cluster order_id belongs to.

        group_freq: str
            Group datetime accorindg group_freq
    """
    group_freq = '60min'
    df_cluster = df[['order_id', 'timestamp',
                     'pickup_cluster', 'driver_cluster']]
    df['pickup_datetime_group'] = df['timestamp'].dt.round(group_freq)
    return df, df_cluster, group_freq


def trip_to_cluster(df):
    """
    Computes the number of trips going into each pickup cluster and assigns value to each order_id

    Parameters
    ----------
        df: DataFrame
            A Datraframe with columns:

        df.order_id: int
            Order_id.

        df.timestamp: datetime64
            Timestamp of each order_id

        df.pickup_cluster: int
            Pickup cluster order_id belongs to.

        df.driver_cluster: int
            Driver cluster order_id belongs to.

    Returns
    -------
        df: Dataframe
            A dataframe with new column:

        df.driver_cluster_count: float
            Number of trips going into pickup cluster
    """
    df, df_cluster, group_freq = _compute_freq_cluster(df)
    df_counts = df_cluster \
        .set_index('timestamp') \
        .groupby([pd.Grouper(freq=group_freq), 'driver_cluster']) \
        .agg({'order_id': 'count'}) \
        .reset_index().set_index('timestamp') \
        .groupby('driver_cluster').rolling('240min').mean() \
        .drop('driver_cluster', axis=1) \
        .reset_index().set_index('timestamp').shift(freq='-120min').reset_index() \
        .rename(columns={'timestamp': 'pickup_datetime_group', 'order_id': 'driver_cluster_count'})
    df['driver_cluster_count'] = df[['pickup_datetime_group', 'driver_cluster']].merge(
        df_counts, on=['pickup_datetime_group', 'driver_cluster'], how='left')['driver_cluster_count'].fillna(0)
    return df


def trip_from_cluster(df):
    """
    Computes the number of trips coming from each pickup cluster and assigns value to each order_id

    Parameters
    ----------
        df: DataFrame
            A Datraframe with columns:

        df.order_id: int
            Order_id.

        df.timestamp: datetime64
            Timestamp of each order_id

        df.pickup_cluster: int
            Pickup cluster order_id belongs to.

        df.driver_cluster: int
            Driver cluster order_id belongs to.

    Returns
    -------
        df: Dataframe
            A dataframe with new column:

        df.pickup_cluster_count: float
            Number of trips coming from pickup cluster
    """
    df, df_cluster, group_freq = _compute_freq_cluster(df)
    df_counts = df_cluster \
        .set_index('timestamp') \
        .groupby([pd.Grouper(freq=group_freq), 'pickup_cluster']) \
        .agg({'order_id': 'count'}) \
        .reset_index().set_index('timestamp') \
        .groupby('pickup_cluster').rolling('240min').mean() \
        .drop('pickup_cluster', axis=1) \
        .reset_index().set_index('timestamp').shift(freq='-120min').reset_index() \
        .rename(columns={'timestamp': 'pickup_datetime_group', 'order_id': 'pickup_cluster_count'})
    df['pickup_cluster_count'] = df[['pickup_datetime_group', 'pickup_cluster']].merge(
        df_counts, on=['pickup_datetime_group', 'pickup_cluster'], how='left')['pickup_cluster_count'].fillna(0)
    return df


def create_feature_df(df, no_clusters):
    """
    Creates feature dataframe used for classifier model.
    """
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df[~df['driver_id_pnt'].isnull()]
    df = pick_up_distance(df)
    df = temporal_features(df)
    df = clusters(df, no_clusters)
    df = check_regions(df)
    df = trip_to_cluster(df)
    df = trip_from_cluster(df)
    df.drop_duplicates(['order_id'], inplace=True)
    df.drop(['driver_id_bkg', 'booking_status'], axis=1, inplace=True)
    return df


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """
    Runs build_feature script retrieve dataset (../processed) 
    and transform into a feature dataset use for modelling (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('creating dataset with features')
    dataset = pd.read_csv(os.path.join(input_filepath, 'dataset.csv'))
    df = create_feature_df(dataset, kmeans_cluster)
    df.to_csv(os.path.join(output_filepath, 'features.csv'), index=False)
    logger.info('features dataset created')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
