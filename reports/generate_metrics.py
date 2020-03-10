import pandas as pd
import numpy as np
import click
import os
import logging


def generate_conversion_rate(df, experiment_df):
    """
    Calculates the booking conversion rate for each experiment.
    Defined as number of completed bookings divided by the number
    of created bookings.
    """
    created_df = df[df['booking_status'] == 'CREATED'][
        ['order_id', 'booking_status']]
    completed_df = df[df['booking_status'] == 'COMPLETED'][
        ['order_id', 'booking_status']]
    created_exp = experiment_df.merge(created_df[['order_id']], on='order_id')
    completed_exp = experiment_df.merge(
        completed_df[['order_id']], on='order_id')
    result = completed_exp.groupby('experiment_tag').count(
    ) / created_exp.groupby('experiment_tag').count()
    result = result[['order_id']].rename(
        columns={"order_id": "conversion_rate"})
    return result


def generate_mean_pick_up_time(df, experiment_df):
    """
    Calculates the mean pick up time for each experiment.
    Defined as the difference between the booking creating time
    and the pickup time in minutes.
    """
    df = _duration(df, 'DRIVER_FOUND', 'PICKED_UP', 'mean_pickup')
    df = experiment_df.merge(df[['order_id', 'mean_pickup']], on='order_id')
    df.drop_duplicates(['order_id'],  inplace=True)
    result = df.groupby('experiment_tag').mean()[['mean_pickup']]
    return result


def generate_driver_acceptance_rate(participant_df, experiment_df):
    """
    Calculates the driver acceptance rate for each experiment.
    Defined as number of accepted orders divided by the number
    of created bookings.
    """
    created_df = participant_df[participant_df[
        'participant_status'] == 'CREATED']
    accepted_df = participant_df[participant_df[
        'participant_status'] == 'ACCEPTED']
    created_exp = experiment_df.merge(created_df[['order_id']], on='order_id')
    accepted_exp = experiment_df.merge(
        accepted_df[['order_id']], on='order_id')
    result = accepted_exp.groupby('experiment_tag').count(
    ) / created_exp.groupby('experiment_tag').count()
    result = result[['order_id']].rename(
        columns={"order_id": "driver_acceptance_rate"})
    return result


def generate_customer_cancellation_rate(df, experiment_df):
    """
    Calculates the customer cancellation rate for each experiment.
    Defined as number of customer cancelled orders divided by total orders.
    """
    df_exp = df.merge(
        experiment_df[['order_id', 'experiment_tag']], on='order_id')
    cust_cancelled = df_exp[df_exp['booking_status'] == 'CUSTOMER_CANCELLED']
    unique_orders = df_exp.drop_duplicates(['order_id']).groupby('experiment_tag').count()
    result = cust_cancelled.groupby('experiment_tag').count() / unique_orders
    result = result[['order_id']].rename(
        columns={"order_id": "customer_cancellation_rate"})
    return result


def generate_mean_drop_off_time(df, experiment_df):
    """
    Calculates the mean drop off time for each experiment.
    Defined as the difference between the booking picked up time
    and the completed time in minutes.
    """
    df = _duration(df, 'PICKED_UP', 'COMPLETED', 'mean_dropoff')
    df = experiment_df.merge(df[['order_id', 'mean_dropoff']], on='order_id')
    df.drop_duplicates(['order_id'],  inplace=True)
    result = df.groupby('experiment_tag').mean()[['mean_dropoff']]
    return result


def _duration(df, start, end, name):
    """
    Calculates the time diff between two booking statuses for each order_id.

    Parameters
    ----------
        df: DataFrame
            Booking logs dataframe.

        start: str
            First booking status.

        end: str
            Second booking status.

        name: str
            Name for duration column.

    Returns
    -------
        df: Dataframe with new column:

        df.name: datetime64
            Time difference between two booking statuses for each order_id. 
    """
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['time_diff'] = df.sort_values(['order_id', 'timestamp']).groupby('order_id')[
        'timestamp'].diff()
    g = df.groupby('order_id')
    duration = g.apply(lambda x: sum(x[x['booking_status'] == start]['time_diff'], x[
                       x['booking_status'] == end]['time_diff']) / np.timedelta64(1, 'm'))
    duration_df = duration.reset_index()
    df = df.merge(duration_df, on='order_id')
    df.drop(['level_1'], axis=1, inplace=True)
    df.rename(columns={"time_diff_y": name}, inplace=True)
    return df


@click.command()
@click.argument('dataset_filepath', type=click.Path(exists=True))
@click.argument('participant_filepath', type=click.Path(exists=True))
@click.argument('experiment_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(dataset_filepath, participant_filepath, experiment_filepath, output_filepath):
    """ 
    Runs data processing scripts to turn interim data from (../interim) into
    cleaned data (saved in ../processed).
    """
    logger = logging.getLogger(__name__)

    df = pd.read_csv(os.path.join(dataset_filepath, 'dataset.csv'))
    participant_df = pd.read_csv(os.path.join(
        participant_filepath, 'participant_log.csv'))
    experiment_df = pd.read_csv(os.path.join(
        experiment_filepath, 'experiment_log.csv'))

    logger.info('calculating conversation rate metric')
    df1 = generate_conversion_rate(df, experiment_df)
    logger.info('calculating mean pick up time metric')
    df2 = generate_mean_pick_up_time(df, experiment_df)
    logger.info('calculating driver acceptance rate metric')
    df3 = generate_driver_acceptance_rate(participant_df, experiment_df)
    logger.info('calculating customer_cancellation_rate')
    df4 = generate_customer_cancellation_rate(df, experiment_df)

    metrics = pd.concat([df1, df2, df3, df4], axis=1)
    metrics.to_csv(os.path.join(output_filepath, 'metrics.csv'))
    logger.info('created all metrics and saved file')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()