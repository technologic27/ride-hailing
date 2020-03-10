import click
import pandas as pd
import logging
import os


error_booking_status = [
    ['DRIVER_CANCELLED', 'PICKED_UP'],
    ['COMPLETED', 'PICKED_UP', 'DRIVER_FOUND'],
    ['PICKED_UP', 'CUSTOMER_CANCELLED'],
    ['CREATED', 'COMPLETED'],
    ['CREATED'],
    ['CREATED', 'COMPLETED', 'DRIVER_FOUND'],
    ['CREATED', 'PICKED_UP', 'DRIVER_FOUND'],
    ['CREATED', 'DRIVER_FOUND'],
    ['DRIVER_CANCELLED'],
    ['DRIVER_CANCELLED', 'CREATED', 'PICKED_UP', 'DRIVER_FOUND'],
    ['DRIVER_CANCELLED', 'PICKED_UP'],
    ['PICKED_UP', 'CUSTOMER_CANCELLED'],
    ['CREATED', 'CUSTOMER_CANCELLED'],
    ['PICKED_UP', 'DRIVER_FOUND', 'CUSTOMER_CANCELLED']
]

error_participant_status = [['CREATED'], [
    'IGNORED'], ['ACCEPTED'], ['REJECTED']]


def get_timestamp(df):
    """
    Converts `event_timestamp` to `timestamp` in datetime64
    and drops rows with null `timestamp` values.

    Parameters
    ----------
        df: DataFrame
            A dataframe with column:

        df.event_timestamp: object

    Returns
    -------
        df: Dataframe
            Dataframe with column:

        df.timestamp: datetime64.
    """
    df[['date', 'time', 'utc']] = df['event_timestamp'].str.split(expand=True)
    df.drop(["utc", 'event_timestamp'], axis=1, inplace=True)
    df["timestamp"] = df[["date", "time"]].astype(
        str).apply(lambda x: ' '.join(x), axis=1)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
    df = df[df['timestamp'].notnull()]
    return df


def df_deduplicate(df, columns):
    """
    Deduplicates df on columns specified.

    Parameters
    ----------
        df: DataFrame
            A dataframe df that contains columns specified.
        columns: list
            List of names of columns to deduplicate on.

    Returns
    -------
        df: Dataframe
            Dataframe with deduplicated rows
    """
    return df.drop_duplicates(subset=columns)


def generate_order_ids(group, field, combi_list):
    """
    Generates a list of `order_id` based on a list of 
    participant status or booking status combinations.

    Parameters
    ----------
        group: GroupBy
            A grouped dataframe.
        field: str
            Dataframe column to check combinations.
        combi_list: list
            List of participant status or booking status
            combinations.

    Returns
    -------
        order_ids: list
            List of order_ids to be removed.
    """
    order_ids = []
    for idx, df in group:
        for item in combi_list:
            if set(df[field].values) == set(item):
                order_ids.append(df['order_id'].values[0])
    return order_ids


def remove_order_ids(df, order_ids):
    """
    Removes rows of a list of order_ids from df.

    Parameters
    ----------
        df: DataFrame
            A dataframe.
        order_id: list
            A list of order_ids.

    Returns
    -------
        df: DataFrame
            Dataframe without rows that have removed order_ids.
    """
    return df[~df['order_id'].isin(order_ids)]


def read_csv(input_filepath, filename):
    filepath = os.path.join(input_filepath, filename)
    return pd.read_csv(filepath)


def save_csv(df, output_filepath, filename):
    filepath = os.path.join(output_filepath, filename)
    df.to_csv(filepath, index=False)


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ 
    Runs clean_raw to turn raw data from (../raw) into
    cleaned data (saved in ../interim).
    """
    logger = logging.getLogger(__name__)

    bkg_raw = read_csv(input_filepath, 'booking_log.csv')
    pnt_raw = read_csv(input_filepath, 'participant_log.csv')
    tst_raw = read_csv(input_filepath, 'test_data.csv')
    exp_raw = read_csv(input_filepath, 'experiment_log.csv')

    logger.info('changing timestamp and deduplicating booking data from ../raw')
    bkg_raw = get_timestamp(bkg_raw)
    bkg_raw = df_deduplicate(bkg_raw, ['order_id', 'booking_status'])

    logger.info(
        'changing timestamp and deduplicating participant data from ../raw')
    pnt_raw = get_timestamp(pnt_raw)
    pnt_raw = df_deduplicate(
        pnt_raw, ['order_id', 'driver_id', 'participant_status'])

    logger.info('changing timestamp and deduplicating test data from ../raw')
    tst_raw = get_timestamp(tst_raw)
    tst_raw = df_deduplicate(tst_raw, ['order_id', 'driver_id'])

    logger.info(
        'changing timestamp and deduplicating experiment data from ../raw')
    exp_raw = get_timestamp(exp_raw)
    exp_raw = df_deduplicate(exp_raw, ['order_id'])

    logger.info('completed changing timestamp and deduplicating on all datasets')

    logger.info('removing blacklist order ids')
    bkg_group = bkg_raw.groupby('order_id')
    rm_bkg_order_ids = generate_order_ids(
        bkg_group, 'booking_status', error_booking_status)

    pnt_group = pnt_raw.groupby(['driver_id', 'order_id'])
    rm_pnt_order_ids = generate_order_ids(
        pnt_group, 'participant_status', error_participant_status)

    black_list_ids = []
    for _id in rm_pnt_order_ids:
        black_list_ids.append(_id)
    black_list_ids = set(black_list_ids)

    cleaned_bkg_df = remove_order_ids(bkg_raw, black_list_ids)

    cleaned_pnt_df = remove_order_ids(pnt_raw, black_list_ids)

    cleaned_exp_df = remove_order_ids(exp_raw, black_list_ids)

    logger.info('saving dataframes')
    save_csv(cleaned_bkg_df, output_filepath, 'booking_log.csv')
    save_csv(cleaned_pnt_df, output_filepath, 'participant_log.csv')
    save_csv(tst_raw, output_filepath, 'test_data.csv')
    save_csv(cleaned_exp_df, output_filepath, 'experiment_log.csv')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
