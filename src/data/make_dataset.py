import click
import pandas as pd
import logging
import os
from src.data.clean_raw import save_csv, read_csv, generate_order_ids


def create_label_accept(df_bkg, df_pnt, grp_bkg, booking_status, label):
    """
    Creates label column on dataframe:
        Assigns Boolean 1: Driver accepts, order completed or
        Assigns Boolean 0: Driver accepts, customer cancels order.

    Parameters
    ----------
        df_bkg: DataFrame
            Booking logs dataframe.

        df_pnt: DataFrame
            Participant logs dataframe.

        grp_bkg: GroupBy
            Grouped booking log dataframe.

        booking_status: list
            List of booking status combinations.

        label: Bool
        	1 or 0 bool value to assign to each row
 
    Returns
    -------
        df: Dataframe
            Dataframe with column:

        df.label: Bool
        	Boolean assigned to each row
    """
    pnt_columns = ['order_id', 'driver_id',
                   'driver_latitude', 'driver_longitude']
    order_ids = generate_order_ids(grp_bkg, 'booking_status', booking_status)
    df_pnt = df_pnt[df_pnt['participant_status'] == "ACCEPTED"][pnt_columns]
    df_bkg = df_bkg[df_bkg['order_id'].isin(order_ids)]
    merge_on = 'order_id'
    df = df_bkg.merge(df_pnt, how='left', on=merge_on,
                      suffixes=('_bkg', '_pnt'))
    df['label'] = label
    return df


def create_label_nonaccept(df_bkg, df_pnt, grp_bkg):
    """
    Creates label column on dataframe:
        Assigns Boolean 0: Driver does not accept order, 
        with associated booking status.

    Parameters
    ----------
        df_bkg: DataFrame
            Booking logs dataframe.

        df_pnt: DataFrame
            Participant logs dataframe.

        grp_bkg: GroupBy
            A grouped booking log dataframe.

    Returns
    -------
        df: Dataframe
            Dataframe with column:

        df.label: Bool
        	Boolean assigned to each row
    """
    pnt_columns = ['order_id', 'driver_id',
                   'driver_latitude', 'driver_longitude']
    booking_status = [['CREATED', 'COMPLETED', 'PICKED_UP', 'DRIVER_FOUND'], [
        'CREATED', 'DRIVER_FOUND', 'CUSTOMER_CANCELLED'], ['CREATED', 'PICKED_UP', 'DRIVER_FOUND', 'CUSTOMER_CANCELLED']]
    order_ids = generate_order_ids(grp_bkg, 'booking_status', booking_status)
    df_pnt = df_pnt[df_pnt['participant_status'] != "ACCEPTED"][pnt_columns]
    df_bkg = df_bkg[~df_bkg['order_id'].isin(order_ids)]
    merge_on = 'order_id'
    df = df_bkg.merge(df_pnt, how='left', on=merge_on,
                      suffixes=('_bkg', '_pnt'))
    df['label'] = 0
    return df


def unmatching_driver_id(df):
	"""
	Removes `order_ids` where `driver_ids` from booking logs 
	do not match with `driver_ids` from participant logs

    Parameters
    ----------
        df: DataFrame
            A dataframe where booking merged with participant dfs
            and had columns:

        df.driver_id_bkg: int
            driver_id from booking df.

        df.driver_id_pnt: int
            driver_id from participant df.

    Returns
    -------
        df: Dataframe
            Dataframe with rows that have mismatched `driver_id`
    """
	temp = df[df['driver_id_bkg'].notnull()]
	order_ids = temp[temp['driver_id_bkg'] != temp['driver_id_pnt']]['order_id'].values
	return df[~df['order_id'].isin(order_ids)]


def create_dataset(df_bkg, df_pnt, grp_bkg):
    """
    Creates dataset to be used for classifier model. 
    Contains `label` column that assigns each row boolean 1 or 0.
    """
    # drivers accepted completed trip label 1
    a = [['CREATED', 'COMPLETED', 'PICKED_UP', 'DRIVER_FOUND']]
    df_a = create_label_accept(df_bkg, df_pnt, grp_bkg, a, 1)
    df_a = df_a[df_a['driver_id_pnt'].notnull()]

    # drivers accepted customer cancelled label 0
    b = [['CREATED', 'DRIVER_FOUND', 'CUSTOMER_CANCELLED']]
    df_b = create_label_accept(df_bkg, df_pnt, grp_bkg, b, 0)
    df_b = df_b[df_b['driver_id_pnt'].notnull()]

    # drivers accepted customer cancelled label 0
    c = [['CREATED', 'PICKED_UP', 'DRIVER_FOUND', 'CUSTOMER_CANCELLED']]
    df_c = create_label_accept(df_bkg, df_pnt, grp_bkg, c, 0)
    df_c = df_c[df_c['driver_id_pnt'].notnull()]

    # driver does not accept
    df_d = create_label_nonaccept(df_bkg, df_pnt, grp_bkg)
    df_d = df_d[df_d['driver_id_pnt'].notnull()]

    dataset = pd.concat([df_a, df_b, df_c, df_d],
                        sort=False, ignore_index=True)
    dataset = dataset.drop_duplicates(
        ['order_id', 'driver_id_pnt', 'booking_status'])

    dataset = unmatching_driver_id(dataset)
    print (len(dataset))
    return dataset


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ 
    Runs make_dataset script to turn interim data from (../interim) into
    processed dataset ready for modelling (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('creating dataset for modelling')

    df_bkg = read_csv(input_filepath, 'booking_log.csv')
    df_pnt = read_csv(input_filepath, 'participant_log.csv')

    grp_bkg = df_bkg.groupby('order_id')

    dataset = create_dataset(df_bkg, df_pnt, grp_bkg)

    logger.info('dataset created and saving file')
    save_csv(dataset, output_filepath, 'dataset.csv')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
