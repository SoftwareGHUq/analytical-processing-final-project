# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from data.make_preprocessing import save_x_and_y_data


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    X_train, X_test, y_train, y_test = save_x_and_y_data(
        input_filepath, output_filepath)
    logger.info(
        f'Data set save and splitted:\nX train length:\n {len(X_train)}, \nX test length:\n {len(X_test)}, \ny train length:\n {len(y_train)}, \ny test length:\n {len(y_test)}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main(f'{project_dir}/data/raw', f'{project_dir}/data/interim')
