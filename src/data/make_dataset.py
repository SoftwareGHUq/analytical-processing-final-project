# -*- coding: utf-8 -*-
# libraries to import function from other folder
from dotenv import find_dotenv, load_dotenv
from pathlib import Path
import logging
import click
from make_preprocessing import process


def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    data_clean = process(input_filepath, output_filepath)
    logger.info(f'Dataset save: {len(data_clean)}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]
    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())
    input_url = f'{project_dir}\data\\raw'
    output_url = f'{project_dir}\data\interim'
    print(input_url)
    print(output_url)
    main(input_url, output_url)
